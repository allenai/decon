use crate::review::ContaminationResult;
use anyhow::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Normalize eval suite name by removing shard suffixes like "_train-1", "_test-2", etc.
/// This ensures proper aggregation across sharded datasets.
fn normalize_eval_suite_name(name: &str) -> String {
    // Pattern: _{split}-{number} where split is train/test/validation/dev
    // Examples: drop_train-1 -> drop, mmlu_test-2 -> mmlu, jeopardy_train-10 -> jeopardy
    let splits = ["_train-", "_test-", "_validation-", "_dev-"];
    
    for split_pattern in &splits {
        if let Some(pos) = name.rfind(split_pattern) {
            // Check if what follows is a number
            let after_split = &name[pos + split_pattern.len()..];
            if after_split.chars().all(|c| c.is_ascii_digit()) {
                return name[..pos].to_string();
            }
        }
    }
    
    // If no standard split pattern found, check for just -{number} at the end
    // This handles cases like "bbeh_causal_understanding-1"
    if let Some(pos) = name.rfind('-') {
        let after_dash = &name[pos + 1..];
        if !after_dash.is_empty() && after_dash.chars().all(|c| c.is_ascii_digit()) {
            // Only remove if the part before the dash doesn't look like it's part of the dataset name
            // Be conservative: only remove if there's an underscore before the dash
            if pos > 0 && name[..pos].contains('_') {
                return name[..pos].to_string();
            }
        }
    }
    
    // No normalization needed
    name.to_string()
}

/// Display evaluation dataset statistics with bar charts
pub fn display_eval_dataset_stats(contamination_results: &[ContaminationResult], output_dir: Option<&Path>) -> Result<(), Error> {
    // Count unique training documents per eval suite
    let mut training_docs_per_suite: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in contamination_results {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let raw_eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        
        // Normalize to remove shard suffixes
        let eval_suite = normalize_eval_suite_name(&raw_eval_suite);

        // Track unique training documents for this suite
        training_docs_per_suite
            .entry(eval_suite)
            .or_default()
            .insert((result.training_file.clone(), result.training_line));
    }

    // Convert to counts
    let mut eval_counts: HashMap<String, usize> = HashMap::new();
    for (suite, docs) in training_docs_per_suite {
        eval_counts.insert(suite, docs.len());
    }

    // Sort by count (descending)
    let mut sorted_counts: Vec<(String, usize)> = eval_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    // Compute metrics for header display
    let metrics = compute_contamination_metrics(contamination_results);

    // Check if we should write to CSV or display
    if let Some(dir) = output_dir {
        // Create output directory if it doesn't exist
        fs::create_dir_all(dir)?;
        
        // Write summary metrics
        write_summary_csv(dir, &metrics)?;
        
        // Write training docs contaminated by suite
        write_training_docs_csv(dir, &sorted_counts)?;
        
        println!("Exported statistics to: {}", dir.display());
        println!("  - summary.csv");
        println!("  - training_docs_by_suite.csv");
    } else {
        // Display to console
        println!("=== CONTAMINATION STATISTICS ===");
        println!();
        println!("Summary:");
        println!("  Training docs contaminated: {}", metrics.training_docs_contaminated);
        println!("  Total contamination instances: {}", metrics.contamination_instances);
        println!("  Unique eval instances: {}", metrics.contaminated_evals);
        println!();
        println!("=== TRAINING DOCUMENTS CONTAMINATED BY EVAL SUITE ===");
        println!("(Each count represents unique training documents that need removal)");
        println!();

        // Find the maximum count for scaling the bar chart
        let max_count = sorted_counts.first().map(|(_, count)| *count).unwrap_or(0);
        let bar_width = 50; // Width of the bar chart in characters

        // Display each eval suite with a horizontal bar chart
        for (suite, count) in &sorted_counts {
            // Calculate bar length proportional to count
            let bar_length = if max_count > 0 {
                ((*count as f64 / max_count as f64) * bar_width as f64) as usize
            } else {
                0
            };

            // Create the bar using Unicode block characters
            let bar = "█".repeat(bar_length);
            let empty = " ".repeat(bar_width - bar_length);

            // Format the output with aligned columns
            println!("  {:<45} {:>8} │{}{}│", suite, count, bar, empty);
        }

        println!();
        println!();
    }

    // Count unique eval instances
    let mut unique_eval_counts: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in contamination_results {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let raw_eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        
        // Normalize to remove shard suffixes
        let eval_suite = normalize_eval_suite_name(&raw_eval_suite);

        // Track unique (normalized_dataset, eval_instance) pairs per suite
        // IMPORTANT: Normalize the dataset name to avoid counting the same instance
        // multiple times across shards (e.g., instance 5 in both suite-1 and suite-2
        // should only be counted once)
        let eval_instance_key = result.eval_instance_index.unwrap_or(result.eval_line);
        let normalized_dataset = normalize_eval_suite_name(&result.eval_dataset);
        let unique_key = (normalized_dataset, eval_instance_key);
        unique_eval_counts
            .entry(eval_suite)
            .or_default()
            .insert(unique_key);
    }

    // Convert to counts of unique instances per suite
    let mut unique_sorted_counts: Vec<(String, usize)> = unique_eval_counts
        .into_iter()
        .map(|(suite, instances)| (suite, instances.len()))
        .collect();
    unique_sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    if let Some(dir) = output_dir {
        // Write eval instances by suite
        write_eval_instances_csv(dir, &unique_sorted_counts)?;
        println!("  - eval_instances_by_suite.csv");
        
        // Write n-gram distribution
        display_ngram_match_distribution(contamination_results, Some(dir))?;
        println!("  - ngram_distribution.csv");
    } else {
        println!("=== CONTAMINATED EVAL INSTANCES BY SUITE ===");
        println!("(Unique eval examples found in training data)");
        println!();

        // Find the maximum count for scaling the bar chart
        let max_unique_count = unique_sorted_counts.first().map(|(_, count)| *count).unwrap_or(0);
        let bar_width = 50; // Width of the bar chart in characters

        // Display each eval suite with a horizontal bar chart
        for (suite, count) in &unique_sorted_counts {
            // Calculate bar length proportional to count
            let bar_length = if max_unique_count > 0 {
                ((*count as f64 / max_unique_count as f64) * bar_width as f64) as usize
            } else {
                0
            };

            // Create the bar using Unicode block characters
            let bar = "█".repeat(bar_length);
            let empty = " ".repeat(bar_width - bar_length);

            // Format the output with aligned columns
            println!("  {:<45} {:>8} │{}{}│", suite, count, bar, empty);
        }

        println!();
        println!();

        // Display n-gram match distribution
        display_ngram_match_distribution(contamination_results, None)?;
    }

    Ok(())
}

/// Display n-gram match count histogram
fn display_ngram_match_distribution(contamination_results: &[ContaminationResult], output_dir: Option<&Path>) -> Result<(), Error> {
    println!("=== PROMPT N-GRAM MATCH DISTRIBUTION ===");
    println!("(Counts unique n-gram matches from evaluation prompts/questions only)");
    println!("(Answer and passage matches are tracked separately)");
    println!();

    // Collect n-gram match counts
    let mut ngram_counts: Vec<usize> = Vec::new();
    let mut missing_count = 0;

    for result in contamination_results {
        if let Some(count) = result.ngram_match_cnt {
            ngram_counts.push(count);
        } else {
            missing_count += 1;
        }
    }

    if ngram_counts.is_empty() {
        println!("No prompt n-gram match data available.");
        return Ok(());
    }

    // Calculate statistics
    ngram_counts.sort();
    let min_count = *ngram_counts.first().unwrap();
    let max_count = *ngram_counts.last().unwrap();
    let median_count = if ngram_counts.len().is_multiple_of(2) {
        (ngram_counts[ngram_counts.len() / 2 - 1] + ngram_counts[ngram_counts.len() / 2]) / 2
    } else {
        ngram_counts[ngram_counts.len() / 2]
    };
    let avg_count = ngram_counts.iter().sum::<usize>() as f64 / ngram_counts.len() as f64;

    println!(
        "Contamination instances with prompt match data: {}",
        ngram_counts.len()
    );
    if missing_count > 0 {
        println!("Instances without prompt match data: {}", missing_count);
    }
    println!();
    println!("Prompt n-gram match statistics:");
    println!("  Minimum matches:    {}", min_count);
    println!("  Maximum matches:    {}", max_count);
    println!("  Median matches:     {}", median_count);
    println!("  Average matches:    {:.1}", avg_count);
    println!();

    // Create buckets for histogram
    let buckets: Vec<(usize, usize, &str)> = vec![
        (1, 5, "1-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, usize::MAX, "500+"),
    ];

    // Count matches in each bucket
    let mut bucket_counts: Vec<(String, usize)> = Vec::new();
    for (min, max, label) in &buckets {
        let count = ngram_counts
            .iter()
            .filter(|&&c| c >= *min && c <= *max)
            .count();
        if count > 0 {
            bucket_counts.push((label.to_string(), count));
        }
    }

    // Find the maximum bucket count for scaling
    let max_bucket_count = bucket_counts
        .iter()
        .map(|(_, count)| *count)
        .max()
        .unwrap_or(0);

    if let Some(dir) = output_dir {
        // Write n-gram distribution to CSV
        write_ngram_distribution_csv(dir, &bucket_counts, min_count, max_count, median_count, avg_count)?;
    } else {
        let bar_width = 50; // Width of the bar chart in characters

        println!("Distribution of prompt n-gram matches per contamination instance:");
        println!();

        // Display histogram, calculate bar length proportional to count
        for (label, count) in &bucket_counts {
            let bar_length = if max_bucket_count > 0 {
                ((*count as f64 / max_bucket_count as f64) * bar_width as f64) as usize
            } else {
                0
            };

            let bar = "█".repeat(bar_length);
            let empty = " ".repeat(bar_width - bar_length);

            println!("  {:<15} {:>8} │{}{}│", label, count, bar, empty);
        }

        println!();
        println!();
    }

    Ok(())
}

/// Write summary metrics to CSV
fn write_summary_csv(dir: &Path, metrics: &ContaminationMetrics) -> Result<(), Error> {
    let path = dir.join("summary.csv");
    let mut file = File::create(path)?;
    
    writeln!(file, "metric,value")?;
    writeln!(file, "training_docs_contaminated,{}", metrics.training_docs_contaminated)?;
    writeln!(file, "total_contamination_instances,{}", metrics.contamination_instances)?;
    writeln!(file, "unique_eval_instances,{}", metrics.contaminated_evals)?;
    
    Ok(())
}

/// Write training documents contaminated by suite to CSV
fn write_training_docs_csv(dir: &Path, counts: &[(String, usize)]) -> Result<(), Error> {
    let path = dir.join("training_docs_by_suite.csv");
    let mut file = File::create(path)?;
    
    writeln!(file, "eval_suite,training_docs_contaminated")?;
    for (suite, count) in counts {
        writeln!(file, "{},{}", suite, count)?;
    }
    
    Ok(())
}

/// Write eval instances by suite to CSV
fn write_eval_instances_csv(dir: &Path, counts: &[(String, usize)]) -> Result<(), Error> {
    let path = dir.join("eval_instances_by_suite.csv");
    let mut file = File::create(path)?;
    
    writeln!(file, "eval_suite,unique_eval_instances")?;
    for (suite, count) in counts {
        writeln!(file, "{},{}", suite, count)?;
    }
    
    Ok(())
}

/// Write n-gram distribution to CSV
fn write_ngram_distribution_csv(
    dir: &Path,
    bucket_counts: &[(String, usize)],
    min_count: usize,
    max_count: usize,
    median_count: usize,
    avg_count: f64,
) -> Result<(), Error> {
    let path = dir.join("ngram_distribution.csv");
    let mut file = File::create(path)?;
    
    // Write stats first
    writeln!(file, "statistic,value")?;
    writeln!(file, "min_matches,{}", min_count)?;
    writeln!(file, "max_matches,{}", max_count)?;
    writeln!(file, "median_matches,{}", median_count)?;
    writeln!(file, "avg_matches,{:.1}", avg_count)?;
    writeln!(file)?;
    
    // Write distribution
    writeln!(file, "match_range,contamination_instances")?;
    for (label, count) in bucket_counts {
        writeln!(file, "{},{}", label, count)?;
    }
    
    Ok(())
}

/// Struct to hold the three distinct contamination metrics
#[derive(Debug)]
pub struct ContaminationMetrics {
    pub training_docs_contaminated: usize,  // Unique (training_file, training_line) tuples
    pub contamination_instances: usize,      // Total number of report lines
    pub contaminated_evals: usize,           // Unique (eval_dataset, eval_line) tuples
}

/// Helper function to compute all three contamination metrics
pub fn compute_contamination_metrics(results: &[ContaminationResult]) -> ContaminationMetrics {
    let mut unique_training_docs = HashSet::new();
    let mut unique_eval_instances = HashSet::new();

    for result in results {
        unique_training_docs.insert((result.training_file.clone(), result.training_line));

        // Track unique eval instances
        // Use eval_instance_index if available, otherwise fall back to eval_line
        let eval_instance_key = result.eval_instance_index.unwrap_or(result.eval_line);
        unique_eval_instances.insert((result.eval_dataset.clone(), eval_instance_key));
    }

    ContaminationMetrics {
        training_docs_contaminated: unique_training_docs.len(),
        contamination_instances: results.len(),
        contaminated_evals: unique_eval_instances.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result(
        training_file: &str,
        training_line: usize,
        eval_dataset: &str,
        eval_line: usize,
        eval_instance_index: Option<usize>,
    ) -> ContaminationResult {
        ContaminationResult {
            training_file: training_file.to_string(),
            training_line,
            eval_dataset: eval_dataset.to_string(),
            eval_key: None,
            eval_line,
            eval_instance_index,
            split: None,
            method: None,
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: None,
            eval_overlap_text: None,
            ngram_match_cnt: None,
            eval_unique_ngrams: None,
            contamination_score: None,
            length_penalty: None,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_start_idx: None,
            answer_end_idx: None,
            passage_start_idx: None,
            passage_end_idx: None,
            idf_overlap: None,
            cluster_token_length: None,
            eval_token_length: None,
            token_length_delta: None,
            ngram_jaccard: None,
            length_adjusted_question_threshold: None,
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            eval_question_text: None,
            eval_answer_text: None,
            eval_passage_text: None,
            fingerprint: None,
            is_correct: None,
            reference_file: None,
        }
    }

    #[test]
    fn test_compute_contamination_metrics_basic() {
        let results = vec![
            create_test_result("file1.txt", 10, "mmlu", 100, None),
            create_test_result("file1.txt", 20, "mmlu", 200, None),
            create_test_result("file2.txt", 30, "gsm8k", 300, None),
        ];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 3);
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 3);
    }

    #[test]
    fn test_compute_contamination_metrics_duplicate_training_docs() {
        let results = vec![
            create_test_result("file1.txt", 10, "mmlu", 100, None),
            create_test_result("file1.txt", 10, "mmlu", 200, None),
            create_test_result("file1.txt", 10, "gsm8k", 300, None),
        ];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 1);
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 3);
    }

    #[test]
    fn test_compute_contamination_metrics_duplicate_eval_instances() {
        let results = vec![
            create_test_result("file1.txt", 10, "mmlu", 100, None),
            create_test_result("file2.txt", 20, "mmlu", 100, None),
            create_test_result("file3.txt", 30, "mmlu", 100, None),
        ];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 3);
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 1);
    }

    #[test]
    fn test_compute_contamination_metrics_with_eval_instance_index() {
        let results = vec![
            create_test_result("file1.txt", 10, "mmlu", 100, Some(1)),
            create_test_result("file2.txt", 20, "mmlu", 200, Some(1)),
            create_test_result("file3.txt", 30, "mmlu", 300, Some(2)),
        ];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 3);
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 2);
    }

    #[test]
    fn test_compute_contamination_metrics_empty() {
        let results = vec![];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 0);
        assert_eq!(metrics.contamination_instances, 0);
        assert_eq!(metrics.contaminated_evals, 0);
    }

    #[test]
    fn test_compute_contamination_metrics_mixed_instance_index() {
        let results = vec![
            create_test_result("file1.txt", 10, "mmlu", 100, Some(1)),
            create_test_result("file2.txt", 20, "mmlu", 200, None),
            create_test_result("file3.txt", 30, "gsm8k", 300, Some(3)),
        ];

        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 3);
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 3);
    }

    #[test]
    fn test_normalize_eval_suite_name_with_split_and_shard() {
        assert_eq!(normalize_eval_suite_name("drop_train-1"), "drop");
        assert_eq!(normalize_eval_suite_name("mmlu_test-2"), "mmlu");
        assert_eq!(normalize_eval_suite_name("jeopardy_train-10"), "jeopardy");
        assert_eq!(normalize_eval_suite_name("gsm8k_validation-5"), "gsm8k");
        assert_eq!(normalize_eval_suite_name("squad_dev-3"), "squad");
    }

    #[test]
    fn test_normalize_eval_suite_name_with_shard_only() {
        assert_eq!(normalize_eval_suite_name("bbeh_causal_understanding-1"), "bbeh_causal_understanding");
        assert_eq!(normalize_eval_suite_name("super_gpqa_train-8"), "super_gpqa");
        assert_eq!(normalize_eval_suite_name("medmcqa_train-25"), "medmcqa");
    }

    #[test]
    fn test_normalize_eval_suite_name_no_normalization() {
        // Names without shard numbers should remain unchanged
        assert_eq!(normalize_eval_suite_name("drop"), "drop");
        assert_eq!(normalize_eval_suite_name("mmlu"), "mmlu");
        assert_eq!(normalize_eval_suite_name("gsm8k"), "gsm8k");
        
        // Names with dashes that aren't followed by numbers
        assert_eq!(normalize_eval_suite_name("ai2_arc-challenge"), "ai2_arc-challenge");
        
        // Names with hyphens in the dataset name (not shards)
        assert_eq!(normalize_eval_suite_name("simple-qa"), "simple-qa");
    }

    #[test]
    fn test_normalize_eval_suite_name_complex() {
        assert_eq!(normalize_eval_suite_name("humaneval_infilling_multiline_test-1"), "humaneval_infilling_multiline");
        assert_eq!(normalize_eval_suite_name("repobench_python_in_file_in_file-1"), "repobench_python_in_file_in_file");
        assert_eq!(normalize_eval_suite_name("alpaca-multiturn_self_compare_split_prompts"), "alpaca-multiturn_self_compare_split_prompts");
    }

    #[test]
    fn test_unique_eval_instances_across_shards() {
        // Test that the same eval instance appearing in multiple shards
        // is only counted once after normalization
        let results = vec![
            // Same instance (line 100) appears in two different shards
            create_test_result("file1.txt", 10, "mmlu_test-1", 100, Some(5)),
            create_test_result("file2.txt", 20, "mmlu_test-2", 200, Some(5)),
            // Different instance in same shard
            create_test_result("file3.txt", 30, "mmlu_test-1", 150, Some(10)),
        ];

        // Count unique eval instances manually using the same logic as the main function
        let mut unique_eval_counts: HashMap<String, HashSet<(String, usize)>> = HashMap::new();
        
        for result in &results {
            let raw_eval_suite = result.eval_key.as_ref()
                .unwrap_or(&result.eval_dataset)
                .clone();
            let eval_suite = normalize_eval_suite_name(&raw_eval_suite);
            let eval_instance_key = result.eval_instance_index.unwrap_or(result.eval_line);
            let normalized_dataset = normalize_eval_suite_name(&result.eval_dataset);
            let unique_key = (normalized_dataset, eval_instance_key);
            unique_eval_counts
                .entry(eval_suite)
                .or_default()
                .insert(unique_key);
        }

        // Should have only 1 suite
        assert_eq!(unique_eval_counts.len(), 1);
        
        // The suite should be "mmlu" (normalized)
        let mmlu_instances = unique_eval_counts.get("mmlu").unwrap();
        
        // Should count only 2 unique instances (5 and 10), not 3
        // Instance 5 appears in both mmlu_test-1 and mmlu_test-2 but counts as 1
        assert_eq!(mmlu_instances.len(), 2);
    }
}
