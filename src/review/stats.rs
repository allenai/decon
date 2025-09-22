use crate::review::ContaminationResult;
use anyhow::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Display evaluation dataset statistics with bar charts
pub fn display_eval_dataset_stats(contamination_results: &[ContaminationResult]) -> Result<(), Error> {
    // Count unique training documents per eval suite
    let mut training_docs_per_suite: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in contamination_results {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();

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

    // Count unique eval instances
    let mut unique_eval_counts: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in contamination_results {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();

        // Track unique (eval_dataset, eval_instance) pairs per suite
        // Use eval_instance_index if available, otherwise fall back to eval_line
        let eval_instance_key = result.eval_instance_index.unwrap_or(result.eval_line);
        let unique_key = (result.eval_dataset.clone(), eval_instance_key);
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

    println!("=== CONTAMINATED EVAL INSTANCES BY SUITE ===");
    println!("(Unique eval examples found in training data)");
    println!();

    // Find the maximum count for scaling the bar chart
    let max_unique_count = unique_sorted_counts.first().map(|(_, count)| *count).unwrap_or(0);

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
    display_ngram_match_distribution(contamination_results)?;

    Ok(())
}

/// Display n-gram match count histogram
fn display_ngram_match_distribution(contamination_results: &[ContaminationResult]) -> Result<(), Error> {
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
}
