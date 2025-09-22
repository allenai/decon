use anyhow::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::review::{ContaminationResult, display::display_contamination_case};

use super::{ContaminationKey, compute_contamination_metrics};

pub struct ComparisonData<'a> {
    pub dir1: &'a Path,
    pub dir2: &'a Path,
    pub map1: &'a HashMap<ContaminationKey, ContaminationResult>,
    pub map2: &'a HashMap<ContaminationKey, ContaminationResult>,
    pub only_in_1: &'a HashSet<ContaminationKey>,
    pub only_in_2: &'a HashSet<ContaminationKey>,
    pub in_both: &'a HashSet<ContaminationKey>,
    pub original_count1: usize,
    pub original_count2: usize,
    pub min_score: Option<f32>,
    pub eval_filter: Option<&'a str>,
}

pub fn display_comparison_stats(data: &ComparisonData) -> Result<(), Error> {
    let dir1 = data.dir1;
    let dir2 = data.dir2;
    let map1 = data.map1;
    let map2 = data.map2;
    let only_in_1 = data.only_in_1;
    let only_in_2 = data.only_in_2;
    let in_both = data.in_both;
    let original_count1 = data.original_count1;
    let original_count2 = data.original_count2;
    let min_score = data.min_score;
    let eval_filter = data.eval_filter;
    // Compute contamination metrics for both directories
    let results1: Vec<ContaminationResult> = map1.values().cloned().collect();
    let results2: Vec<ContaminationResult> = map2.values().cloned().collect();
    let metrics1 = compute_contamination_metrics(&results1);
    let metrics2 = compute_contamination_metrics(&results2);

    // Calculate average contamination scores as a point of interest
    let avg_score1 = if !map1.is_empty() {
        map1.values()
            .filter_map(|r| r.contamination_score)
            .sum::<f32>() / map1.len() as f32
    } else {
        0.0
    };

    let avg_score2 = if !map2.is_empty() {
        map2.values()
            .filter_map(|r| r.contamination_score)
            .sum::<f32>() / map2.len() as f32
    } else {
        0.0
    };

    println!("=== DIRECTORY 1: {:?} ===", dir1);
    println!("  Training docs contaminated: {}", metrics1.training_docs_contaminated);
    println!("  Total contamination instances: {}", metrics1.contamination_instances);
    println!("  Unique eval instances: {}", metrics1.contaminated_evals);
    if original_count1 != map1.len() {
        let filtered_count = original_count1 - map1.len();
        let mut filter_reasons = Vec::new();

        if min_score.is_some() {
            filter_reasons.push(format!("min score: {:.3}", min_score.unwrap()));
        }
        if eval_filter.is_some() {
            filter_reasons.push(format!("eval dataset: {}", eval_filter.unwrap()));
        }

        if filter_reasons.is_empty() {
            println!("  ({} filtered out)", filtered_count);
        } else {
            println!("  ({} filtered out by: {})", filtered_count, filter_reasons.join(", "));
        }
    }
    println!("  Average contamination score: {:.3}", avg_score1);
    println!();

    println!("=== DIRECTORY 2: {:?} ===", dir2);
    println!("  Training docs contaminated: {}", metrics2.training_docs_contaminated);
    println!("  Total contamination instances: {}", metrics2.contamination_instances);
    println!("  Unique eval instances: {}", metrics2.contaminated_evals);
    if original_count2 != map2.len() {
        let filtered_count = original_count2 - map2.len();
        let mut filter_reasons = Vec::new();

        if min_score.is_some() {
            filter_reasons.push(format!("min score: {:.3}", min_score.unwrap()));
        }
        if eval_filter.is_some() {
            filter_reasons.push(format!("eval dataset: {}", eval_filter.unwrap()));
        }

        if filter_reasons.is_empty() {
            println!("  ({} filtered out)", filtered_count);
        } else {
            println!("  ({} filtered out by: {})", filtered_count, filter_reasons.join(", "));
        }
    }
    println!("  Average contamination score: {:.3}", avg_score2);
    println!();

    println!("=== COMPARISON SUMMARY ===");
    println!("  Training docs delta: {} → {} ({:+})",
        metrics1.training_docs_contaminated,
        metrics2.training_docs_contaminated,
        metrics2.training_docs_contaminated as i32 - metrics1.training_docs_contaminated as i32);
    println!("  Contamination instances delta: {} → {} ({:+})",
        metrics1.contamination_instances,
        metrics2.contamination_instances,
        metrics2.contamination_instances as i32 - metrics1.contamination_instances as i32);
    println!("  Eval instances delta: {} → {} ({:+})",
        metrics1.contaminated_evals,
        metrics2.contaminated_evals,
        metrics2.contaminated_evals as i32 - metrics1.contaminated_evals as i32);
    println!();
    println!("  Only in dir1: {} contamination instances", only_in_1.len());
    println!("  Only in dir2: {} contamination instances", only_in_2.len());
    println!("  Common: {} contamination instances", in_both.len());

    // Calculate score differences for common items
    if !in_both.is_empty() {
        let mut score_improved = 0;
        let mut score_worsened = 0;
        let mut score_unchanged = 0;
        let mut total_improvement = 0.0;
        let mut total_worsening = 0.0;

        for key in in_both {
            let score1 = map1[key].contamination_score.unwrap_or(0.0);
            let score2 = map2[key].contamination_score.unwrap_or(0.0);

            if (score2 - score1).abs() < 0.001 {
                score_unchanged += 1;
            } else if score2 < score1 {
                score_improved += 1;
                total_improvement += score1 - score2;
            } else {
                score_worsened += 1;
                total_worsening += score2 - score1;
            }
        }

        println!();
        println!("  Score improved: {} cases", score_improved);
        if score_improved > 0 {
            println!("    (avg delta: -{:.3})", total_improvement / score_improved as f32);
        }
        println!("  Score worsened: {} cases", score_worsened);
        if score_worsened > 0 {
            println!("    (avg delta: +{:.3})", total_worsening / score_worsened as f32);
        }
        println!("  Score unchanged: {} cases", score_unchanged);
    }

    // Per-dataset breakdown: Training Documents Removed
    println!();
    println!("=== TRAINING DOCUMENTS REMOVED PER DATASET ===");
    println!("(Unique training documents that need removal per eval suite)");
    println!();

    // Collect unique training docs per eval dataset for both directories
    let mut training_docs_per_suite1: HashMap<String, HashSet<(String, usize)>> = HashMap::new();
    let mut training_docs_per_suite2: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in map1.values() {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        training_docs_per_suite1
            .entry(eval_suite)
            .or_default()
            .insert((result.training_file.clone(), result.training_line));
    }

    for result in map2.values() {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        training_docs_per_suite2
            .entry(eval_suite)
            .or_default()
            .insert((result.training_file.clone(), result.training_line));
    }

    // Convert to counts
    let mut training_counts1: HashMap<String, usize> = HashMap::new();
    let mut training_counts2: HashMap<String, usize> = HashMap::new();

    for (suite, docs) in training_docs_per_suite1 {
        training_counts1.insert(suite, docs.len());
    }
    for (suite, docs) in training_docs_per_suite2 {
        training_counts2.insert(suite, docs.len());
    }

    // Get all unique datasets
    let mut all_datasets: HashSet<String> = HashSet::new();
    all_datasets.extend(training_counts1.keys().cloned());
    all_datasets.extend(training_counts2.keys().cloned());

    // Create a vector of (dataset, count1, count2, delta) for sorting
    let mut training_deltas: Vec<(String, usize, usize, i32)> = all_datasets
        .iter()
        .map(|dataset| {
            let count1 = *training_counts1.get(dataset).unwrap_or(&0);
            let count2 = *training_counts2.get(dataset).unwrap_or(&0);
            let delta = count2 as i32 - count1 as i32;
            (dataset.clone(), count1, count2, delta)
        })
        .collect();

    // Sort by delta in descending order (largest increases first, then smallest decreases)
    training_deltas.sort_by(|a, b| b.3.cmp(&a.3));

    if !training_deltas.is_empty() {
        // Calculate column widths
        let dataset_width = training_deltas
            .iter()
            .map(|(name, _, _, _)| name.len())
            .max()
            .unwrap_or(20)
            .max(10);

        // Print table header
        println!("  {:<width$}   {:>8}   {:>8}   {:>10}",
            "Dataset", "Dir 1", "Dir 2", "Change",
            width = dataset_width
        );
        println!("  {:-<width$}   {:->8}   {:->8}   {:->10}",
            "", "", "", "",
            width = dataset_width
        );

        for (dataset, count1, count2, diff) in training_deltas {
            let diff_str = if diff > 0 {
                format!("+{}", diff)
            } else if diff < 0 {
                format!("{}", diff)
            } else {
                "0".to_string()
            };

            println!("  {:<width$}   {:>8}   {:>8}   {:>10}",
                dataset, count1, count2, diff_str,
                width = dataset_width
            );
        }
    }

    // Per-dataset breakdown: Eval Instances Contaminated
    println!();
    println!("=== EVAL INSTANCES CONTAMINATED PER DATASET ===");
    println!("(Unique eval examples found in training data per eval suite)");
    println!();

    // Collect unique eval instances per eval dataset for both directories
    // Using (split, eval_instance_index) as the unique identifier
    let mut eval_instances_per_suite1: HashMap<String, HashSet<(String, usize)>> = HashMap::new();
    let mut eval_instances_per_suite2: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result in map1.values() {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        // Require eval_instance_index to be present
        let eval_instance_key = result.eval_instance_index
            .expect("eval_instance_index is required but was not found in contamination result");
        let split = result.split.clone()
            .unwrap_or_else(|| "unknown".to_string());
        eval_instances_per_suite1
            .entry(eval_suite)
            .or_default()
            .insert((split, eval_instance_key));
    }

    for result in map2.values() {
        // Use eval_key if available (clean dataset identifier)
        // Fall back to eval_dataset for backward compatibility
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        // Require eval_instance_index to be present
        let eval_instance_key = result.eval_instance_index
            .expect("eval_instance_index is required but was not found in contamination result");
        let split = result.split.clone()
            .unwrap_or_else(|| "unknown".to_string());
        eval_instances_per_suite2
            .entry(eval_suite)
            .or_default()
            .insert((split, eval_instance_key));
    }

    // Convert to counts
    let mut eval_counts1: HashMap<String, usize> = HashMap::new();
    let mut eval_counts2: HashMap<String, usize> = HashMap::new();

    for (suite, instances) in eval_instances_per_suite1 {
        eval_counts1.insert(suite, instances.len());
    }
    for (suite, instances) in eval_instances_per_suite2 {
        eval_counts2.insert(suite, instances.len());
    }

    // Create a vector of (dataset, count1, count2, delta) for sorting
    let mut eval_deltas: Vec<(String, usize, usize, i32)> = all_datasets
        .into_iter()
        .map(|dataset| {
            let count1 = *eval_counts1.get(&dataset).unwrap_or(&0);
            let count2 = *eval_counts2.get(&dataset).unwrap_or(&0);
            let delta = count2 as i32 - count1 as i32;
            (dataset, count1, count2, delta)
        })
        .collect();

    // Sort by delta in descending order (largest increases first, then smallest decreases)
    eval_deltas.sort_by(|a, b| b.3.cmp(&a.3));

    if !eval_deltas.is_empty() {
        // Calculate column widths
        let dataset_width = eval_deltas
            .iter()
            .map(|(name, _, _, _)| name.len())
            .max()
            .unwrap_or(20)
            .max(10);

        // Print table header
        println!("  {:<width$}   {:>8}   {:>8}   {:>10}",
            "Dataset", "Dir 1", "Dir 2", "Change",
            width = dataset_width
        );
        println!("  {:-<width$}   {:->8}   {:->8}   {:->10}",
            "", "", "", "",
            width = dataset_width
        );

        for (dataset, count1, count2, diff) in eval_deltas {
            let diff_str = if diff > 0 {
                format!("+{}", diff)
            } else if diff < 0 {
                format!("{}", diff)
            } else {
                "0".to_string()
            };

            println!("  {:<width$}   {:>8}   {:>8}   {:>10}",
                dataset, count1, count2, diff_str,
                width = dataset_width
            );
        }
    }

    // Show active filters if any
    if min_score.is_some() || eval_filter.is_some() {
        println!();
        println!("Active filters:");
        if let Some(score) = min_score {
            println!("  Minimum contamination score: {:.3}", score);
        }
        if let Some(filter) = eval_filter {
            println!("  Eval dataset filter: {}", filter);
        }
    }

    println!();

    Ok(())
}

pub fn display_common(
    map1: &HashMap<ContaminationKey, ContaminationResult>,
    map2: &HashMap<ContaminationKey, ContaminationResult>,
    in_both: &HashSet<ContaminationKey>,
    _verbose: bool,
) -> Result<(), Error> {
    if in_both.is_empty() {
        println!("=== NO COMMON CONTAMINATIONS ===");
        return Ok(());
    }

    println!("=== COMMON CONTAMINATIONS ===");
    println!("Found {} contaminations in both directories", in_both.len());
    println!();

    // Sort by score difference (largest differences first)
    let mut sorted_keys: Vec<_> = in_both.iter().cloned().collect();
    sorted_keys.sort_by(|a, b| {
        let score_a1 = map1[a].contamination_score.unwrap_or(0.0);
        let score_a2 = map2[a].contamination_score.unwrap_or(0.0);
        let diff_a = (score_a2 - score_a1).abs();

        let score_b1 = map1[b].contamination_score.unwrap_or(0.0);
        let score_b2 = map2[b].contamination_score.unwrap_or(0.0);
        let diff_b = (score_b2 - score_b1).abs();

        diff_b.partial_cmp(&diff_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (idx, key) in sorted_keys.iter().enumerate() {
        if idx > 0 {
            println!();
            println!("{}", "=".repeat(80));
            println!();
        }

        println!("COMMON #{} of {}", idx + 1, sorted_keys.len());
        println!("{}", "=".repeat(80));

        let result1 = &map1[key];
        let result2 = &map2[key];

        println!("TRAINING FILE: {} (line {})", result1.training_file, result1.training_line);
        println!("EVAL DATASET:  {} (line {})", result1.eval_dataset, result1.eval_line);
        println!();

        let score1 = result1.contamination_score.unwrap_or(0.0);
        let score2 = result2.contamination_score.unwrap_or(0.0);
        let diff = score2 - score1;

        println!("⚡ CONTAMINATION SCORES:");
        println!("   Directory 1: {:.3}", score1);
        println!("   Directory 2: {:.3}", score2);

        if diff.abs() > 0.001 {
            if diff > 0.0 {
                println!("   Change: +{:.3} (worsened)", diff);
            } else {
                println!("   Change: {:.3} (improved)", diff);
            }
        } else {
            println!("   Change: 0.000 (unchanged)");
        }

        // Show additional metrics if they differ significantly
        if let (Some(idf1), Some(idf2)) = (result1.idf_overlap, result2.idf_overlap)
            && (idf1 - idf2).abs() > 0.001 {
                println!();
                println!("IDF OVERLAP:");
                println!("   Directory 1: {:.3}", idf1);
                println!("   Directory 2: {:.3}", idf2);
            }

        if let (Some(cnt1), Some(cnt2)) = (result1.ngram_match_cnt, result2.ngram_match_cnt)
            && cnt1 != cnt2 {
                println!();
                println!("N-GRAM MATCHES:");
                println!("   Directory 1: {}", cnt1);
                println!("   Directory 2: {}", cnt2);
            }

        // Show the text (should be the same for both)
        println!();
        println!("TRAINING:");
        if let Some(ref overlap_text) = result1.training_overlap_text {
            println!("   \"{}\"", overlap_text);
        }

        println!();
        println!("EVAL TEXT:");
        if let Some(ref overlap_text) = result1.eval_overlap_text {
            println!("   \"{}\"", overlap_text);
        }
    }

    println!();
    println!("=== END OF COMMON ===");

    Ok(())
}

pub fn display_only_in_first(
    map1: &HashMap<ContaminationKey, ContaminationResult>,
    map2: &HashMap<ContaminationKey, ContaminationResult>,
) -> Result<(), Error> {
    let mut fingerprints1: HashMap<String, Vec<ContaminationResult>> = HashMap::new();
    let mut fingerprints2: HashSet<String> = HashSet::new();

    // Collect fingerprints from first directory
    for result in map1.values() {
        if let Some(ref fingerprint) = result.fingerprint {
            fingerprints1
                .entry(fingerprint.clone())
                .or_default()
                .push(result.clone());
        }
    }

    // Collect fingerprints from second directory
    for result in map2.values() {
        if let Some(ref fingerprint) = result.fingerprint {
            fingerprints2.insert(fingerprint.clone());
        }
    }

    // Find fingerprints only in first directory
    let mut only_in_first: Vec<(String, Vec<ContaminationResult>)> = Vec::new();
    for (fingerprint, mut results) in fingerprints1 {
        if !fingerprints2.contains(&fingerprint) {
            // Sort results within each fingerprint group by contamination score
            results.sort_by(|a, b| {
                let score_a = a.contamination_score.unwrap_or(0.0);
                let score_b = b.contamination_score.unwrap_or(0.0);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });
            only_in_first.push((fingerprint, results));
        }
    }

    // Sort fingerprint groups by number of contamination instances (descending)
    only_in_first.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    if only_in_first.is_empty() {
        println!("=== NO FINGERPRINTS UNIQUE TO FIRST DIRECTORY ===");
        return Ok(());
    }

    let total_instances: usize = only_in_first.iter().map(|(_, v)| v.len()).sum();

    println!("=== FINGERPRINTS ONLY IN FIRST DIRECTORY ===");
    println!("Found {} unique fingerprints with {} total contamination instances",
        only_in_first.len(),
        total_instances
    );
    println!();

    // Display all contamination instances grouped by fingerprint
    let mut global_idx = 0;
    for (fp_idx, (fingerprint, results)) in only_in_first.iter().enumerate() {
        if fp_idx > 0 {
            println!();
            println!("{}", "=".repeat(80));
            println!();
        }

        println!("FINGERPRINT GROUP #{} of {}", fp_idx + 1, only_in_first.len());
        println!("FINGERPRINT: {}", fingerprint);
        println!("{} contamination instances in this group", results.len());
        println!("{}", "=".repeat(80));

        // Display each contamination result in this fingerprint group
        for (result_idx, result) in results.iter().enumerate() {
            global_idx += 1;

            if result_idx > 0 {
                println!();
                println!("{}", "-".repeat(80));
                println!();
            }

            println!("ONLY IN FIRST #{} of {} (Group {}, Item {} of {})",
                global_idx, total_instances, fp_idx + 1, result_idx + 1, results.len());
            println!("{}", "=".repeat(80));

            // Use the same display function as --only-in-first
            display_contamination_case(result, true)?;
        }
    }

    println!();
    println!("=== END OF FINGERPRINTS ONLY IN FIRST ===");

    Ok(())
}

pub fn display_only_in_second(
    map1: &HashMap<ContaminationKey, ContaminationResult>,
    map2: &HashMap<ContaminationKey, ContaminationResult>,
) -> Result<(), Error> {
    let mut fingerprints1: HashSet<String> = HashSet::new();
    let mut fingerprints2: HashMap<String, Vec<ContaminationResult>> = HashMap::new();

    for result in map1.values() {
        if let Some(ref fingerprint) = result.fingerprint {
            fingerprints1.insert(fingerprint.clone());
        }
    }

    for result in map2.values() {
        if let Some(ref fingerprint) = result.fingerprint {
            fingerprints2
                .entry(fingerprint.clone())
                .or_default()
                .push(result.clone());
        }
    }

    // Find fingerprints only in second directory
    let mut only_in_second: Vec<(String, Vec<ContaminationResult>)> = Vec::new();
    for (fingerprint, mut results) in fingerprints2 {
        if !fingerprints1.contains(&fingerprint) {
            // Sort results within each fingerprint group by contamination score
            results.sort_by(|a, b| {
                let score_a = a.contamination_score.unwrap_or(0.0);
                let score_b = b.contamination_score.unwrap_or(0.0);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });
            only_in_second.push((fingerprint, results));
        }
    }

    // Sort fingerprint groups by number of contamination instances (descending)
    only_in_second.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    if only_in_second.is_empty() {
        println!("=== NO FINGERPRINTS UNIQUE TO SECOND DIRECTORY ===");
        return Ok(());
    }

    let total_instances: usize = only_in_second.iter().map(|(_, v)| v.len()).sum();

    println!("=== FINGERPRINTS ONLY IN SECOND DIRECTORY ===");
    println!("Found {} unique fingerprints with {} total contamination instances",
        only_in_second.len(),
        total_instances
    );
    println!();

    // Display all contamination instances grouped by fingerprint
    let mut global_idx = 0;
    for (fp_idx, (fingerprint, results)) in only_in_second.iter().enumerate() {
        if fp_idx > 0 {
            println!();
            println!("{}", "=".repeat(80));
            println!();
        }

        println!("FINGERPRINT GROUP #{} of {}", fp_idx + 1, only_in_second.len());
        println!("FINGERPRINT: {}", fingerprint);
        println!("{} contamination instances in this group", results.len());
        println!("{}", "=".repeat(80));

        // Display each contamination result in this fingerprint group
        for (result_idx, result) in results.iter().enumerate() {
            global_idx += 1;

            if result_idx > 0 {
                println!();
                println!("{}", "-".repeat(80));
                println!();
            }

            println!("ONLY IN SECOND #{} of {} (Group {}, Item {} of {})",
                global_idx, total_instances, fp_idx + 1, result_idx + 1, results.len());
            println!("{}", "=".repeat(80));

            // Use the same display function as --only-in-second
            display_contamination_case(result, true)?;
        }
    }

    println!();
    println!("=== END OF FINGERPRINTS ONLY IN SECOND ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::ContaminationResult;

    fn create_test_contamination_result(
        training_file: &str,
        training_line: usize,
        eval_dataset: &str,
        eval_line: usize,
        score: Option<f32>,
        fingerprint: Option<String>,
    ) -> ContaminationResult {
        ContaminationResult {
            training_file: training_file.to_string(),
            training_line,
            eval_dataset: eval_dataset.to_string(),
            eval_key: Some(eval_dataset.to_string()),
            eval_line,
            eval_instance_index: Some(eval_line),
            split: Some("test".to_string()),
            method: None,
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: Some("test text".to_string()),
            eval_overlap_text: Some("eval text".to_string()),
            ngram_match_cnt: None,
            eval_unique_ngrams: None,
            contamination_score: score,
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
            fingerprint,
            is_correct: None,
            reference_file: None,
        }
    }

    #[test]
    fn test_average_score_calculation_empty() {
        let map: HashMap<ContaminationKey, ContaminationResult> = HashMap::new();

        // Calculate average score for empty map
        let avg_score = if !map.is_empty() {
            map.values()
                .filter_map(|r| r.contamination_score)
                .sum::<f32>() / map.len() as f32
        } else {
            0.0
        };

        assert_eq!(avg_score, 0.0);
    }

    #[test]
    fn test_average_score_calculation_single_item() {
        let mut map: HashMap<ContaminationKey, ContaminationResult> = HashMap::new();
        let key = ("train1.jsonl".to_string(), 10, "mmlu".to_string(), 5);
        let result = create_test_contamination_result(
            "train1.jsonl", 10, "mmlu", 5, Some(0.75), None
        );
        map.insert(key, result);

        let avg_score = if !map.is_empty() {
            map.values()
                .filter_map(|r| r.contamination_score)
                .sum::<f32>() / map.len() as f32
        } else {
            0.0
        };

        assert_eq!(avg_score, 0.75);
    }

    #[test]
    fn test_average_score_calculation_multiple_items() {
        let mut map: HashMap<ContaminationKey, ContaminationResult> = HashMap::new();

        let key1 = ("train1.jsonl".to_string(), 10, "mmlu".to_string(), 5);
        let result1 = create_test_contamination_result(
            "train1.jsonl", 10, "mmlu", 5, Some(0.8), None
        );
        map.insert(key1, result1);

        let key2 = ("train2.jsonl".to_string(), 20, "gsm8k".to_string(), 15);
        let result2 = create_test_contamination_result(
            "train2.jsonl", 20, "gsm8k", 15, Some(0.6), None
        );
        map.insert(key2, result2);

        let key3 = ("train3.jsonl".to_string(), 30, "humaneval".to_string(), 25);
        let result3 = create_test_contamination_result(
            "train3.jsonl", 30, "humaneval", 25, Some(0.4), None
        );
        map.insert(key3, result3);

        let avg_score = if !map.is_empty() {
            map.values()
                .filter_map(|r| r.contamination_score)
                .sum::<f32>() / map.len() as f32
        } else {
            0.0
        };

        assert!((avg_score - 0.6).abs() < 0.001); // (0.8 + 0.6 + 0.4) / 3
    }

    #[test]
    fn test_score_difference_calculations() {
        let mut map1: HashMap<ContaminationKey, ContaminationResult> = HashMap::new();
        let mut map2: HashMap<ContaminationKey, ContaminationResult> = HashMap::new();
        let mut in_both: HashSet<ContaminationKey> = HashSet::new();

        let key = ("train1.jsonl".to_string(), 10, "mmlu".to_string(), 5);
        in_both.insert(key.clone());

        let result1 = create_test_contamination_result(
            "train1.jsonl", 10, "mmlu", 5, Some(0.8), None
        );
        let result2 = create_test_contamination_result(
            "train1.jsonl", 10, "mmlu", 5, Some(0.5), None
        );

        map1.insert(key.clone(), result1);
        map2.insert(key.clone(), result2);

        // Test score difference calculation logic
        let mut score_improved = 0;
        let mut score_worsened = 0;
        let mut score_unchanged = 0;
        let mut total_improvement = 0.0;
        let mut total_worsening = 0.0;

        for key in &in_both {
            let score1 = map1[key].contamination_score.unwrap_or(0.0);
            let score2 = map2[key].contamination_score.unwrap_or(0.0);

            if (score2 - score1).abs() < 0.001 {
                score_unchanged += 1;
            } else if score2 < score1 {
                score_improved += 1;
                total_improvement += score1 - score2;
            } else {
                score_worsened += 1;
                total_worsening += score2 - score1;
            }
        }

        assert_eq!(score_improved, 1);
        assert_eq!(score_worsened, 0);
        assert_eq!(score_unchanged, 0);
        assert!((total_improvement - 0.3).abs() < 0.001); // 0.8 - 0.5 = 0.3
        assert_eq!(total_worsening, 0.0); // No worsening in this test case
    }

    #[test]
    fn test_fingerprint_grouping() {
        let mut fingerprints: HashMap<String, Vec<ContaminationResult>> = HashMap::new();

        let result1 = create_test_contamination_result(
            "train1.jsonl", 10, "mmlu", 5, Some(0.8), Some("fp1".to_string())
        );
        let result2 = create_test_contamination_result(
            "train2.jsonl", 20, "mmlu", 15, Some(0.7), Some("fp1".to_string())
        );
        let result3 = create_test_contamination_result(
            "train3.jsonl", 30, "gsm8k", 25, Some(0.6), Some("fp2".to_string())
        );

        // Group by fingerprint
        for result in vec![result1, result2, result3] {
            if let Some(ref fingerprint) = result.fingerprint {
                fingerprints
                    .entry(fingerprint.clone())
                    .or_default()
                    .push(result);
            }
        }

        assert_eq!(fingerprints.len(), 2);
        assert_eq!(fingerprints.get("fp1").unwrap().len(), 2);
        assert_eq!(fingerprints.get("fp2").unwrap().len(), 1);
    }

    #[test]
    fn test_delta_calculations() {
        let count1 = 100;
        let count2 = 150;
        let delta = count2 - count1;

        assert_eq!(delta, 50);

        let count1 = 200;
        let count2 = 150;
        let delta = count2 - count1;

        assert_eq!(delta, -50);
    }

    #[test]
    fn test_filter_message_generation() {
        let mut filter_reasons = Vec::new();
        let min_score = 0.5;
        let eval_filter = "mmlu";

        filter_reasons.push(format!("min score: {:.3}", min_score));
        filter_reasons.push(format!("eval dataset: {}", eval_filter));

        let message = filter_reasons.join(", ");
        assert_eq!(message, "min score: 0.500, eval dataset: mmlu");
    }
}
