use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::{get_results_filename, read_config, Config};
use mj_io::{expand_dirs, read_pathbuf_to_mem};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminationResult {
    pub training_file: String,
    pub training_line: usize,
    pub eval_dataset: String,
    pub eval_line: usize,
    #[serde(alias = "overlap_ratio")]
    pub jaccard_similarity: f32,
    #[serde(default)]
    pub toxic_score: f32,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub matching_ngrams: Option<Vec<String>>,
    #[serde(default)]
    pub bucket_sizes: Option<Vec<usize>>,
    #[serde(default)]
    pub bucket_ids: Option<Vec<u64>>,
    #[serde(default)]
    pub contamination_start_idx: Option<usize>,
    #[serde(default)]
    pub contamination_end_idx: Option<usize>,
    #[serde(default)]
    pub training_overlap_text: Option<String>,
    #[serde(default)]
    pub eval_overlap_text: Option<String>,
    #[serde(default)]
    pub ngram_match_cnt: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GroundTruthRecord {
    pub text: String,
    pub source: String,
    pub id: usize,
    pub annotation: String,
    pub ground_truth: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum ClassificationType {
    TruePositive,
    FalsePositive,
    TrueNegative,
    FalseNegative,
}

#[derive(Debug)]
pub struct ClassificationStats {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub total_ground_truth: usize,
    pub total_detected: usize,
}

impl ClassificationStats {
    pub fn new() -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
            total_ground_truth: 0,
            total_detected: 0,
        }
    }

    pub fn precision(&self) -> f64 {
        if self.total_detected == 0 {
            0.0
        } else {
            self.true_positives as f64 / self.total_detected as f64
        }
    }

    pub fn recall(&self) -> f64 {
        let total_contaminated = self.true_positives + self.false_negatives;
        if total_contaminated == 0 {
            0.0
        } else {
            self.true_positives as f64 / total_contaminated as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_ground_truth == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / self.total_ground_truth as f64
        }
    }

    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * (p * r) / (p + r)
        }
    }
}

pub fn review_contamination(
    config: Option<&PathBuf>,
    results_file: Option<&PathBuf>,
    dir: Option<&PathBuf>,
    step: bool,
    metric: bool,
    fp: bool,
    fn_: bool,
    tp: bool,
    tn: bool,
    stats: bool,
    min_overlap_ratio: Option<f32>,
    min_idf_score: Option<f32>,
    min_length: Option<usize>,
    eval_filter: Option<&str>,
) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    // Handle directory-based operations
    if dir.is_some() {
        let dir_path = dir.unwrap();
        if !dir_path.exists() {
            println!("Directory not found: {:?}", dir_path);
            return Err(anyhow::anyhow!("Directory not found"));
        }

        // Load all result files from directory
        let mut all_results = load_contamination_results_from_directory(dir_path)?;

        if all_results.is_empty() {
            println!("No contamination results found in directory: {:?}", dir_path);
            return Ok(());
        }

        let original_count = all_results.len();

        // Apply filters if specified
        all_results = filter_contamination_results_by_thresholds(
            all_results,
            min_overlap_ratio,
            min_idf_score,
            min_length,
            eval_filter,
        );

        if all_results.is_empty() {
            println!("No contamination results matched the filter criteria.");
            println!("Original count: {}", original_count);
            if min_overlap_ratio.is_some() {
                println!("  - Minimum overlap ratio: {:.3}", min_overlap_ratio.unwrap());
            }
            if min_idf_score.is_some() {
                println!("  - Minimum IDF score: {:.3}", min_idf_score.unwrap());
            }
            if min_length.is_some() {
                println!("  - Minimum n-gram matches: {}", min_length.unwrap());
            }
            if eval_filter.is_some() {
                println!("  - Eval dataset filter: {}", eval_filter.unwrap());
            }
            return Ok(());
        }

        println!("Found {} contamination instances from directory", all_results.len());
        if original_count != all_results.len() {
            println!("({} filtered out by threshold criteria)", original_count - all_results.len());
        }
        println!();

        if stats {
            // Display eval dataset statistics with bar chart
            display_eval_dataset_stats(&all_results)?;
            return Ok(());
        }

        // For step-by-step review with directory
        if step {
            println!("=== REVIEWING ALL CONTAMINATION CASES ===\n");

            // Review each contamination case
            for (idx, result) in all_results.iter().enumerate() {
                if idx > 0 {
                    // Wait for user input before showing next case
                    println!("\nPress Enter to continue to next contamination case...");
                    let mut input = String::new();
                    io::stdin().read_line(&mut input).unwrap();

                    // Clear the screen
                    print!("\x1B[2J\x1B[1;1H");
                }

                println!("{}", "=".repeat(80));
                println!("CONTAMINATION #{} of {}", idx + 1, all_results.len());
                println!("{}", "=".repeat(80));

                // We don't have ground truth or config when using directory mode
                // Pass None for the config parameter
                display_contamination_case_without_config(result)?;
                println!();
            }

            println!("=== REVIEW COMPLETE ===");
            return Ok(());
        }

        // If neither stats nor step, just show summary
        println!("Use --stats to see statistics or --step to review cases interactively.");
        return Ok(());
    }

    // For non-stats operations, config is required
    if config.is_none() {
        return Err(anyhow::anyhow!("--config is required unless using --stats with --dir"));
    }

    let config_obj = read_config(config.unwrap())?;

    // Determine results file path
    let results_path = match results_file {
        Some(path) => path.clone(),
        None => config_obj
            .report_output_dir
            .join(get_results_filename(&config_obj.mode)),
    };

    if !results_path.exists() {
        println!("No contamination results file found at: {:?}", results_path);
        println!("Run contamination detection first, or specify --results-file");
        return Ok(());
    }

    // Load contamination results
    println!("Loading contamination results from: {:?}", results_path);
    let mut contamination_results = load_contamination_results(&results_path)?;

    if contamination_results.is_empty() {
        println!("No contamination found in results file.");
        return Ok(());
    }

    let original_count = contamination_results.len();

    // Apply filters if specified
    contamination_results = filter_contamination_results_by_thresholds(
        contamination_results,
        min_overlap_ratio,
        min_idf_score,
        min_length,
        eval_filter,
    );

    if contamination_results.is_empty() {
        println!("No contamination results matched the filter criteria.");
        println!("Original count: {}", original_count);
        if min_overlap_ratio.is_some() {
            println!("  - Minimum overlap ratio: {:.3}", min_overlap_ratio.unwrap());
        }
        if min_idf_score.is_some() {
            println!("  - Minimum IDF score: {:.3}", min_idf_score.unwrap());
        }
        if min_length.is_some() {
            println!("  - Minimum n-gram matches: {}", min_length.unwrap());
        }
        if eval_filter.is_some() {
            println!("  - Eval dataset filter: {}", eval_filter.unwrap());
        }
        return Ok(());
    }

    println!(
        "Found {} contamination instances to review",
        contamination_results.len()
    );
    if original_count != contamination_results.len() {
        println!("({} filtered out by threshold criteria)", original_count - contamination_results.len());
    }
    println!();

    if stats {
        // Display eval dataset statistics with bar chart
        display_eval_dataset_stats(&contamination_results)?;
        return Ok(());
    }

    // Load ground truth if available (only needed for metric and filtering)
    let ground_truth = load_ground_truth(&config_obj.local_input)?;

    if metric {
        // Calculate and display statistics
        calculate_and_display_stats(&contamination_results, &ground_truth)?;
        return Ok(());
    }

    // Handle filtering flags
    let filter_requested = fp || fn_ || tp || tn;
    let filtered_results = if filter_requested {
        filter_contamination_results(&contamination_results, &ground_truth, fp, fn_, tp, tn)?
    } else {
        contamination_results.clone()
    };

    if filtered_results.is_empty() {
        if filter_requested {
            println!("No contamination instances match the selected filter criteria.");
        } else {
            println!("No contamination found in results file.");
        }
        return Ok(());
    }

    println!(
        "Found {} contamination instances to review{}\n",
        filtered_results.len(),
        if filter_requested {
            " (after filtering)"
        } else {
            ""
        }
    );

    // Review each contamination case
    for (idx, result) in filtered_results.iter().enumerate() {
        if step && idx > 0 {
            // Wait for user input before showing next case
            println!("\nPress Enter to continue to next contamination case...");
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            // Clear the screen
            print!("\x1B[2J\x1B[1;1H");
        }

        println!("{}", "=".repeat(80));
        println!("CONTAMINATION #{} of {}", idx + 1, filtered_results.len());
        println!("{}", "=".repeat(80));

        display_contamination_case(result, &ground_truth, &config_obj)?;
        println!();
    }

    println!("=== REVIEW COMPLETE ===");
    Ok(())
}

fn load_ground_truth(input_dir: &PathBuf) -> Result<Vec<GroundTruthRecord>, Error> {
    let mut ground_truth = Vec::new();
    let training_files = expand_dirs(vec![input_dir.clone()], Some(vec![".jsonl"].as_slice()))?;

    // Load eval data for resolving ground truth when not explicitly present
    let eval_data = load_eval_data_for_ground_truth()?;

    for file_path in training_files {
        let data = read_pathbuf_to_mem(&file_path)?;
        let file_name = file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for (line_idx, line) in data.lines().enumerate() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: serde_json::Value = serde_json::from_str(&line)?;

                // Check if this record has ground truth information
                if let Some(text) = json_obj.get("text").and_then(|v| v.as_str()) {
                    let label = json_obj
                        .get("label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("UNKNOWN");

                    let ground_truth_text = if let Some(explicit_gt) =
                        json_obj.get("ground_truth").and_then(|v| v.as_str())
                    {
                        // Use explicit ground truth field
                        explicit_gt.to_string()
                    } else if let (Some(eval_dataset), Some(eval_line)) = (
                        json_obj.get("eval_dataset").and_then(|v| v.as_str()),
                        json_obj.get("eval_line").and_then(|v| v.as_u64()),
                    ) {
                        // Map eval_dataset to file names and resolve from eval data
                        let eval_file_name = format!("{}_test", eval_dataset);
                        let key = format!("{}:{}", eval_file_name, eval_line);
                        eval_data.get(&key).cloned().unwrap_or_else(|| {
                            // Try alternative mappings
                            let alt_key = format!("{}:{}", eval_dataset, eval_line);
                            eval_data.get(&alt_key).cloned().unwrap_or_else(|| {
                                format!("Could not resolve {}:{}", eval_dataset, eval_line)
                            })
                        })
                    } else if label == "CLEAN" {
                        // For clean records, use the text itself as ground truth
                        text.to_string()
                    } else {
                        // No ground truth available for non-clean records without eval references
                        continue;
                    };

                    ground_truth.push(GroundTruthRecord {
                        text: text.to_string(),
                        source: file_name.to_string(),
                        id: line_idx,
                        annotation: label.to_string(),
                        ground_truth: ground_truth_text,
                    });
                }
            }
        }
    }

    Ok(ground_truth)
}

fn load_eval_data_for_ground_truth() -> Result<std::collections::HashMap<String, String>, Error> {
    let mut eval_data = std::collections::HashMap::new();

    // Load eval files from fixtures/reference
    let eval_dir = std::path::PathBuf::from("fixtures/reference");
    if eval_dir.exists() {
        let eval_files = expand_dirs(vec![eval_dir], Some(vec![".jsonl"].as_slice()))?;

        for file_path in eval_files {
            let file_stem = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let data = read_pathbuf_to_mem(&file_path)?;
            for line in data.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    if let Ok(json_obj) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let (Some(text), Some(index), Some(record_type)) = (
                            json_obj.get("text").and_then(|v| v.as_str()),
                            json_obj.get("index").and_then(|v| v.as_u64()),
                            json_obj.get("type").and_then(|v| v.as_str()),
                        ) {
                            // Only use question records for ground truth
                            if record_type == "question" {
                                let key = format!("{}:{}", file_stem, index);
                                eval_data.insert(key, text.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(eval_data)
}

fn calculate_and_display_stats(
    contamination_results: &[ContaminationResult],
    ground_truth: &[GroundTruthRecord],
) -> Result<(), Error> {
    if ground_truth.is_empty() {
        println!("No ground truth data available. Cannot calculate classification statistics.");
        println!(
            "Found {} contamination detections total.",
            contamination_results.len()
        );
        return Ok(());
    }

    let mut stats = ClassificationStats::new();

    // Simple count of detections
    let total_detections = contamination_results.len();
    stats.total_detected = total_detections;

    // Count ground truth annotations
    stats.total_ground_truth = ground_truth.len();

    // Note: This is a simplified stats calculation
    // Without the actual text content, we can only provide basic statistics
    println!("Warning: Statistics are simplified without access to training file contents.");

    // Since we don't have the actual text, we can't calculate precise TP/FP/TN/FN
    // Just show the detection counts

    // Display statistics
    println!("=== CLASSIFICATION STATISTICS ===");
    println!();
    println!("CONFUSION MATRIX:");
    println!("                    Predicted");
    println!("                CONTAMINATED  CLEAN");
    println!(
        "    CONTAMINATED    {:>4}     {:>4}    (TP: {}, FN: {})",
        stats.true_positives, stats.false_negatives, stats.true_positives, stats.false_negatives
    );
    println!(
        "Actual CLEAN        {:>4}     {:>4}    (FP: {}, TN: {})",
        stats.false_positives, stats.true_negatives, stats.false_positives, stats.true_negatives
    );
    println!();

    println!("PERFORMANCE METRICS:");
    println!(
        "  Document-level Precision: {:.3} ({} TP / {} unique detected)",
        stats.precision(),
        stats.true_positives,
        stats.total_detected
    );
    // Removed detection-level precision since we don't have unique detection counts
    println!(
        "  Recall:     {:.3} ({} TP / {} actual contaminated)",
        stats.recall(),
        stats.true_positives,
        stats.true_positives + stats.false_negatives
    );
    println!(
        "  Accuracy:   {:.3} ({} correct / {} total)",
        stats.accuracy(),
        stats.true_positives + stats.true_negatives,
        stats.total_ground_truth
    );
    println!("  F1 Score:   {:.3}", stats.f1_score());
    println!();

    println!("DETECTION SUMMARY:");
    println!("  Total samples:      {}", stats.total_ground_truth);
    println!(
        "  Ground truth contaminated: {}",
        stats.true_positives + stats.false_negatives
    );
    println!(
        "  Ground truth clean:        {}",
        stats.true_negatives + stats.false_positives
    );
    println!("  Total detections:           {}", total_detections);

    Ok(())
}

fn filter_contamination_results(
    contamination_results: &[ContaminationResult],
    ground_truth: &[GroundTruthRecord],
    show_fp: bool,
    show_fn: bool,
    show_tp: bool,
    show_tn: bool,
) -> Result<Vec<ContaminationResult>, Error> {
    let mut filtered = Vec::new();

    // Handle TN (True Negatives) separately since they're not in contamination results
    if show_tn {
        // True negatives are ground truth CLEAN records that weren't detected
        // We'll need to create placeholder contamination results for display
        let detected_ids: std::collections::HashSet<usize> = contamination_results
            .iter()
            .filter_map(|result| {
                ground_truth
                    .iter()
                    .find(|r| {
                        let training_file_matches = result.training_file == "training_sample.jsonl"
                            || result.training_file.contains(&r.source)
                            || r.source == "training_data";
                        training_file_matches
                            && ((r.id as isize - 7) as usize == result.training_line)
                    })
                    .map(|r| r.id)
            })
            .collect();

        for record in ground_truth {
            if record.annotation.to_uppercase() == "CLEAN" && !detected_ids.contains(&record.id) {
                // Create a placeholder result for true negatives
                let placeholder = ContaminationResult {
                    training_file: format!("{}.jsonl", record.source),
                    training_line: (record.id as isize - 7) as usize,
                    eval_dataset: "N/A".to_string(),
                    eval_line: 0,
                    jaccard_similarity: 0.0,
                    toxic_score: 0.0,
                    method: Some("true_negative".to_string()),
                    matching_ngrams: None,
                    bucket_sizes: None,
                    bucket_ids: None,
                    contamination_start_idx: None,
                    contamination_end_idx: None,
                    training_overlap_text: None,
                    eval_overlap_text: None,
                    ngram_match_cnt: None,
                };
                filtered.push(placeholder);
            }
        }
    }

    // Since we don't have the actual text content, we can't accurately classify
    // For now, if filtering is requested, just return all results with a warning
    if show_tp || show_fp {
        println!(
            "Warning: TP/FP filtering not available without access to training file contents."
        );
        println!("Showing all detected contamination results.");
        filtered.extend(contamination_results.iter().cloned());
    }

    // Handle FN (False Negatives) - contaminated records that weren't detected
    if show_fn {
        println!("Warning: FN detection not available without access to training file contents.");
        // Can't determine false negatives without the actual text content
    }

    Ok(filtered)
}

fn load_contamination_results(results_path: &PathBuf) -> Result<Vec<ContaminationResult>, Error> {
    let data = read_pathbuf_to_mem(results_path)?;
    let mut results = Vec::new();

    for line in data.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let result: ContaminationResult = serde_json::from_str(&line)?;
            results.push(result);
        }
    }

    Ok(results)
}

fn filter_contamination_results_by_thresholds(
    results: Vec<ContaminationResult>,
    min_overlap_ratio: Option<f32>,
    min_idf_score: Option<f32>,
    min_length: Option<usize>,
    eval_filter: Option<&str>,
) -> Vec<ContaminationResult> {
    results.into_iter().filter(|result| {
        // Check eval dataset filter (strip suffix after last underscore)
        if let Some(eval_name) = eval_filter {
            // Strip the split suffix (everything after last underscore) from result's eval_dataset
            let parts: Vec<&str> = result.eval_dataset.split('_').collect();
            let eval_suite = if parts.len() > 1 {
                parts[..parts.len() - 1].join("_")
            } else {
                result.eval_dataset.clone()
            };
            
            // Check if it matches the filter
            if eval_suite != eval_name {
                return false;
            }
        }
        
        // Check overlap ratio (jaccard_similarity)
        if let Some(min_ratio) = min_overlap_ratio {
            if result.jaccard_similarity < min_ratio {
                return false;
            }
        }

        // Check IDF score (toxic_score)
        if let Some(min_idf) = min_idf_score {
            if result.toxic_score < min_idf {
                return false;
            }
        }

        // Check n-gram match count
        if let Some(min_len) = min_length {
            if let Some(match_cnt) = result.ngram_match_cnt {
                if match_cnt < min_len {
                    return false;
                }
            } else {
                // If ngram_match_cnt is not present, filter out
                return false;
            }
        }

        true
    }).collect()
}

fn load_contamination_results_from_directory(dir_path: &PathBuf) -> Result<Vec<ContaminationResult>, Error> {
    let mut all_results = Vec::new();

    // Find all .jsonl files in the directory
    let jsonl_files = expand_dirs(vec![dir_path.clone()], Some(vec![".jsonl"].as_slice()))?;

    println!("Processing {} JSONL files from directory...", jsonl_files.len());

    for file_path in jsonl_files {
        match load_contamination_results(&file_path) {
            Ok(results) => {
                all_results.extend(results);
            }
            Err(e) => {
                // Skip files that can't be parsed as contamination results
                println!("  Skipping file (not a contamination results file): {}", e);
            }
        }
    }

    Ok(all_results)
}

fn display_contamination_case_without_config(
    result: &ContaminationResult,
) -> Result<(), Error> {
    // This is a simplified version that doesn't need ground truth or config
    display_contamination_case_internal(result)
}

fn display_contamination_case_internal(
    result: &ContaminationResult,
) -> Result<(), Error> {
    println!("📁 TRAINING FILE: {}", result.training_file);

    // Handle special cases for FN and TN placeholders
    match result.method.as_deref() {
        Some("false_negative") => {
            println!("🚨 FALSE NEGATIVE: Contaminated but not detected");
            println!("📋 EVAL DATASET:  (Not applicable - missed detection)");
            println!();
        }
        Some("true_negative") => {
            println!("✅ TRUE NEGATIVE: Clean and correctly not detected");
            println!("📋 EVAL DATASET:  (Not applicable - correctly not flagged)");
            println!();
        }
        _ => {
            println!("📋 EVAL DATASET:  {}", result.eval_dataset);
            let similarity_label = match result.method.as_deref() {
                Some("toxic") | Some("simple") => "OVERLAP RATIO",
                _ => "JACCARD SIM",
            };
            println!(
                "🎯 {}:   {:.3}",
                similarity_label, result.jaccard_similarity
            );
            if result.toxic_score > 0.0 {
                println!("🧪 IDF SUM:    {:.3}", result.toxic_score);
            }
            if let Some(ngram_match_cnt) = result.ngram_match_cnt {
                println!("🔢 N-GRAM MATCHES: {}", ngram_match_cnt);
            }
            println!();
        }
    }

    // Display training information
    println!("🔍 TRAINING (line {}):", result.training_line);

    // Show the training overlap text if available
    if let Some(ref overlap_text) = result.training_overlap_text {
        if let (Some(start_idx), Some(end_idx)) =
            (result.contamination_start_idx, result.contamination_end_idx)
        {
            println!(
                "   📍 Training overlap (tokens {} to {}):",
                start_idx, end_idx
            );
        } else {
            println!("   📍 Training overlap:");
        }
        println!("   \"{}\"", overlap_text);
    }

    println!();

    match result.method.as_deref() {
        Some("false_negative") => {
            println!("🔍 GROUND TRUTH (expected):");
            println!("   [Ground truth not available without file access]");
        }
        _ => {
            println!("🔍 EVAL TEXT (line {}):", result.eval_line);
            // Use pre-computed eval_overlap_text if available
            if let Some(ref overlap_text) = result.eval_overlap_text {
                println!("   \"{}\"", overlap_text);
            } else {
                println!("   [No eval overlap text available in results]");
            }
        }
    }
    println!();

    // Check similarity based on method type
    match result.method.as_deref() {
        Some("false_negative") => {
            println!("❌ MISSED CONTAMINATION - This should have been detected");
        }
        Some("true_negative") => {
            println!("✅ CORRECTLY IDENTIFIED AS CLEAN - No contamination detected");
        }
        _ => {
            // Check similarity based on the scores in the result
            if result.jaccard_similarity >= 1.0 {
                println!("✅ EXACT MATCH - Definite contamination");
            } else if result.jaccard_similarity > 0.9 {
                println!("⚠️  VERY HIGH SIMILARITY - Likely contamination");
            } else if result.jaccard_similarity > 0.6 {
                println!("⚠️  HIGH SIMILARITY - Likely contamination");
            } else if result.jaccard_similarity > 0.3 {
                println!("🤔 MODERATE SIMILARITY - Manual review needed");
            } else {
                println!("🔍 LOW SIMILARITY - Edge case detection");
            }
        }
    }

    // Display matching ngrams with bucket heat if available (debug mode data)
    if let Some(ref ngrams) = result.matching_ngrams {
        if !ngrams.is_empty() {
            println!();
            println!("🔗 MATCHING N-GRAMS WITH BUCKET HEAT:");

            // Calculate bucket heat statistics for weighting analysis
            let bucket_heats: Vec<usize> = if let Some(ref sizes) = result.bucket_sizes {
                sizes.clone()
            } else {
                vec![0; ngrams.len()]
            };

            for (i, ngram) in ngrams.iter().enumerate() {
                let heat = bucket_heats.get(i).unwrap_or(&0);
                let heat_indicator = match *heat {
                    1 => "🟢",      // Cold (unique)
                    2..=5 => "🟡",  // Warm
                    6..=15 => "🟠", // Hot
                    _ => "🔴",      // Very hot
                };
                let rarity_score = if *heat > 0 { 1.0 / (*heat as f64) } else { 0.0 };

                // Include bucket ID if available
                let bucket_id_display = if let Some(ref bucket_ids) = result.bucket_ids {
                    bucket_ids
                        .get(i)
                        .map(|id| format!(" bucket_id:{}", id))
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                println!(
                    "   {}: \"{}\" {} heat:{} rarity:{:.3}{}",
                    i + 1,
                    ngram,
                    heat_indicator,
                    heat,
                    rarity_score,
                    bucket_id_display
                );
            }

            println!();
        }
    }

    Ok(())
}

fn display_contamination_case(
    result: &ContaminationResult,
    _ground_truth: &[GroundTruthRecord],
    _config: &Config,
) -> Result<(), Error> {
    // Just call the internal function since we don't use ground_truth or config
    display_contamination_case_internal(result)
}

fn display_eval_dataset_stats(contamination_results: &[ContaminationResult]) -> Result<(), Error> {
    // Count occurrences of each eval dataset
    let mut eval_counts: HashMap<String, usize> = HashMap::new();

    for result in contamination_results {
        // Strip the split suffix (everything after last underscore) to get the eval suite
        let parts: Vec<&str> = result.eval_dataset.split('_').collect();
        if parts.len() > 1 {
            // Join all parts except the last one
            let eval_suite = parts[..parts.len() - 1].join("_");
            *eval_counts.entry(eval_suite).or_insert(0) += 1;
        } else {
            // If no underscore, use the full name
            *eval_counts.entry(result.eval_dataset.clone()).or_insert(0) += 1;
        }
    }

    // Sort by count (descending)
    let mut sorted_counts: Vec<(String, usize)> = eval_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("=== EVAL DATASET STATISTICS ===");
    println!();
    println!("Total contamination incidents: {:?}", contamination_results.len());
    println!("Unique eval suites: {}", sorted_counts.len());
    println!();
    println!("Counts by eval suite:");
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
        println!(
            "  {:<30} {:>8} │{}{}│",
            suite,
            count,
            bar,
            empty
        );
    }

    println!();
    println!();

    Ok(())
}
