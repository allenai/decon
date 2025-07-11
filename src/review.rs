use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::{
    clean_text, get_nested_json_val, get_results_filename, read_config, Config, OmniTokenizer,
};
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
    config: &PathBuf,
    results_file: Option<&PathBuf>,
    step: bool,
    stats: bool,
    fp: bool,
    fn_: bool,
    tp: bool,
    tn: bool,
    full: bool,
) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    let config_obj = read_config(config)?;

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
    let contamination_results = load_contamination_results(&results_path)?;

    if contamination_results.is_empty() {
        println!("No contamination found in results file.");
        return Ok(());
    }

    println!(
        "Found {} contamination instances to review\n",
        contamination_results.len()
    );

    // Load file content caches
    let training_cache = load_training_files(&config_obj.local_input, &config_obj.content_key)?;
    let eval_cache = load_eval_files(&config_obj.reference_input, &config_obj.content_key)?;
    let ground_truth = load_ground_truth(&config_obj.local_input)?;

    if stats {
        // Calculate and display statistics
        calculate_and_display_stats(&contamination_results, &ground_truth)?;
        return Ok(());
    }

    // Handle filtering flags
    let filter_requested = fp || fn_ || tp || tn;
    let filtered_results = if filter_requested {
        filter_contamination_results(
            &contamination_results,
            &ground_truth,
            &training_cache,
            fp,
            fn_,
            tp,
            tn,
        )?
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

        display_contamination_case(
            result,
            &training_cache,
            &eval_cache,
            &ground_truth,
            full,
            &config_obj,
        )?;
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

    // Load training cache to get actual text content for mapping
    let training_cache =
        load_training_files(&std::path::PathBuf::from("fixtures/training"), "text")?;

    // Create a set of detected texts for accurate mapping
    let mut detected_texts: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unmapped_detections = 0;
    let mut all_detected_texts: Vec<String> = Vec::new();

    for result in contamination_results {
        if let Some(lines) = training_cache.get(&result.training_file) {
            if result.training_line < lines.len() {
                let text = lines[result.training_line].clone();
                detected_texts.insert(text.clone());
                all_detected_texts.push(text);
            } else {
                unmapped_detections += 1;
            }
        } else {
            unmapped_detections += 1;
        }
    }

    let unique_detections = detected_texts.len();
    let total_detections = all_detected_texts.len();

    // Count ground truth annotations
    for record in ground_truth {
        stats.total_ground_truth += 1;
        let is_contaminated = record.annotation.to_uppercase() == "CONTAMINATED";
        let is_detected = detected_texts.contains(&record.text);

        match (is_contaminated, is_detected) {
            (true, true) => stats.true_positives += 1,
            (true, false) => stats.false_negatives += 1,
            (false, true) => stats.false_positives += 1,
            (false, false) => stats.true_negatives += 1,
        }
    }

    // Use unique detected texts for document-level precision calculation
    stats.total_detected = unique_detections;

    // Calculate detection-level precision for comparison
    let detection_level_precision = if total_detections == 0 {
        0.0
    } else {
        stats.true_positives as f64 / total_detections as f64
    };

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
    if total_detections != unique_detections {
        println!(
            "  Detection-level Precision: {:.3} ({} TP / {} total detections)",
            detection_level_precision, stats.true_positives, total_detections
        );
    }
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
    println!("  Unique detected texts:      {}", unique_detections);
    println!("  Total detections:           {}", total_detections);
    println!("  Missed contamination:       {}", stats.false_negatives);
    println!("  False alarms:               {}", stats.false_positives);

    if unmapped_detections > 0 {
        println!("  Unmapped detections:        {}", unmapped_detections);
    }

    if total_detections != unique_detections {
        println!(
            "  Duplicate detections:       {}",
            total_detections - unique_detections
        );
    }

    Ok(())
}

fn classify_contamination_result(
    result: &ContaminationResult,
    ground_truth: &[GroundTruthRecord],
    training_cache: &HashMap<String, Vec<String>>,
) -> ClassificationType {
    // Get the actual text from the training file at the specified line
    if let Some(lines) = training_cache.get(&result.training_file) {
        if result.training_line < lines.len() {
            let training_text = &lines[result.training_line];

            // Find the corresponding ground truth record by matching the text content
            if let Some(record) = ground_truth.iter().find(|r| r.text == *training_text) {
                let is_actually_contaminated = record.annotation.to_uppercase() == "CONTAMINATED";
                let is_detected = true; // If it's in results, it was detected

                match (is_actually_contaminated, is_detected) {
                    (true, true) => ClassificationType::TruePositive,
                    (false, true) => ClassificationType::FalsePositive,
                    _ => ClassificationType::TruePositive, // Fallback, shouldn't happen for detected items
                }
            } else {
                eprintln!(
                    "DEBUG: Could not find ground truth for text: {}",
                    &training_text[..std::cmp::min(25, training_text.len())]
                );
                ClassificationType::FalsePositive
            }
        } else {
            eprintln!(
                "DEBUG: Line {} out of bounds for file {}",
                result.training_line, result.training_file
            );
            ClassificationType::FalsePositive
        }
    } else {
        eprintln!(
            "DEBUG: Could not find training file: {}",
            result.training_file
        );
        ClassificationType::FalsePositive
    }
}

fn filter_contamination_results(
    contamination_results: &[ContaminationResult],
    ground_truth: &[GroundTruthRecord],
    training_cache: &HashMap<String, Vec<String>>,
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

    // Filter actual contamination results
    for result in contamination_results {
        let classification = classify_contamination_result(result, ground_truth, training_cache);

        let should_include = match classification {
            ClassificationType::TruePositive => show_tp,
            ClassificationType::FalsePositive => show_fp,
            ClassificationType::TrueNegative => show_tn,
            ClassificationType::FalseNegative => show_fn,
        };

        if should_include {
            filtered.push(result.clone());
        }
    }

    // Handle FN (False Negatives) - contaminated records that weren't detected
    if show_fn {
        // Create a set of detected texts to avoid complex ID mapping
        let detected_texts: std::collections::HashSet<String> = contamination_results
            .iter()
            .filter_map(|result| {
                training_cache
                    .get(&result.training_file)
                    .and_then(|lines| lines.get(result.training_line))
                    .cloned()
            })
            .collect();

        for record in ground_truth {
            if record.annotation.to_uppercase() == "CONTAMINATED"
                && !detected_texts.contains(&record.text)
            {
                // Find which file this record is in by searching the training cache
                let mut found_file = None;
                let mut found_line = 0;

                for (filename, lines) in training_cache.iter() {
                    if let Some(line_idx) = lines.iter().position(|line| line == &record.text) {
                        found_file = Some(filename.clone());
                        found_line = line_idx;
                        break;
                    }
                }

                let placeholder = ContaminationResult {
                    training_file: found_file
                        .unwrap_or_else(|| format!("unknown_{}.jsonl", record.source)),
                    training_line: found_line,
                    eval_dataset: "N/A".to_string(),
                    eval_line: 0,
                    jaccard_similarity: 0.0,
                    toxic_score: 0.0,
                    method: Some("false_negative".to_string()),
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

fn load_training_files(
    input_dir: &PathBuf,
    content_key: &str,
) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let training_files = expand_dirs(
        vec![input_dir.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;

    for file_path in training_files {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            // For .gz files, use file_stem to get the .jsonl name
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // For regular files, use the full filename
            file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for (line_num, line) in data.lines().enumerate() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = match serde_json::from_str(&line) {
                    Ok(obj) => obj,
                    Err(e) => {
                        eprintln!("JSON parse error at line {}: {}", line_num, e);
                        eprintln!("Line content: {}", &line[..std::cmp::min(100, line.len())]);
                        return Err(e.into());
                    }
                };
                let text = match get_nested_json_val(&json_obj, &content_key.to_string()) {
                    Ok(t) => t,
                    Err(_) => {
                        // Skip files that don't have the expected schema
                        eprintln!(
                            "Skipping file {:?} - doesn't have expected key '{}'",
                            file_path, content_key
                        );
                        break;
                    }
                };
                lines.push(text);
            }
        }

        println!(
            "Caching training file: {} ({} lines)",
            file_name,
            lines.len()
        );
        cache.insert(file_name, lines);
    }

    println!("Training file cache contains {} files:", cache.len());
    for key in cache.keys() {
        println!("  - {}", key);
    }

    Ok(cache)
}

fn load_eval_files(
    reference_dir: &PathBuf,
    content_key: &str,
) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let eval_files = expand_dirs(
        vec![reference_dir.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;

    for file_path in eval_files {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            // For .gz files, use file_stem to get the .jsonl name
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // For regular files, use file_stem to get the name without .jsonl
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for line in data.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = serde_json::from_str(&line)?;
                let text = match get_nested_json_val(&json_obj, &content_key.to_string()) {
                    Ok(t) => t,
                    Err(_) => {
                        // Skip files that don't have the expected schema
                        eprintln!(
                            "Skipping file {:?} - doesn't have expected key '{}'",
                            file_path, content_key
                        );
                        break;
                    }
                };
                lines.push(text);
            }
        }

        cache.insert(file_name, lines);
    }

    Ok(cache)
}

fn truncate_text(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();

    if lines.len() <= max_lines {
        text.to_string()
    } else {
        let truncated_lines = &lines[..max_lines];
        let mut result = truncated_lines.join("\n");
        result.push_str(&format!(
            "\n... [truncated: showing {} of {} lines, use --full to see all]",
            max_lines,
            lines.len()
        ));
        result
    }
}

/// Extract contaminated text segment using token indices with context
fn extract_contaminated_segment_with_context(
    original_text: &str,
    start_idx: usize,
    end_idx: usize,
    config: &Config,
    context_words: usize,
) -> Option<String> {
    // Initialize tokenizer based on config
    let tokenizer = match OmniTokenizer::new(&config.tokenizer_str) {
        Ok(t) => t,
        Err(_) => return None,
    };

    // First, clean the text the same way as during detection
    let cleaned_text = clean_text(original_text, &config.punctuation_chars);

    // For word tokenizer, tokens are just words split by whitespace
    if config.tokenizer_str == "word" {
        let words: Vec<&str> = cleaned_text.split_whitespace().collect();

        // Check if indices are valid
        if start_idx >= words.len() || end_idx > words.len() || start_idx >= end_idx {
            return None;
        }

        // Calculate context boundaries
        let context_start = start_idx.saturating_sub(context_words);
        let context_end = (end_idx + context_words).min(words.len());

        // Build the output with highlighted contamination
        let mut result = String::new();

        // Add leading context
        if context_start < start_idx {
            result.push_str("... ");
            result.push_str(&words[context_start..start_idx].join(" "));
            result.push_str(" ");
        }

        // Add contaminated section with highlighting
        result.push_str("【");
        result.push_str(&words[start_idx..end_idx].join(" "));
        result.push_str("】");

        // Add trailing context
        if end_idx < context_end {
            result.push_str(" ");
            result.push_str(&words[end_idx..context_end].join(" "));
            result.push_str(" ...");
        }

        return Some(result);
    }

    // For other tokenizers, we need to tokenize and then decode
    let Ok(tokens) = std::panic::catch_unwind(|| tokenizer.encode(&cleaned_text)) else {
        return None;
    };

    // Check if indices are valid
    if start_idx >= tokens.len() || end_idx > tokens.len() || start_idx >= end_idx {
        return None;
    }

    // For BPE tokenizers, try to decode with context
    if let Some(inner) = tokenizer.inner.as_ref() {
        let context_start = start_idx.saturating_sub(context_words);
        let context_end = (end_idx + context_words).min(tokens.len());

        let mut result = String::new();

        // Decode and add leading context
        if context_start < start_idx {
            if let Ok(prefix) = inner.decode(tokens[context_start..start_idx].to_vec()) {
                result.push_str("... ");
                result.push_str(&prefix);
                result.push_str(" ");
            }
        }

        // Decode and add contaminated section
        if let Ok(contaminated) = inner.decode(tokens[start_idx..end_idx].to_vec()) {
            result.push_str("【");
            result.push_str(&contaminated);
            result.push_str("】");
        }

        // Decode and add trailing context
        if end_idx < context_end {
            if let Ok(suffix) = inner.decode(tokens[end_idx..context_end].to_vec()) {
                result.push_str(" ");
                result.push_str(&suffix);
                result.push_str(" ...");
            }
        }

        if !result.is_empty() {
            return Some(result);
        }
    }

    // Fallback: show token IDs
    let token_str = tokens[start_idx..end_idx]
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Some(format!("[Tokens: {}]", token_str))
}

fn display_contamination_case(
    result: &ContaminationResult,
    training_cache: &HashMap<String, Vec<String>>,
    eval_cache: &HashMap<String, Vec<String>>,
    ground_truth: &[GroundTruthRecord],
    full: bool,
    config: &Config,
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
                println!("🧪 TOXIC SCORE:    {:.3}", result.toxic_score);
            }
            if let Some(ngram_match_cnt) = result.ngram_match_cnt {
                println!("🔢 N-GRAM MATCHES: {}", ngram_match_cnt);
            }
            println!();
        }
    }

    // Get training text
    let training_text = match training_cache.get(&result.training_file) {
        Some(lines) => {
            if result.training_line < lines.len() {
                &lines[result.training_line]
            } else {
                "❌ Training line index out of bounds"
            }
        }
        None => "❌ Training file not found",
    };

    // Get eval text (or ground truth for false negatives)
    let eval_text = match result.method.as_deref() {
        Some("false_negative") => {
            // For false negatives, find the ground truth text
            ground_truth
                .iter()
                .find(|gt| gt.text == *training_text)
                .map(|gt| gt.ground_truth.as_str())
                .unwrap_or("❌ Ground truth not found")
        }
        _ => match eval_cache.get(&result.eval_dataset) {
            Some(lines) => {
                if result.eval_line < lines.len() {
                    &lines[result.eval_line]
                } else {
                    "❌ Eval line index out of bounds"
                }
            }
            None => "❌ Eval file not found",
        },
    };

    // Display side by side
    println!("🔍 TRAINING TEXT (line {}):", result.training_line);

    // Apply truncation if not in full mode
    let displayed_text = if full {
        training_text.to_string()
    } else {
        truncate_text(training_text, 25)
    };

    println!("   \"{}\"", displayed_text);

    // Show the actual contaminated text segment
    // For toxic mode, use the pre-computed overlap text
    if let Some(ref overlap_text) = result.training_overlap_text {
        println!();
        println!(
            "📍 CONTAMINATED SEGMENT (tokens {} to {}):",
            result.contamination_start_idx.unwrap_or(0),
            result.contamination_end_idx.unwrap_or(0)
        );
        println!("   {}", overlap_text);
    } else if let (Some(start_idx), Some(end_idx)) =
        (result.contamination_start_idx, result.contamination_end_idx)
    {
        // For other modes, try to extract the segment
        println!();
        println!(
            "📍 CONTAMINATED SEGMENT (tokens {} to {}):",
            start_idx, end_idx
        );

        // Try to extract the contaminated segment with 10 words of context on each side
        if let Some(segment) =
            extract_contaminated_segment_with_context(training_text, start_idx, end_idx, config, 10)
        {
            println!("   {}", segment);
        } else {
            println!(
                "   [Unable to extract segment - {} tokens]",
                end_idx - start_idx
            );
        };
    }

    println!();

    match result.method.as_deref() {
        Some("false_negative") => {
            println!("🔍 GROUND TRUTH (expected):");
            println!("   \"{}\"", eval_text);
        }
        _ => {
            println!("🔍 EVAL TEXT (line {}):", result.eval_line);
            // Use pre-computed eval_overlap_text if available (from toxic mode)
            if let Some(ref overlap_text) = result.eval_overlap_text {
                println!("   \"{}\"", overlap_text);
            } else {
                println!("   \"{}\"", eval_text);
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
            // Check if they're identical
            if training_text == eval_text {
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
