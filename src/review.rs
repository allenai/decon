use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;

use mj_io::{expand_dirs, read_pathbuf_to_mem};

// Helper function to convert bracket highlights to ANSI red formatting
fn format_with_bold_highlights(text: &str) -> String {
    text.replace("【", "\x1b[31m").replace("】", "\x1b[0m")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminationResult {
    pub training_file: String,
    pub training_line: usize,
    pub eval_dataset: String,
    pub eval_line: usize,
    #[serde(alias = "overlap_ratio")]
    pub jaccard_similarity: f32,
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
    #[serde(default)]
    pub eval_unique_ngrams: Option<usize>,
    #[serde(default)]
    pub contamination_score: Option<f32>,
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub answer_overlap_ratio: Option<f32>,
    #[serde(default)]
    pub answer_idf_overlap: Option<f32>,
    #[serde(default)]
    pub matched_answer_tokens: Option<Vec<String>>,
    #[serde(default)]
    pub idf_overlap: Option<f32>,
    #[serde(default)]
    pub cluster_token_length: Option<usize>,
    #[serde(default)]
    pub eval_token_length: Option<usize>,
    #[serde(default)]
    pub token_length_delta: Option<i32>,
    #[serde(default)]
    pub ngram_jaccard: Option<f32>,
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

pub fn review_contamination(
    _config: Option<&PathBuf>,
    dir: Option<&PathBuf>,
    step: bool,
    stats: bool,
    all: bool,
    min_score: Option<f32>,
    min_length: Option<usize>,
    eval_filter: Option<&str>,
    skip_exact: bool,
    top_eval_examples: Option<usize>,
    sort_match_length_descending: bool,
) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    // Directory is now required
    if dir.is_none() {
        return Err(anyhow::anyhow!("--dir is required for review command"));
    }

    let dir_path = dir.unwrap();
    if !dir_path.exists() {
        println!("Directory not found: {:?}", dir_path);
        return Err(anyhow::anyhow!("Directory not found"));
    }

    // Load all result files from directory
    let mut all_results = load_contamination_results_from_directory(dir_path)?;

    if all_results.is_empty() {
        println!(
            "No contamination results found in directory: {:?}",
            dir_path
        );
        return Ok(());
    }

    // Sort results based on flag
    if sort_match_length_descending {
        // Sort by ngram_match_cnt in descending order (highest first)
        all_results.sort_by(|a, b| {
            let count_a = a.ngram_match_cnt.unwrap_or(0);
            let count_b = b.ngram_match_cnt.unwrap_or(0);
            count_b.cmp(&count_a) // Note: b compared to a for descending order
        });
    } else {
        // Default: Sort by contamination_score in ascending order
        all_results.sort_by(|a, b| {
            let score_a = a.contamination_score.unwrap_or(0.0);
            let score_b = b.contamination_score.unwrap_or(0.0);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let original_count = all_results.len();

    // Apply filters if specified
    all_results = filter_contamination_results_by_thresholds(
        all_results,
        min_score,
        min_length,
        eval_filter,
        skip_exact,
    );

    if all_results.is_empty() {
        println!("No contamination results matched the filter criteria.");
        println!("Original count: {}", original_count);
        if min_score.is_some() {
            println!("  - Minimum contamination score: {:.3}", min_score.unwrap());
        }
        if min_length.is_some() {
            println!("  - Minimum n-gram matches: {}", min_length.unwrap());
        }
        if eval_filter.is_some() {
            println!("  - Eval dataset filter: {}", eval_filter.unwrap());
        }
        if skip_exact {
            println!("  - Skipping exact matches (contamination_score == 1.0)");
        }
        return Ok(());
    }

    println!(
        "Found {} contamination instances from directory",
        all_results.len()
    );
    if original_count != all_results.len() {
        println!(
            "({} filtered out by threshold criteria)",
            original_count - all_results.len()
        );
    }
    println!();

    if stats {
        // Display eval dataset statistics with bar chart
        display_eval_dataset_stats(&all_results)?;
        return Ok(());
    }

    if let Some(top_n) = top_eval_examples {
        // Display top N most commonly matched eval examples
        display_top_eval_examples(&all_results, top_n)?;
        return Ok(());
    }

    // Display all results at once if --all flag is set
    if all {
        println!("=== DISPLAYING ALL CONTAMINATION CASES ===\n");

        // Review each contamination case without stepping
        for (idx, result) in all_results.iter().enumerate() {
            println!("{}", "=".repeat(80));
            println!("CONTAMINATION #{} of {}", idx + 1, all_results.len());
            println!("{}", "=".repeat(80));

            display_contamination_case_without_config(result)?;
            println!();
        }

        println!("=== END OF RESULTS ===");
        return Ok(());
    }

    // Default to step-by-step review if no specific flag is set (or if --step is explicitly set)
    if step || (!stats && !all && top_eval_examples.is_none()) {
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

    // This should not be reached anymore since we default to step mode
    Ok(())
}

#[allow(dead_code)]
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
                    method: Some("true_negative".to_string()),
                    matching_ngrams: None,
                    bucket_sizes: None,
                    bucket_ids: None,
                    contamination_start_idx: None,
                    contamination_end_idx: None,
                    training_overlap_text: None,
                    eval_overlap_text: None,
                    ngram_match_cnt: None,
                    eval_unique_ngrams: None,
                    contamination_score: None,
                    length_penalty: None,
                    answer_overlap_ratio: None,
                    answer_idf_overlap: None,
                    matched_answer_tokens: None,
                    idf_overlap: None,
                    cluster_token_length: None,
                    eval_token_length: None,
                    token_length_delta: None,
                    ngram_jaccard: None,
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
    min_score: Option<f32>,
    min_length: Option<usize>,
    eval_filter: Option<&str>,
    skip_exact: bool,
) -> Vec<ContaminationResult> {
    results
        .into_iter()
        .filter(|result| {
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

            // Check contamination score
            if let Some(min_score_threshold) = min_score {
                if let Some(score) = result.contamination_score {
                    if score < min_score_threshold {
                        return false;
                    }
                } else {
                    // If contamination_score is not present, filter out
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

            // Skip exact matches if requested
            if skip_exact {
                if let Some(score) = result.contamination_score {
                    if score == 1.0 {
                        return false;
                    }
                }
            }

            true
        })
        .collect()
}

fn load_contamination_results_from_directory(
    dir_path: &PathBuf,
) -> Result<Vec<ContaminationResult>, Error> {
    let mut all_results = Vec::new();

    // Find all .jsonl files in the directory
    let jsonl_files = expand_dirs(vec![dir_path.clone()], Some(vec![".jsonl"].as_slice()))?;

    println!(
        "Processing {} JSONL files from directory...",
        jsonl_files.len()
    );

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

fn display_contamination_case_without_config(result: &ContaminationResult) -> Result<(), Error> {
    // This is a simplified version that doesn't need ground truth or config
    display_contamination_case_internal(result)
}

fn display_contamination_case_internal(result: &ContaminationResult) -> Result<(), Error> {
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
            println!("📋 EVAL DATASET:  {}\n", result.eval_dataset);
            let similarity_label = match result.method.as_deref() {
                Some("toxic") | Some("simple") => "OVERLAP RATIO",
                _ => "JACCARD SIM",
            };
            println!(
                "🎯 {}:   {:.3}",
                similarity_label, result.jaccard_similarity
            );
            if let Some(idf_overlap) = result.idf_overlap {
                println!("📈 IDF OVERLAP:    {:.3}", idf_overlap);
            }
            if let Some(ngram_jaccard) = result.ngram_jaccard {
                println!("🔗 N-GRAM JACCARD: {:.3}\n", ngram_jaccard);
            }

            if let Some(ngram_match_cnt) = result.ngram_match_cnt {
                println!("🔢 N-GRAM MATCHES: {}", ngram_match_cnt);
            }
            if let Some(eval_unique_ngrams) = result.eval_unique_ngrams {
                println!("📊 EVAL UNIQUE N-GRAMS: {}\n", eval_unique_ngrams);
            }

            // if let Some(penalty) = result.length_penalty {
            //     println!("📏 LENGTH PENALTY: {:.3}\n", penalty);
            // }
            // Display token length information
            if let Some(delta) = result.token_length_delta {
                let delta_str = if delta > 0 {
                    format!("+{}", delta)
                } else {
                    delta.to_string()
                };
                println!(
                    "📐 TOKEN LENGTH DELTA: {} (cluster: {}, eval: {})",
                    delta_str,
                    result.cluster_token_length.unwrap_or(0),
                    result.eval_token_length.unwrap_or(0)
                );
            }

            // Display answer contamination information
            if let Some(answer_ratio) = result.answer_overlap_ratio {
                println!("🎯 ANSWER OVERLAP RATIO: {:.3}", answer_ratio);
            }
            if let Some(answer_idf) = result.answer_idf_overlap {
                println!("📊 ANSWER IDF OVERLAP: {:.3}", answer_idf);
            }
            if let Some(ref answer_tokens) = result.matched_answer_tokens {
                if !answer_tokens.is_empty() {
                    let display_tokens: Vec<String> = answer_tokens
                        .iter()
                        .map(|token| {
                            if token.trim().is_empty() {
                                "<whitespace>".to_string()
                            } else {
                                token.clone()
                            }
                        })
                        .collect();
                    println!("📝 MATCHED ANSWER TOKENS: {}\n", display_tokens.join(" "));
                }
            }

            if let Some(score) = result.contamination_score {
                println!("⚡ CONTAMINATION SCORE: {:.3}", score);
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
        println!("   \"{}\"", format_with_bold_highlights(overlap_text));
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
                // Use cyan color for eval text - good readability on dark backgrounds
                println!(
                    "   \"\x1b[36m{}\x1b[0m\"",
                    format_with_bold_highlights(overlap_text)
                );
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
    println!(
        "Total contamination incidents: {:?}",
        contamination_results.len()
    );
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
        println!("  {:<45} {:>8} │{}{}│", suite, count, bar, empty);
    }

    println!();
    println!();

    // N-gram match count histogram
    println!("=== N-GRAM MATCH COUNT DISTRIBUTION ===");
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
        println!("No n-gram match count data available.");
        return Ok(());
    }

    // Calculate statistics
    ngram_counts.sort();
    let min_count = *ngram_counts.first().unwrap();
    let max_count = *ngram_counts.last().unwrap();
    let median_count = if ngram_counts.len() % 2 == 0 {
        (ngram_counts[ngram_counts.len() / 2 - 1] + ngram_counts[ngram_counts.len() / 2]) / 2
    } else {
        ngram_counts[ngram_counts.len() / 2]
    };
    let avg_count = ngram_counts.iter().sum::<usize>() as f64 / ngram_counts.len() as f64;

    println!("Total results with n-gram match data: {}", ngram_counts.len());
    if missing_count > 0 {
        println!("Results without n-gram match data: {}", missing_count);
    }
    println!();
    println!("Statistics:");
    println!("  Min:    {}", min_count);
    println!("  Max:    {}", max_count);
    println!("  Median: {}", median_count);
    println!("  Average: {:.1}", avg_count);
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
        let count = ngram_counts.iter().filter(|&&c| c >= *min && c <= *max).count();
        if count > 0 {
            bucket_counts.push((label.to_string(), count));
        }
    }

    // Find the maximum bucket count for scaling
    let max_bucket_count = bucket_counts.iter().map(|(_, count)| *count).max().unwrap_or(0);

    println!("N-gram match count distribution:");
    println!();

    // Display histogram
    for (label, count) in &bucket_counts {
        // Calculate bar length proportional to count
        let bar_length = if max_bucket_count > 0 {
            ((*count as f64 / max_bucket_count as f64) * bar_width as f64) as usize
        } else {
            0
        };

        // Create the bar using Unicode block characters
        let bar = "█".repeat(bar_length);
        let empty = " ".repeat(bar_width - bar_length);

        // Format the output with aligned columns
        println!("  {:<15} {:>8} │{}{}│", label, count, bar, empty);
    }

    println!();
    println!();

    Ok(())
}

fn display_top_eval_examples(
    contamination_results: &[ContaminationResult],
    top_n: usize,
) -> Result<(), Error> {
    // Count occurrences of each (eval_dataset, eval_line) pair
    let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();

    for result in contamination_results {
        let key = (result.eval_dataset.clone(), result.eval_line);
        *eval_counts.entry(key).or_insert(0) += 1;
    }

    // Sort by count (descending)
    let mut sorted_counts: Vec<((String, usize), usize)> = eval_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("=== TOP {} MOST COMMONLY MATCHED EVAL EXAMPLES ===", top_n);
    println!();
    println!(
        "Total contamination incidents: {}",
        contamination_results.len()
    );
    println!(
        "Showing top {} most frequent eval matches:",
        top_n.min(sorted_counts.len())
    );
    println!();

    // Find the maximum count for scaling the bar chart
    let max_count = sorted_counts.first().map(|(_, count)| *count).unwrap_or(0);
    let bar_width = 40; // Width of the bar chart in characters

    // Display summary table
    println!(
        "{:<5} {:<8} {:<40} {:<8} Bar Chart",
        "Rank", "Count", "Eval Dataset", "Line"
    );
    println!("{}", "-".repeat(80));

    for (rank, ((eval_dataset, eval_line), count)) in sorted_counts.iter().take(top_n).enumerate() {
        // Calculate bar length proportional to count
        let bar_length = if max_count > 0 {
            ((*count as f64 / max_count as f64) * bar_width as f64) as usize
        } else {
            0
        };

        // Create the bar using Unicode block characters
        let bar = "█".repeat(bar_length);

        // Format the output with aligned columns
        println!(
            "{:<5} {:<8} {:<40} {:<8} {}",
            rank + 1,
            count,
            eval_dataset,
            eval_line,
            bar
        );
    }

    println!();
    println!("Detailed view of top matches:");
    println!();

    // Display detailed information for each top match
    for (rank, ((eval_dataset, eval_line), count)) in sorted_counts.iter().take(top_n).enumerate() {
        println!("{}", "=".repeat(80));
        println!(
            "Top matched eval example #{} ({} occurrences):",
            rank + 1,
            count
        );
        println!("  Dataset: {}, Line: {}", eval_dataset, eval_line);

        // Find the first matching result to get the eval text
        let eval_text = contamination_results
            .iter()
            .find(|r| &r.eval_dataset == eval_dataset && r.eval_line == *eval_line)
            .and_then(|r| r.eval_overlap_text.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("[No eval text available]");

        println!("  Text: \"{}\"", eval_text);

        // Show a few example training files that matched this eval
        let matching_files: Vec<&str> = contamination_results
            .iter()
            .filter(|r| &r.eval_dataset == eval_dataset && r.eval_line == *eval_line)
            .map(|r| r.training_file.as_str())
            .take(3)
            .collect();

        if !matching_files.is_empty() {
            println!("  Example training files that matched:");
            for file in matching_files {
                println!("    - {}", file);
            }
            if *count > 3 {
                println!("    ... and {} more", count - 3);
            }
        }
    }

    println!();
    println!("=== END OF TOP EVAL EXAMPLES ===");

    Ok(())
}
