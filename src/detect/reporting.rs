use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashSet;
use std::fs::create_dir_all;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use mj_io::write_mem_to_pathbuf;

use crate::common::{write_purified_file, Config};

use crate::detect::{
    detection::ContaminationResults,
    reference_index::{EvalTextSnippets, EvalAnswerTextSnippets, EvalPassageTextSnippets},
    scoring::interpolate_threshold,
};
use crate::detect::utils::build_pbar_quiet;

pub fn save_contamination_results(
    config: &Config,
    contamination_results: &ContaminationResults,
    filename: &str,
    _eval_text_snippets: &EvalTextSnippets,
    _eval_answer_text_snippets: &EvalAnswerTextSnippets,
    _eval_passage_text_snippets: &EvalPassageTextSnippets,
) -> Result<PathBuf, Error> {
    let output_file = config.report_output_dir.join(filename);

    if let Some(parent) = output_file.parent() {
        create_dir_all(parent)?;
    }

    let mut output_data = Vec::new();

    for entry in contamination_results.iter() {
        let training_file = entry.key();
        for contamination_entry in entry.value() {
            let mut result = json!({
                "training_file": training_file,
                "training_line": contamination_entry.training_line,
                "eval_dataset": contamination_entry.eval_key,
                "eval_key": contamination_entry.eval_key,
                "eval_line": contamination_entry.eval_line,
                "eval_instance_index": contamination_entry.eval_instance_index,
                "ngram_match_cnt": contamination_entry.ngram_match_cnt,
                "eval_unique_ngrams": contamination_entry.eval_unique_ngrams,
                "contamination_score": contamination_entry.contamination_score.expect("contamination_score should always be set during detection"),
                "length_adjusted_question_threshold": interpolate_threshold(
                    contamination_entry.eval_token_length.unwrap_or(usize::MAX),
                    config.perfect_match_decay_start,
                    config.perfect_match_decay_end,
                    config.minimum_question_idf_threshold,
                ),
                "method": "simple"
            });

            if let Some(idf_overlap) = contamination_entry.idf_overlap {
                result["idf_overlap"] = json!(idf_overlap);
            }
            if let Some(start_idx) = contamination_entry.contamination_start_idx {
                result["question_start_idx"] = json!(start_idx);
            }
            if let Some(end_idx) = contamination_entry.contamination_end_idx {
                result["question_end_idx"] = json!(end_idx);
            }
            if let Some(ref overlap_text) = contamination_entry.training_overlap_text {
                result["training_overlap_text"] = json!(overlap_text);
            }
            if let Some(ref question_text) = contamination_entry.eval_question_text {
                result["eval_question_text"] = json!(question_text);
                result["eval_overlap_text"] = json!(question_text);
            }
            if let Some(ref answer_text) = contamination_entry.eval_answer_text {
                result["eval_answer_text"] = json!(answer_text);
            }
            if let Some(ref passage_text) = contamination_entry.eval_passage_text {
                result["eval_passage_text"] = json!(passage_text);
            }
            if let Some(ratio) = contamination_entry.answer_overlap_ratio {
                result["answer_overlap_ratio"] = json!(ratio);
            }
            if let Some(idf_overlap) = contamination_entry.answer_idf_overlap {
                result["answer_idf_overlap"] = json!(idf_overlap);
            }
            if let Some(start_idx) = contamination_entry.answer_start_idx {
                result["answer_start_idx"] = json!(start_idx);
            }
            if let Some(end_idx) = contamination_entry.answer_end_idx {
                result["answer_end_idx"] = json!(end_idx);
            }
            if let Some(ref boundaries) = contamination_entry.answer_boundaries {
                result["answer_boundaries"] = json!(boundaries);
            }
            if let Some(ratio) = contamination_entry.passage_overlap_ratio {
                result["passage_overlap_ratio"] = json!(ratio);
            }
            if let Some(idf_overlap) = contamination_entry.passage_idf_overlap {
                result["passage_idf_overlap"] = json!(idf_overlap);
            }
            if let Some(cluster_len) = contamination_entry.cluster_token_length {
                result["cluster_token_length"] = json!(cluster_len);
            }
            if let Some(eval_len) = contamination_entry.eval_token_length {
                result["eval_token_length"] = json!(eval_len);
            }
            if let (Some(cluster_len), Some(eval_len)) = (
                contamination_entry.cluster_token_length,
                contamination_entry.eval_token_length,
            ) {
                let delta = cluster_len as i32 - eval_len as i32;
                result["token_length_delta"] = json!(delta);
            }
            if let Some(ref fingerprint) = contamination_entry.fingerprint {
                result["fingerprint"] = json!(fingerprint);
            }
            if let Some(is_correct) = contamination_entry.is_correct {
                result["is_correct"] = json!(is_correct);
            }
            if let Some(ref reference_file) = contamination_entry.reference_file {
                result["reference_file"] = json!(reference_file);
            }
            if let Some(ref split) = contamination_entry.split {
                result["split"] = json!(split);
            }

            output_data.push(serde_json::to_vec(&result)?);
        }
    }

    let mut output_bytes = Vec::new();
    for line in output_data {
        output_bytes.extend(line);
        output_bytes.push(b'\n');
    }

    write_mem_to_pathbuf(&output_bytes, &output_file)?;

    Ok(output_file)
}

pub fn create_purified_files_streaming(config: &Config, contaminated_files: &Arc<DashMap<String, HashSet<usize>>>, training_files: &[PathBuf]) -> Result<(), Error> {
    crate::vprintln!("\nCreating purified files...");

    // Determine output directory for cleaned files
    let cleaned_dir = config
        .cleaned_output_dir
        .as_ref()
        .unwrap_or(&config.report_output_dir);

    let pbar = build_pbar_quiet(training_files.len(), "Purifying files");

    // Track statistics
    let total_files_processed = Arc::new(AtomicUsize::new(0));
    let total_lines_removed = Arc::new(AtomicUsize::new(0));
    let total_clean_files = Arc::new(AtomicUsize::new(0));

    training_files.par_iter().for_each(|file_path| {
        // Match the same logic used in process_training_file
        let file_name = match file_path.extension().and_then(|s| s.to_str()) {
            Some("gz") | Some("zst") | Some("zstd") => file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            _ => file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
        };

        let contaminated_lines = if let Some(entry) = contaminated_files.get(&file_name) {
            entry.value().clone()
        } else {
            HashSet::new()
        };

        let is_clean = contaminated_lines.is_empty();
        let lines_removed = contaminated_lines.len();

        // Always create a purified file when purify mode is enabled
        match write_purified_file(file_path,cleaned_dir, &contaminated_lines,&config.training_dir) {
            Ok(_) => {
                total_files_processed.fetch_add(1, Ordering::Relaxed);
                if is_clean {
                    total_clean_files.fetch_add(1, Ordering::Relaxed);
                } else {
                    total_lines_removed.fetch_add(lines_removed, Ordering::Relaxed);
                }
            }
            Err(e) => {
                eprintln!("Error purifying file {:?}: {:?}", file_path, e);
            }
        }

        pbar.inc(1);
    });

    pbar.finish_with_message("Purification complete");

    // Print summary statistics
    let files_processed = total_files_processed.load(Ordering::Relaxed);
    let clean_files = total_clean_files.load(Ordering::Relaxed);
    let contaminated_files_count = files_processed - clean_files;
    let lines_removed = total_lines_removed.load(Ordering::Relaxed);

    crate::vprintln!("\nPurification Summary:");
    crate::vprintln!("  Total files processed: {}", files_processed);
    crate::vprintln!("  Clean files copied: {}", clean_files);
    crate::vprintln!(
        "  Contaminated files purified: {}",
        contaminated_files_count
    );
    if contaminated_files_count > 0 {
        crate::vprintln!("  Total lines removed: {}", lines_removed);
    }

    Ok(())
}
