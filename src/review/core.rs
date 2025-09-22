use anyhow::{Error, Result};
use std::io;

use super::display::display_contamination_case;
use super::stats::display_eval_dataset_stats;
use super::counts::display_dataset_counts;
use super::top_examples::display_top_eval_examples;
use super::loader::load_contamination_results_from_directory_with_source;
use super::args::ReviewArgs;

pub fn review_contamination(args: &ReviewArgs) -> Result<(), Error> {
    // Extract values from args for backward compatibility
    let dir = args.dir.as_path();
    let step = !args.stats && !args.dump && !args.dataset_counts && args.top_eval_examples.is_none();
    let stats = args.stats;
    let dataset_counts = args.dataset_counts;
    let dump = args.dump;
    let min_score = args.min_score;
    let min_length = args.min_length;
    let eval_filter = args.eval.as_deref();
    let top_eval_examples = args.top_eval_examples;
    let sort_match_length_descending = args.sort_match_length_descending;
    let sort_match_length_ascending = args.sort_match_length_ascending;
    let verbose = args.verbose;

    println!("=== CONTAMINATION REVIEW ===");

    let dir_path = dir;
    if !dir_path.exists() {
        println!("Directory not found: {:?}", dir_path);
        return Err(anyhow::anyhow!("Directory not found"));
    }

    let mut all_results = crate::review::load_contamination_results_from_directory(dir_path)?;
    // Only load with source if needed for dataset_counts
    let all_results_with_source = if dataset_counts {
        load_contamination_results_from_directory_with_source(dir_path)?
    } else {
        Vec::new()
    };

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
    } else if sort_match_length_ascending {
        // Sort by ngram_match_cnt in ascending order (lowest first)
        all_results.sort_by(|a, b| {
            let count_a = a.ngram_match_cnt.unwrap_or(0);
            let count_b = b.ngram_match_cnt.unwrap_or(0);
            count_a.cmp(&count_b) // Note: a compared to b for ascending order
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
    all_results = crate::review::filter_contamination_results_by_thresholds(
        all_results,
        min_score,
        min_length,
        eval_filter,
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
        return Ok(());
    }

    println!(
        "Found {} contamination instances from directory",
        all_results.len()
    );
    if original_count != all_results.len() {
        let filtered_count = original_count - all_results.len();
        let mut filter_reasons = Vec::new();

        if min_score.is_some() {
            filter_reasons.push(format!("min score: {:.3}", min_score.unwrap()));
        }
        if min_length.is_some() {
            filter_reasons.push(format!("min {} n-grams", min_length.unwrap()));
        }
        if eval_filter.is_some() {
            filter_reasons.push(format!("eval dataset: {}", eval_filter.unwrap()));
        }

        if filter_reasons.is_empty() {
            println!("({} filtered out)", filtered_count);
        } else {
            println!(
                "({} filtered out by: {})",
                filtered_count,
                filter_reasons.join(", ")
            );
        }
    }
    println!();

    if stats {
        display_eval_dataset_stats(&all_results)?;
        return Ok(());
    }

    if dataset_counts {
        display_dataset_counts(&all_results_with_source, dir_path)?;
        return Ok(());
    }

    if let Some(top_n) = top_eval_examples {
        display_top_eval_examples(&all_results, top_n)?;
        return Ok(());
    }

    if dump {
        println!("=== DISPLAYING ALL CONTAMINATION CASES ===\n");

        // Review each contamination case without stepping
        for (idx, result) in all_results.iter().enumerate() {
            println!("{}", "=".repeat(80));
            println!("CONTAMINATION #{} of {}", idx + 1, all_results.len());
            println!("{}", "=".repeat(80));

            display_contamination_case(result, verbose)?;
            println!();
        }

        println!("=== END OF RESULTS ===");
        return Ok(());
    }

    // Default to step-by-step review if no specific flag is set (or if --step is explicitly set)
    if step || (!stats && !dataset_counts && !dump && top_eval_examples.is_none()) {
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

            display_contamination_case(result, verbose)?;
            println!();
        }

        println!("=== REVIEW COMPLETE ===");
        return Ok(());
    }

    Ok(())
}
