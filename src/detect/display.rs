use std::time::Duration;

/// Format a number with comma separators
fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count == 3 {
            result.insert(0, ',');
            count = 0;
        }
        result.insert(0, c);
        count += 1;
    }

    result
}

/// Display index building results in a formatted table
pub fn display_index_building_results(
    stats: &crate::detect::reference_index::IndexBuildingStats,
    index: &crate::detect::reference_index::SimpleReferenceIndex,
    index_time: Duration,
) {
    // Extract values from structs for cleaner code below
    let total_lines_examined = stats.total_lines_examined;
    let lines_indexed = stats.lines_indexed;
    let skipped_duplicates = stats.skipped_duplicates;
    let skipped_min_tokens = stats.skipped_min_tokens;
    let skipped_min_unique_words = stats.skipped_min_unique_words;
    let unique_ngrams = index.question_ngram_to_id.len();
    let hot_ngrams = stats.hot_buckets_replaced;
    let eval_suites = index.unique_eval_suites.len();

    println!("\n\n┌──────────────────────────────────────────────────────────────────┐");
    println!("│                  Reference Index Building Summary                │");
    println!("├──────────────────────────────────────────────────────────────────┤");

    // Reference preprocessing section
    let total_skipped = skipped_duplicates + skipped_min_tokens + skipped_min_unique_words;

    if total_lines_examined > 0 {
        let examined_str = format_with_commas(total_lines_examined);
        let padding = 41 - examined_str.len();  // Fixed: reduced by 6
        println!("│   Total lines examined{:width$} {} │", "", examined_str, width = padding);

        let indexed_str = format_with_commas(lines_indexed);
        let padding = 48 - indexed_str.len();  // Fixed: reduced by 4
        println!("│   Lines indexed{:width$} {} │", "", indexed_str, width = padding);

        if total_skipped > 0 {
            let skipped_str = format_with_commas(total_skipped).to_string();
            let padding = 48 - skipped_str.len();  // Fixed: reduced by 4
            println!("│   Lines skipped{:width$} {} │", "", skipped_str, width = padding);

            if skipped_duplicates > 0 {
                let dup_str = format_with_commas(skipped_duplicates);
                let padding = 47 - dup_str.len();  // Fixed: reduced by 3
                println!("│     - Duplicates{:width$} {} │", "", dup_str, width = padding);
            }
            if skipped_min_tokens > 0 {
                let tok_str = format_with_commas(skipped_min_tokens);
                let padding = 37 - tok_str.len();  // Fixed: reduced by 2
                println!("│     - Below minimum tokens{:width$} {} │", "", tok_str, width = padding);
            }
            if skipped_min_unique_words > 0 {
                let word_str = format_with_commas(skipped_min_unique_words);
                let padding = 35 - word_str.len();  // 70 total - 30 (text) - 3 (spaces) - 2 (borders) = 35
                println!("│     - Below min unique words{:width$} {} │", "", word_str, width = padding);
            }
        }

        println!("├──────────────────────────────────────────────────────────────────┤");
    }

    // Index statistics section
    let ngram_str = format_with_commas(unique_ngrams);
    let padding = 47 - ngram_str.len();  // Fixed: reduced by 4
    println!("│   Unique n-grams{:width$} {} │", "", ngram_str, width = padding);

    let hot_str = format_with_commas(hot_ngrams);
    let padding = 50 - hot_str.len();  // Fixed: reduced by 4
    println!("│   Hot n-grams{:width$} {} │", "", hot_str, width = padding);

    let suite_str = if eval_suites == 1 {
        "1 suite".to_string()
    } else {
        format!("{} suites", eval_suites)
    };
    let padding = 48 - suite_str.len();  // Fixed: reduced by 4
    println!("│   Eval datasets{:width$} {} │", "", suite_str, width = padding);

    println!("├──────────────────────────────────────────────────────────────────┤");

    // Build performance section
    let time_str = format!("{:.2}s", index_time.as_secs_f64());
    let padding = 51 - time_str.len();  // Fixed: reduced by 5
    println!("│   Build time{:width$} {} │", "", time_str, width = padding);

    println!("└──────────────────────────────────────────────────────────────────┘\n");
}

/// Display detection results in a formatted table
pub fn display_detection_results(
    index_time: Duration,
    detection_time: Duration,
    total_time: Duration,
    total_contaminations: usize,
    _contaminated_documents: usize,
    contaminated_lines: usize,
    lines_processed: usize,
) {
    println!("\n\n┌───────────────────────────────────────────┐");
    println!("│     Contamination Detection Results       │");
    println!("├───────────────────────────────────────────┤");

    // Processing stats section
    let lines_str = format_with_commas(lines_processed);
    let padding = 26 - lines_str.len();  // 43 total - 15 ("Training lines") - 2 spaces = 26
    println!("│ Training lines{:width$} {} │", "", lines_str, width = padding);

    // Calculate throughput
    let micros_per_line = if lines_processed > 0 {
        (detection_time.as_micros() as f64) / lines_processed as f64
    } else {
        0.0
    };

    if micros_per_line >= 1000.0 {
        let rate_str = format!("{:.1} ms/doc", micros_per_line / 1000.0);
        let padding = 25 - rate_str.len();  // 43 total - 16 ("Processing rate") - 1 space before = 26
        println!("│ Processing rate{:width$} {} │", "", rate_str, width = padding);
    } else if micros_per_line > 0.0 {
        let rate_str = format!("{:.0} μs/doc", micros_per_line);
        let padding = 26 - rate_str.len();
        println!("│ Processing rate{:width$} {} │", "", rate_str, width = padding);
    }

    println!("├───────────────────────────────────────────┤");

    // Timing section
    let index_str = format!("{:.2}s", index_time.as_secs_f64());
    let index_padding = 21 - index_str.len();  // 43 total - 20 ("Index building time") - 2 spaces = 21
    println!("│ Index building time{:width$} {} │", "", index_str, width = index_padding);

    let detect_str = format!("{:.2}s", detection_time.as_secs_f64());
    let detect_padding = 26 - detect_str.len();  // 43 total - 15 ("Detection time") - 2 spaces = 26
    println!("│ Detection time{:width$} {} │", "", detect_str, width = detect_padding);

    let total_str = format!("{:.2}s", total_time.as_secs_f64());
    let total_padding = 30 - total_str.len();  // 43 total - 11 ("Total time") - 2 spaces = 30
    println!("│ Total time{:width$} {} │", "", total_str, width = total_padding);

    println!("├───────────────────────────────────────────┤");

    // Results section
    let contam_str = format_with_commas(total_contaminations);
    let contam_padding = 20 - contam_str.len();  // 43 total - 21 ("Contaminated matches") - 2 spaces = 20
    println!("│ Contaminated matches{:width$} {} │", "", contam_str, width = contam_padding);

    let docs_str = format_with_commas(contaminated_lines);
    let docs_padding = 18 - docs_str.len();  // 43 total - 22 ("Contaminated documents") - 2 spaces = 19
    println!("│ Contaminated documents{:width$} {} │", "", docs_str, width = docs_padding);

    println!("└───────────────────────────────────────────┘");
}

/// Display completion message with instructions for reviewing results
pub fn display_completion_message(config: &crate::common::Config, total_contaminations: usize) {
    println!();

    if total_contaminations == 0 {
        println!("No contamination detected!");
    } else {
        println!("To review results:");
        println!("  decon review {}", config.report_output_dir.display());
    }

    println!();
}

