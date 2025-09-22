use crate::review::ContaminationResult;
use anyhow::{Error, Result};

// Helper function to convert bracket highlights to ANSI colored formatting
// Passages use cyan (36m), questions use red (31m), answers use purple/magenta (35m)
pub fn format_with_bold_highlights(text: &str) -> String {
    text.replace("⟨", "\x1b[36m")       // Passage start - cyan
        .replace("⟩", "\x1b[0m")        // Passage end - reset
        .replace("【", "\x1b[31m")      // Question start - red
        .replace("】", "\x1b[0m")       // Question end - reset
        .replace("⟦", "\x1b[35m")       // Answer start - purple/magenta
        .replace("⟧", "\x1b[0m")        // Answer end - reset
}

/// Display a single contamination case with all its details
pub fn display_contamination_case(result: &ContaminationResult, verbose: bool) -> Result<(), Error> {
    println!("Eval Dataset:   {}", result.eval_key.as_ref().unwrap_or(&result.eval_dataset));
    if let Some(split) = &result.split {
        println!("Eval Split:     {}", split);
    }
    println!();
    println!("Training File:  {} (line {})", result.training_file, result.training_line);
    if let Some(ref_file) = &result.reference_file {
        println!("Reference File: {} (line {})", ref_file, result.eval_line);
    } else {
        // Fallback to eval_dataset if reference_file not available
        println!("Reference File: {} (line {})", result.eval_dataset, result.eval_line);
    }
    println!();

    if let Some(score) = result.contamination_score {
        println!("CONTAMINATION SCORE: {:.3}\n", score);
    }

    if verbose {
        display_verbose_metrics(result);
    }

    // Display training information
    println!("TRAINING:");

    // Show the training overlap text if available
    if let Some(ref overlap_text) = result.training_overlap_text {
        // Only show token boundaries in verbose mode
        if verbose {
            display_token_boundaries(result);
        }

        println!("   \"{}\"", format_with_bold_highlights(overlap_text));
    }
    println!();

    display_eval_text(result);
    println!();

    display_contamination_assessment(result);

    Ok(())
}

fn display_verbose_metrics(result: &ContaminationResult) {
    // Show split information if available
    if let Some(ref split) = result.split {
        println!("SPLIT: {}", split);
    }

    if let Some(idf_overlap) = result.idf_overlap {
        println!("IDF OVERLAP:    {:.3}", idf_overlap);
    }

    if let Some(ngram_jaccard) = result.ngram_jaccard {
        println!("N-GRAM JACCARD: {:.3}\n", ngram_jaccard);
    }

    if let Some(ngram_match_cnt) = result.ngram_match_cnt {
        println!("PROMPT N-GRAM MATCHES: {}", ngram_match_cnt);
    }
    if let Some(eval_unique_ngrams) = result.eval_unique_ngrams {
        println!("PROMPT UNIQUE N-GRAMS: {}\n", eval_unique_ngrams);
    }

    // Display token length information
    if let Some(delta) = result.token_length_delta {
        let delta_str = if delta > 0 {
            format!("+{}", delta)
        } else {
            delta.to_string()
        };
        println!(
            "TOKEN LENGTH DELTA: {} (cluster: {}, eval: {})",
            delta_str,
            result.cluster_token_length.unwrap_or(0),
            result.eval_token_length.unwrap_or(0)
        );
    }

    // Display answer contamination information
    if let Some(answer_ratio) = result.answer_overlap_ratio {
        println!("ANSWER OVERLAP RATIO: {:.3}", answer_ratio);
    }
    if let Some(answer_idf) = result.answer_idf_overlap {
        println!("ANSWER IDF OVERLAP: {:.3}", answer_idf);
    }

    // Display passage contamination information
    if let Some(passage_ratio) = result.passage_overlap_ratio {
        println!("PASSAGE OVERLAP RATIO: {:.3}", passage_ratio);
    }
    if let Some(passage_idf) = result.passage_idf_overlap {
        println!("PASSAGE IDF OVERLAP: {:.3}", passage_idf);
    }

    if let Some(threshold) = result.length_adjusted_question_threshold {
        println!("LENGTH-ADJUSTED THRESHOLD: {:.3}", threshold);
    }

    // Display fingerprint and is_correct if available
    if let Some(ref fingerprint) = result.fingerprint {
        println!("FINGERPRINT: {}", fingerprint);
    }
    if let Some(is_correct) = result.is_correct {
        let correctness_indicator = if is_correct { "✅" } else { "❌" };
        println!("ANSWER CORRECTNESS: {} {}", correctness_indicator, is_correct);
    }

    println!();
}

fn display_token_boundaries(result: &ContaminationResult) {
    // Display question boundaries
    if let (Some(start_idx), Some(end_idx)) =
        (result.question_start_idx, result.question_end_idx)
    {
        print!("   Question tokens: {} to {}", start_idx, end_idx);
    }

    // Display passage boundaries if available
    if let (Some(passage_start), Some(passage_end)) =
        (result.passage_start_idx, result.passage_end_idx)
    {
        print!(", Passage tokens: {} to {}", passage_start, passage_end);
    }

    // Display answer boundaries if available
    if let (Some(answer_start), Some(answer_end)) =
        (result.answer_start_idx, result.answer_end_idx)
    {
        print!(", Answer tokens: {} to {}", answer_start, answer_end);
    }
    println!();
}

fn display_eval_text(result: &ContaminationResult) {
    println!("EVAL TEXT:");

    // Display three separate sections if available
    let has_separate_texts = result.eval_question_text.is_some()
        || result.eval_answer_text.is_some()
        || result.eval_passage_text.is_some();

    if has_separate_texts {
        // Display passage if available
        if let Some(ref passage_text) = result.eval_passage_text {
            // Use cyan color for passage label
            println!("   \x1b[36m[PASSAGE]\x1b[0m: {}", passage_text);
            println!();
        }

        // Display question/prompt if available
        if let Some(ref question_text) = result.eval_question_text {
            // Use red color for prompt label (matching question highlighting)
            println!("   \x1b[31m[PROMPT]\x1b[0m: {}", question_text);
            println!();
        }

        // Display answer if available
        if let Some(ref answer_text) = result.eval_answer_text {
            // Use purple/magenta color for answer label
            println!("   \x1b[35m[ANSWER]\x1b[0m: {}", answer_text);
        }
    } else if let Some(ref overlap_text) = result.eval_overlap_text {
        // Fallback to single text display for backward compatibility
        // Use cyan color for eval text - good readability on dark backgrounds
        println!(
            "   \"\x1b[36m{}\x1b[0m\"",
            format_with_bold_highlights(overlap_text)
        );
    } else {
        println!("   [No eval overlap text available in results]");
    }
}

fn display_contamination_assessment(result: &ContaminationResult) {
    // Check contamination based on the contamination_score field
    if let Some(score) = result.contamination_score {
        if score >= 1.0 {
            println!("✅ EXACT MATCH - Definite contamination");
        } else if score > 0.9 {
            println!("⚠️  VERY HIGH SIMILARITY - Likely contamination");
        } else if score > 0.8 {
            println!("⚠️  HIGH SIMILARITY - Likely contamination");
        } else if score > 0.7 {
            println!("🤔 MODERATE SIMILARITY - Manual review needed");
        } else {
            println!("🔍 LOW SIMILARITY - Edge case detection");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_with_bold_highlights_passages() {
        let input = "This is ⟨highlighted passage⟩ text";
        let expected = "This is \x1b[36mhighlighted passage\x1b[0m text";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_questions() {
        let input = "This is 【highlighted question】 text";
        let expected = "This is \x1b[31mhighlighted question\x1b[0m text";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_answers() {
        let input = "This is ⟦highlighted answer⟧ text";
        let expected = "This is \x1b[35mhighlighted answer\x1b[0m text";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_mixed() {
        let input = "⟨passage⟩ and 【question】 and ⟦answer⟧";
        let expected = "\x1b[36mpassage\x1b[0m and \x1b[31mquestion\x1b[0m and \x1b[35manswer\x1b[0m";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_nested() {
        let input = "⟨outer ⟦inner⟧ outer⟩";
        let expected = "\x1b[36mouter \x1b[35minner\x1b[0m outer\x1b[0m";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_empty() {
        let input = "";
        let expected = "";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_no_markers() {
        let input = "Plain text without markers";
        let expected = "Plain text without markers";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_empty_markers() {
        let input = "Empty ⟨⟩ markers 【】 here ⟦⟧";
        let expected = "Empty \x1b[36m\x1b[0m markers \x1b[31m\x1b[0m here \x1b[35m\x1b[0m";
        assert_eq!(format_with_bold_highlights(input), expected);
    }

    #[test]
    fn test_format_with_bold_highlights_unicode_content() {
        let input = "Unicode ⟨こんにちは⟩ content 【世界】 test ⟦🌍⟧";
        let expected = "Unicode \x1b[36mこんにちは\x1b[0m content \x1b[31m世界\x1b[0m test \x1b[35m🌍\x1b[0m";
        assert_eq!(format_with_bold_highlights(input), expected);
    }
}
