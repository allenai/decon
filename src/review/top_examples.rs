use crate::review::ContaminationResult;
use anyhow::{Error, Result};
use std::collections::HashMap;

/// Display top N most commonly matched eval examples
pub fn display_top_eval_examples(
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

        let bar = "█".repeat(bar_length);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::ContaminationResult;

    fn create_test_result(eval_dataset: &str, eval_line: usize, training_file: &str) -> ContaminationResult {
        ContaminationResult {
            training_file: training_file.to_string(),
            training_line: 1,
            eval_dataset: eval_dataset.to_string(),
            eval_key: None,
            eval_line,
            eval_instance_index: None,
            split: None,
            method: None,
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: None,
            eval_overlap_text: Some("test overlap text".to_string()),
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
    fn test_count_eval_examples_basic() {
        let results = vec![
            create_test_result("mmlu", 10, "file1.txt"),
            create_test_result("mmlu", 10, "file2.txt"),
            create_test_result("mmlu", 20, "file3.txt"),
            create_test_result("gsm8k", 30, "file4.txt"),
        ];

        let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();
        for result in &results {
            let key = (result.eval_dataset.clone(), result.eval_line);
            *eval_counts.entry(key).or_insert(0) += 1;
        }

        assert_eq!(eval_counts.len(), 3);
        assert_eq!(eval_counts.get(&("mmlu".to_string(), 10)), Some(&2));
        assert_eq!(eval_counts.get(&("mmlu".to_string(), 20)), Some(&1));
        assert_eq!(eval_counts.get(&("gsm8k".to_string(), 30)), Some(&1));
    }

    #[test]
    fn test_sort_eval_examples_by_count() {
        let results = vec![
            create_test_result("mmlu", 10, "file1.txt"),
            create_test_result("mmlu", 10, "file2.txt"),
            create_test_result("mmlu", 10, "file3.txt"),
            create_test_result("mmlu", 20, "file4.txt"),
            create_test_result("gsm8k", 30, "file5.txt"),
            create_test_result("gsm8k", 30, "file6.txt"),
        ];

        let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();
        for result in &results {
            let key = (result.eval_dataset.clone(), result.eval_line);
            *eval_counts.entry(key).or_insert(0) += 1;
        }

        let mut sorted_counts: Vec<((String, usize), usize)> = eval_counts.into_iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

        assert_eq!(sorted_counts.len(), 3);
        assert_eq!(sorted_counts[0].0, ("mmlu".to_string(), 10));
        assert_eq!(sorted_counts[0].1, 3);
        assert_eq!(sorted_counts[1].0, ("gsm8k".to_string(), 30));
        assert_eq!(sorted_counts[1].1, 2);
        assert_eq!(sorted_counts[2].0, ("mmlu".to_string(), 20));
        assert_eq!(sorted_counts[2].1, 1);
    }

    #[test]
    fn test_empty_results() {
        let results: Vec<ContaminationResult> = vec![];

        let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();
        for result in &results {
            let key = (result.eval_dataset.clone(), result.eval_line);
            *eval_counts.entry(key).or_insert(0) += 1;
        }

        assert_eq!(eval_counts.len(), 0);
    }

    #[test]
    fn test_single_result() {
        let results = vec![
            create_test_result("mmlu", 10, "file1.txt"),
        ];

        let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();
        for result in &results {
            let key = (result.eval_dataset.clone(), result.eval_line);
            *eval_counts.entry(key).or_insert(0) += 1;
        }

        assert_eq!(eval_counts.len(), 1);
        assert_eq!(eval_counts.get(&("mmlu".to_string(), 10)), Some(&1));
    }

    #[test]
    fn test_all_unique_eval_examples() {
        let results = vec![
            create_test_result("mmlu", 10, "file1.txt"),
            create_test_result("mmlu", 20, "file2.txt"),
            create_test_result("gsm8k", 30, "file3.txt"),
            create_test_result("gsm8k", 40, "file4.txt"),
        ];

        let mut eval_counts: HashMap<(String, usize), usize> = HashMap::new();
        for result in &results {
            let key = (result.eval_dataset.clone(), result.eval_line);
            *eval_counts.entry(key).or_insert(0) += 1;
        }

        assert_eq!(eval_counts.len(), 4);
        for count in eval_counts.values() {
            assert_eq!(*count, 1);
        }
    }
}
