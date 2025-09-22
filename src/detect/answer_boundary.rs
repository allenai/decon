use std::collections::HashSet;

use crate::detect::reference_index::{hash_ngram, AnswerTokenIdfMap};
use crate::detect::scoring::calculate_answer_cluster_idf_overlap;

// Find answer boundaries using n-gram index for long answers.
// Returns multiple boundary ranges where answer n-grams match
// Allows for disjoint smaller clusters, which is used to highlight matching portions
// in the review tool. IDF overlap is based on unique n-grams from all clusters.
pub fn find_answer_boundaries_with_ngrams(
    doc_id: u32,
    question_end_idx: usize,
    training_tokens: &[usize],
    window_size: usize,
    ngram_size: usize,
    index: &super::reference_index::SimpleReferenceIndex,
) -> Option<(Vec<(usize, usize)>, f32)> {
    use std::collections::HashSet;

    // Determine the search window
    let window_start = question_end_idx + 1;
    let window_end = (window_start + window_size).min(training_tokens.len());

    if window_end <= window_start || window_end - window_start < ngram_size {
        return None;
    }

    // Get the document's answer n-gram IDs
    let answer_ngram_ids = index.eval_doc_id_to_answer_ngram_ids.get(&doc_id)?;

    // Calculate eval answer IDF total on-the-fly
    // Note: We use total eval documents (not just docs with answers) for IDF calculation.
    // Since we use IDF ratios for scoring, this constant scaling factor cancels out
    // and doesn't affect contamination detection.
    let mut eval_total_idf = 0.0f32;

    // Sort ngram_ids for deterministic iteration order
    let mut sorted_ids: Vec<u32> = answer_ngram_ids.iter().copied().collect();
    sorted_ids.sort_unstable();

    for ngram_id in sorted_ids {
        if let Some(idf_value) = index.answer_ngram_idf.get(&ngram_id) {
            eval_total_idf += idf_value;
        }
    }

    if eval_total_idf == 0.0 {
        return None;
    }

    // Step 1: Batch collect all n-gram hashes in the window
    let mut ngram_hashes = Vec::with_capacity(window_end - window_start);
    for i in window_start..=window_end.saturating_sub(ngram_size) {
        let ngram_tokens = &training_tokens[i..i + ngram_size];
        ngram_hashes.push((i, hash_ngram(ngram_tokens)));
    }

    // Step 2: Process and collect matching n-grams WITH IDF VALUES
    let mut matched_ngrams = HashSet::new();
    let mut matched_positions = Vec::new();
    let mut total_idf = 0.0;

    for (pos, ngram_hash) in ngram_hashes {
        if let Some(ngram_id) = index.question_ngram_to_id.get(&ngram_hash) {
            // Check if this n-gram is in the answer
            if answer_ngram_ids.contains(ngram_id) {
                if matched_ngrams.insert(*ngram_id) {  // Only add IDF once per unique n-gram
                    if let Some(idf_value) = index.answer_ngram_idf.get(ngram_id) {
                        total_idf += idf_value;
                    }
                }
                matched_positions.push((pos, pos + ngram_size - 1));
            }
        }
    }

    // Calculate IDF overlap score
    let idf_overlap = if eval_total_idf > 0.0 {
        total_idf / eval_total_idf
    } else {
        0.0
    };

    // Cluster the matched positions into separate ranges
    if !matched_positions.is_empty() {
        // Sort by start position
        matched_positions.sort_by_key(|(start, _)| *start);

        let mut clusters: Vec<(usize, usize)> = Vec::new();
        let mut current_cluster_start = matched_positions[0].0;
        let mut current_cluster_end = matched_positions[0].1;

        // Gap threshold: if next match starts more than ngram_size tokens after current cluster ends,
        // start a new cluster
        let gap_threshold = ngram_size;

        for &(start, end) in &matched_positions[1..] {
            // Check if this position extends or overlaps with current cluster
            if start <= current_cluster_end + gap_threshold {
                // Extend current cluster
                current_cluster_end = current_cluster_end.max(end);
            } else {
                // Save current cluster and start new one
                clusters.push((current_cluster_start, current_cluster_end));
                current_cluster_start = start;
                current_cluster_end = end;
            }
        }

        // Don't forget the last cluster
        clusters.push((current_cluster_start, current_cluster_end));

        return Some((clusters, idf_overlap));
    }

    None
}

/// Find exact sequence match for short answers (≤3 tokens default config)
/// Returns a single-element vector for consistency with long answer matching
pub fn find_short_answer_exact_match(
    answer_tokens: &[usize],
    answer_token_set: &HashSet<usize>,
    question_end_idx: usize,
    training_tokens: &[usize],
    window_size: usize,
    answer_token_idf: &AnswerTokenIdfMap,
) -> Option<(Vec<(usize, usize)>, f32)> {
    let search_start = question_end_idx + 1;
    let search_end = (search_start + window_size).min(training_tokens.len());

    // Scan through the window looking for exact sequence match
    for start_idx in search_start..search_end {
        // Check if we have enough tokens left for a match
        if start_idx + answer_tokens.len() > training_tokens.len() {
            break;
        }

        // Check for exact sequence match
        let mut matches = true;
        for (i, &answer_token) in answer_tokens.iter().enumerate() {
            if training_tokens[start_idx + i] != answer_token {
                matches = false;
                break;
            }
        }

        if matches {
            // Found exact match
            let end_idx = start_idx + answer_tokens.len() - 1;

            // Calculate token-level IDF overlap for the matched answer
            let matched_tokens: HashSet<usize> = answer_tokens.iter().copied().collect();
            let idf_overlap = calculate_answer_cluster_idf_overlap(
                &matched_tokens,
                answer_token_set,
                answer_token_idf,
            );

            // Return as single-element vector for consistency
            return Some((vec![(start_idx, end_idx)], idf_overlap));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use crate::common::OmniTokenizer;
    use crate::detect::SimpleReferenceIndex;

    // Helper function to create a test reference index
    fn create_test_index() -> SimpleReferenceIndex {
        let mut question_ngram_to_id = HashMap::new();
        let mut eval_doc_id_to_answer_ngram_ids = HashMap::new();
        let mut answer_ngram_idf = HashMap::new();
        let mut eval_doc_id_to_answer_tokens = HashMap::new();
        let mut eval_doc_id_to_answer_tokens_ordered = HashMap::new();
        let mut answer_token_idf = HashMap::new();

        // Setup test data for doc_id 1
        // Add some n-gram mappings
        // hash(100, 101, 102) -> ngram_id 1000
        question_ngram_to_id.insert(hash_ngram(&[100, 101, 102]), 1000);
        // hash(102, 103, 104) -> ngram_id 1001
        question_ngram_to_id.insert(hash_ngram(&[102, 103, 104]), 1001);
        // hash(200, 201, 202) -> ngram_id 2000
        question_ngram_to_id.insert(hash_ngram(&[200, 201, 202]), 2000);
        // hash(300, 301, 302) -> ngram_id 3000 (for gap testing)
        question_ngram_to_id.insert(hash_ngram(&[300, 301, 302]), 3000);

        // Doc 1 has answer n-grams 1000, 1001, and 3000
        let mut doc1_ngrams = HashSet::new();
        doc1_ngrams.insert(1000);
        doc1_ngrams.insert(1001);
        doc1_ngrams.insert(3000);
        eval_doc_id_to_answer_ngram_ids.insert(1, doc1_ngrams);

        // Set IDF values for answer n-grams
        answer_ngram_idf.insert(1000, 2.5);
        answer_ngram_idf.insert(1001, 3.0);
        answer_ngram_idf.insert(2000, 1.5);
        answer_ngram_idf.insert(3000, 4.0);

        // Setup answer tokens for doc 1 (for short answer matching)
        let mut doc1_answer_tokens = HashSet::new();
        doc1_answer_tokens.insert(500);
        doc1_answer_tokens.insert(501);
        doc1_answer_tokens.insert(502);
        eval_doc_id_to_answer_tokens.insert(1, doc1_answer_tokens);

        // Ordered answer tokens for exact match
        eval_doc_id_to_answer_tokens_ordered.insert(1, vec![500, 501, 502]);

        // Answer token IDF values
        answer_token_idf.insert(500, 1.0);
        answer_token_idf.insert(501, 1.5);
        answer_token_idf.insert(502, 2.0);

        // Create other required empty structures
        let question_ngram_id_to_eval_doc_ids = HashMap::new();
        let eval_doc_id_to_passage_tokens = HashMap::new();
        let eval_doc_id_to_passage_ngram_ids = HashMap::new();
        let passage_ngram_idf = HashMap::new();
        let eval_passage_idf_cache = HashMap::new();
        let eval_documents = HashMap::new();
        let eval_document_metadata = HashMap::new();
        let eval_text_snippets = HashMap::new();
        let eval_answer_text_snippets = HashMap::new();
        let eval_passage_text_snippets = HashMap::new();
        let eval_doc_id_to_question_ngram_ids = HashMap::new();
        let reference_filenames = vec![];

        let tokenizer = OmniTokenizer::new("word").expect("Failed to create tokenizer");

        SimpleReferenceIndex::new(
            question_ngram_to_id,
            question_ngram_id_to_eval_doc_ids,
            eval_doc_id_to_answer_tokens,
            eval_doc_id_to_answer_tokens_ordered,
            eval_doc_id_to_answer_ngram_ids,
            answer_ngram_idf,
            eval_doc_id_to_passage_tokens,
            eval_doc_id_to_passage_ngram_ids,
            passage_ngram_idf,
            eval_passage_idf_cache,
            eval_documents,
            eval_document_metadata,
            tokenizer,
            eval_text_snippets,
            eval_answer_text_snippets,
            eval_passage_text_snippets,
            eval_doc_id_to_question_ngram_ids,
            answer_token_idf,
            reference_filenames,
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        )
    }

    #[test]
    fn test_find_answer_boundaries_happy_path() {
        let index = create_test_index();
        let doc_id = 1;
        let question_end_idx = 5;
        // Training tokens containing matching n-grams
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens (0-5)
            100, 101, 102, 103, 104, // answer tokens with matching n-grams (6-10)
            200, 201, 202, // non-matching tokens (11-13)
        ];
        let window_size = 10;
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        assert!(result.is_some());
        let (clusters, idf_overlap) = result.unwrap();

        // Should find one cluster containing the overlapping n-grams
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], (6, 10)); // Positions 6-10 contain matching n-grams

        // IDF overlap: (2.5 + 3.0) / (2.5 + 3.0 + 4.0) = 5.5 / 9.5 ≈ 0.579
        assert!((idf_overlap - 0.579).abs() < 0.01);
    }

    #[test]
    fn test_find_answer_boundaries_multiple_clusters() {
        let index = create_test_index();
        let doc_id = 1;
        let question_end_idx = 5;
        // Training tokens with two separate matching regions
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens (0-5)
            100, 101, 102, 103, 104, // first matching region (6-10)
            200, 201, 202, 203, 204, 205, // gap (11-16)
            300, 301, 302, // second matching region (17-19)
        ];
        let window_size = 20;
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        assert!(result.is_some());
        let (clusters, idf_overlap) = result.unwrap();

        // Should find two separate clusters
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0], (6, 10)); // First cluster
        assert_eq!(clusters[1], (17, 19)); // Second cluster

        // All three n-grams matched: (2.5 + 3.0 + 4.0) / (2.5 + 3.0 + 4.0) = 1.0
        assert_eq!(idf_overlap, 1.0);
    }

    #[test]
    fn test_find_answer_boundaries_window_too_small() {
        let index = create_test_index();
        let doc_id = 1;
        let question_end_idx = 5;
        let training_tokens = vec![0, 1, 2, 3, 4, 5, 100];
        let window_size = 1; // Too small for ngram_size of 3
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_answer_boundaries_no_matching_ngrams() {
        let index = create_test_index();
        let doc_id = 1;
        let question_end_idx = 5;
        // Training tokens with no matching n-grams
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens
            400, 401, 402, 403, 404, // non-matching tokens
        ];
        let window_size = 10;
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_answer_boundaries_missing_doc_id() {
        let index = create_test_index();
        let doc_id = 999; // Non-existent doc ID
        let question_end_idx = 5;
        let training_tokens = vec![0, 1, 2, 3, 4, 5, 100, 101, 102];
        let window_size = 10;
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_answer_boundaries_zero_idf() {
        let mut index = create_test_index();

        // Modify index to have zero IDF values
        let mut answer_ngram_idf = HashMap::new();
        answer_ngram_idf.insert(1000, 0.0);
        answer_ngram_idf.insert(1001, 0.0);

        // Create a new index with zero IDF values
        let question_ngram_to_id = Arc::try_unwrap(index.question_ngram_to_id).unwrap();
        let eval_doc_id_to_answer_ngram_ids = Arc::try_unwrap(index.eval_doc_id_to_answer_ngram_ids).unwrap();

        index.answer_ngram_idf = Arc::new(answer_ngram_idf);
        index.question_ngram_to_id = Arc::new(question_ngram_to_id);
        index.eval_doc_id_to_answer_ngram_ids = Arc::new(eval_doc_id_to_answer_ngram_ids);

        let doc_id = 1;
        let question_end_idx = 5;
        let training_tokens = vec![0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104];
        let window_size = 10;
        let ngram_size = 3;

        let result = find_answer_boundaries_with_ngrams(
            doc_id,
            question_end_idx,
            &training_tokens,
            window_size,
            ngram_size,
            &index,
        );

        // Should return None when total IDF is zero
        assert!(result.is_none());
    }

    #[test]
    fn test_find_short_answer_exact_match_happy_path() {
        let index = create_test_index();
        let answer_tokens = vec![500, 501, 502];
        let mut answer_token_set = HashSet::new();
        answer_token_set.insert(500);
        answer_token_set.insert(501);
        answer_token_set.insert(502);

        let question_end_idx = 5;
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens
            500, 501, 502, // exact answer match (positions 6-8)
            600, 601,
        ];
        let window_size = 10;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        assert!(result.is_some());
        let (clusters, idf_overlap) = result.unwrap();

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], (6, 8));

        // IDF overlap: (1.0 + 1.5 + 2.0) / (1.0 + 1.5 + 2.0) = 1.0
        assert_eq!(idf_overlap, 1.0);
    }

    #[test]
    fn test_find_short_answer_at_window_start() {
        let index = create_test_index();
        let answer_tokens = vec![500, 501, 502];
        let mut answer_token_set = HashSet::new();
        answer_token_set.insert(500);
        answer_token_set.insert(501);
        answer_token_set.insert(502);

        let question_end_idx = 2;
        let training_tokens = vec![
            0, 1, 2, // question tokens
            500, 501, 502, // exact answer match at start of window
            600, 601,
        ];
        let window_size = 5;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        assert!(result.is_some());
        let (clusters, _) = result.unwrap();
        assert_eq!(clusters[0], (3, 5));
    }

    #[test]
    fn test_find_short_answer_at_window_end() {
        let index = create_test_index();
        let answer_tokens = vec![500, 501, 502];
        let mut answer_token_set = HashSet::new();
        answer_token_set.insert(500);
        answer_token_set.insert(501);
        answer_token_set.insert(502);

        let question_end_idx = 2;
        let training_tokens = vec![
            0, 1, 2, // question tokens
            600, 601, 602, // filler
            500, 501, 502, // exact answer match at end
        ];
        let window_size = 6;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        assert!(result.is_some());
        let (clusters, _) = result.unwrap();
        assert_eq!(clusters[0], (6, 8));
    }

    #[test]
    fn test_find_short_answer_no_match() {
        let index = create_test_index();
        let answer_tokens = vec![500, 501, 502];
        let mut answer_token_set = HashSet::new();
        answer_token_set.insert(500);
        answer_token_set.insert(501);
        answer_token_set.insert(502);

        let question_end_idx = 5;
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens
            500, 501, 600, // partial match, not exact
            700, 701,
        ];
        let window_size = 10;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_short_answer_outside_window() {
        let index = create_test_index();
        let answer_tokens = vec![500, 501, 502];
        let mut answer_token_set = HashSet::new();
        answer_token_set.insert(500);
        answer_token_set.insert(501);
        answer_token_set.insert(502);

        let question_end_idx = 5;
        let training_tokens = vec![
            0, 1, 2, 3, 4, 5, // question tokens
            600, 601, // window ends here (window_size = 2)
            500, 501, 502, // answer is outside window
        ];
        let window_size = 2;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_short_answer_empty_answer() {
        let index = create_test_index();
        let answer_tokens = vec![];
        let answer_token_set = HashSet::new();

        let question_end_idx = 5;
        let training_tokens = vec![0, 1, 2, 3, 4, 5, 600, 601];
        let window_size = 10;

        let result = find_short_answer_exact_match(
            &answer_tokens,
            &answer_token_set,
            question_end_idx,
            &training_tokens,
            window_size,
            &index.answer_token_idf,
        );

        // Empty answer should match at the start of the search window
        assert!(result.is_some());
        let (clusters, idf_overlap) = result.unwrap();
        assert_eq!(clusters[0], (6, 5)); // start > end for empty match
        assert_eq!(idf_overlap, 0.0); // Empty sets result in 0.0 IDF overlap
    }
}
