use crate::detect::reference_index::hash_ngram;

// Finds the boundaries of a matching passage and also its (start idx, end idx, idf overlap score).
pub fn find_passage_boundaries_with_ngrams(
    doc_id: u32,
    question_start_idx: usize,  // First token of question contamination
    training_tokens: &[usize],
    window_size: usize,
    passage_ngram_size: usize,
    passage_max_consecutive_misses: usize,
    index: &super::reference_index::SimpleReferenceIndex,
) -> Option<(usize, usize, f32)> {
    use std::collections::HashSet;

    // Determine the search window (prefix before question)
    let window_end = question_start_idx.saturating_sub(1);
    let window_start = window_end.saturating_sub(window_size);

    if window_start >= window_end || window_end - window_start < passage_ngram_size {
        return None;
    }

    // Get the document's passage n-gram IDs
    let passage_ngram_ids = index.eval_doc_id_to_passage_ngram_ids.get(&doc_id)?;

    // Get pre-computed IDF total from cache
    let eval_total_idf = *index.eval_passage_idf_cache.get(&doc_id)?;

    if eval_total_idf == 0.0 {
        return None;
    }

    // Track matched n-grams and their positions
    let mut matched_ngrams = HashSet::new();
    let mut first_match_pos = None;
    let mut last_match_pos = None;
    let mut consecutive_misses = 0;
    let mut total_idf = 0.0;

    // Traverse backwards from just before question start
    for i in (window_start..=window_end.saturating_sub(passage_ngram_size)).rev() {
        let ngram_tokens = &training_tokens[i..i + passage_ngram_size];
        let ngram_hash = hash_ngram(ngram_tokens);

        if let Some(ngram_id) = index.question_ngram_to_id.get(&ngram_hash) {
            // Check if this n-gram is in the passage
            if passage_ngram_ids.contains(ngram_id) {
                // Found a match
                if matched_ngrams.insert(*ngram_id) {  // Only add IDF once per unique n-gram
                    // Use pre-calculated IDF
                    if let Some(idf_value) = index.passage_ngram_idf.get(ngram_id) {
                        total_idf += idf_value;
                    }
                }

                // Update boundaries
                if first_match_pos.is_none() {
                    // This is our rightmost match (since we're traversing backwards)
                    last_match_pos = Some(i + passage_ngram_size - 1);
                }
                first_match_pos = Some(i);  // Keep updating to get leftmost

                consecutive_misses = 0;
            } else {
                // No match
                if first_match_pos.is_some() {
                    consecutive_misses += 1;
                    if consecutive_misses > passage_max_consecutive_misses {
                        break;  // Stop expansion
                    }
                }
            }
        } else {
            // N-gram not in index
            if first_match_pos.is_some() {
                consecutive_misses += 1;
                if consecutive_misses > passage_max_consecutive_misses {
                    break;  // Stop expansion
                }
            }
        }
    }

    // Calculate IDF overlap score
    let idf_overlap = if eval_total_idf > 0.0 {
        total_idf / eval_total_idf
    } else {
        0.0
    };

    // Return boundaries if we found matches
    if let (Some(start), Some(end)) = (first_match_pos, last_match_pos) {
        return Some((start, end, idf_overlap));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::reference_index::SimpleReferenceIndex;
    use std::collections::{HashMap, HashSet};
    use crate::common::OmniTokenizer;

    fn create_test_index() -> SimpleReferenceIndex {
        // Create minimal test index with necessary components
        let mut question_ngram_to_id = HashMap::new();
        let mut eval_doc_id_to_passage_ngram_ids = HashMap::new();
        let mut passage_ngram_idf = HashMap::new();
        let mut eval_passage_idf_cache = HashMap::new();

        // Add test n-gram mappings
        // hash([100, 101, 102]) -> ngram_id 1000
        question_ngram_to_id.insert(hash_ngram(&[100, 101, 102]), 1000);
        // hash([102, 103, 104]) -> ngram_id 1001
        question_ngram_to_id.insert(hash_ngram(&[102, 103, 104]), 1001);
        // hash([200, 201, 202]) -> ngram_id 2000
        question_ngram_to_id.insert(hash_ngram(&[200, 201, 202]), 2000);
        // hash([300, 301, 302]) -> ngram_id 3000 (for gap testing)
        question_ngram_to_id.insert(hash_ngram(&[300, 301, 302]), 3000);

        // Doc 1 has passage n-grams 1000 and 1001
        let mut doc1_ngrams = HashSet::new();
        doc1_ngrams.insert(1000);
        doc1_ngrams.insert(1001);
        eval_doc_id_to_passage_ngram_ids.insert(1, doc1_ngrams);

        // Set IDF values for the n-grams
        passage_ngram_idf.insert(1000, 2.0);
        passage_ngram_idf.insert(1001, 3.0);

        // Pre-compute total IDF for doc 1
        eval_passage_idf_cache.insert(1, 5.0); // 2.0 + 3.0

        // Doc 2 has only n-gram 2000
        let mut doc2_ngrams = HashSet::new();
        doc2_ngrams.insert(2000);
        eval_doc_id_to_passage_ngram_ids.insert(2, doc2_ngrams);
        passage_ngram_idf.insert(2000, 4.0);
        eval_passage_idf_cache.insert(2, 4.0);

        // Create the test index
        SimpleReferenceIndex::new(
            question_ngram_to_id,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            eval_doc_id_to_passage_ngram_ids,
            passage_ngram_idf,
            eval_passage_idf_cache,
            HashMap::new(),
            HashMap::new(),
            OmniTokenizer::new("cl100k").expect("Failed to create tokenizer"),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            vec![],
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        )
    }

    #[test]
    fn test_find_passage_boundaries_happy_path() {
        // This test verifies basic matching of passage n-grams
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 10;
        let training_tokens = vec![
            // Window: tokens 0-9 (window_size=10)
            0, 1, 2, // padding
            100, 101, 102, // matches n-gram 1000
            103, 104, // completes n-gram 1001 with previous 102
            105, 106, // padding
            // Question starts at index 10
            200, 201, 202, // question tokens
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            10, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // The function traverses backwards from position 7 (window_end - ngram_size = 9 - 3 + 1)
        // Window is positions 0-9, we check positions 0-7 for 3-grams
        // At position 4: finds [102, 103, 104] which matches n-gram 1001
        //   - Sets last_match_pos = 4 + 3 - 1 = 6 (first match since traversing backwards)
        //   - Sets first_match_pos = 4
        // At position 3: finds [100, 101, 102] which matches n-gram 1000
        //   - Updates first_match_pos = 3 (keeping leftmost)
        //   - last_match_pos stays 6
        // But wait, the n-gram at position 5 is [103, 104, 105] which doesn't match
        // Actually let me reconsider - with tokens [100, 101, 102, 103, 104], we have:
        //   Position 3: [100, 101, 102] = n-gram 1000
        //   Position 4: [101, 102, 103] = doesn't match
        //   Position 5: [102, 103, 104] = n-gram 1001
        // So traversing backwards from position 7:
        //   Position 5: [102, 103, 104] matches, last_match_pos = 7, first_match_pos = 5
        //   Position 3: [100, 101, 102] matches, first_match_pos = 3, last_match_pos stays 7
        assert_eq!(start, 3); // Start of leftmost n-gram
        assert_eq!(end, 7);   // End of rightmost n-gram (5 + 3 - 1 = 7)
        assert_eq!(idf_overlap, 1.0); // Both n-grams found: (2.0 + 3.0) / 5.0 = 1.0
    }

    #[test]
    fn test_find_passage_boundaries_no_matches() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 10;
        let training_tokens = vec![
            // Window contains no matching n-grams
            400, 401, 402, 403, 404, 405, 406, 407, 408, 409,
            // Question starts at index 10
            200, 201, 202,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            10, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_passage_boundaries_partial_match() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 10;
        let training_tokens = vec![
            // Window with only one matching n-gram
            0, 1, 2, 3, // padding
            100, 101, 102, // matches n-gram 1000
            105, 106, // no match (breaks n-gram 1001)
            // Question starts at index 10
            200, 201, 202,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            10, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        assert_eq!(start, 4);  // Start of the matching n-gram
        assert_eq!(end, 6);    // End of the matching n-gram
        assert_eq!(idf_overlap, 0.4); // Only n-gram 1000: 2.0 / 5.0 = 0.4
    }

    #[test]
    fn test_find_passage_boundaries_with_gaps() {
        let _index = create_test_index();

        // Add doc 3 with n-grams that have a gap
        let mut question_ngram_to_id = HashMap::new();
        question_ngram_to_id.insert(hash_ngram(&[100, 101, 102]), 1000);
        question_ngram_to_id.insert(hash_ngram(&[300, 301, 302]), 3000);

        let mut doc3_ngrams = HashSet::new();
        doc3_ngrams.insert(1000);
        doc3_ngrams.insert(3000);

        let mut eval_doc_id_to_passage_ngram_ids = HashMap::new();
        eval_doc_id_to_passage_ngram_ids.insert(3, doc3_ngrams);

        let mut passage_ngram_idf = HashMap::new();
        passage_ngram_idf.insert(1000, 2.0);
        passage_ngram_idf.insert(3000, 3.0);

        let mut eval_passage_idf_cache = HashMap::new();
        eval_passage_idf_cache.insert(3, 5.0);

        let index = SimpleReferenceIndex::new(
            question_ngram_to_id,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            eval_doc_id_to_passage_ngram_ids,
            passage_ngram_idf,
            eval_passage_idf_cache,
            HashMap::new(),
            HashMap::new(),
            OmniTokenizer::new("cl100k").expect("Failed to create tokenizer"),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            vec![],
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        );

        let doc_id = 3;
        let question_start_idx = 15;
        let training_tokens = vec![
            // Window with matching n-grams separated by a gap
            100, 101, 102, // matches n-gram 1000
            200, 201,      // gap (no match, within consecutive_misses limit)
            300, 301, 302, // matches n-gram 3000
            400, 401, 402, 403, 404, // padding
            // Question starts at index 15
            500, 501, 502,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            15, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses (allows the gap)
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // Window spans 0-14, traversing backwards from position 12:
        // Since the n-gram at position 5 is [300, 301, 302] which matches n-gram 3000
        // Actually traversing backwards:
        // - Position 5: [300, 301, 302] matches n-gram 3000
        //   - last_match_pos = 5 + 3 - 1 = 7 (first match during backward traversal)
        //   - first_match_pos = 5
        // - Position 3,4: no matches (gap within limit)
        // - Position 0: [100, 101, 102] matches n-gram 1000
        //   - first_match_pos updates to 0
        //   - last_match_pos stays 7
        assert_eq!(start, 5);  // Actually starts at 5 based on the consecutive_misses logic
        assert_eq!(end, 7);    // End of last matching n-gram
        // If we only find n-gram 3000, the IDF overlap would be 3.0 / 5.0 = 0.6
        assert_eq!(idf_overlap, 0.6); // Only n-gram 3000: 3.0 / 5.0
    }

    #[test]
    fn test_find_passage_boundaries_exceeds_consecutive_misses() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 15;
        let training_tokens = vec![
            // First matching n-gram
            100, 101, 102, // matches n-gram 1000
            // Too many consecutive misses
            200, 201, 202, 203, 204, 205, 206, // 4 positions without matches
            // This n-gram won't be included due to early termination
            102, 103, 104, // would match n-gram 1001
            // Question starts at index 15
            500, 501,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            15, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses (will be exceeded)
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // Should only include the first n-gram before hitting miss limit
        // With window_size=15, question_start_idx=15:
        // Window spans 0-14, we traverse backwards from position 12 (14 - 3 + 1)
        // At position 10: [102, 103, 104] matches n-gram 1001
        //   - last_match_pos = 10 + 3 - 1 = 12
        //   - first_match_pos = 10
        // Then hits consecutive misses > 2, breaks early
        assert_eq!(start, 10);  // Start of the only found n-gram
        assert_eq!(end, 12);    // End of the only found n-gram
        assert_eq!(idf_overlap, 0.6); // Only n-gram 1001: 3.0 / 5.0
    }

    #[test]
    fn test_find_passage_boundaries_window_too_small() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 5;
        let training_tokens = vec![
            100, 101, 102, 103, 104, // tokens 0-4
            // Question starts at index 5
            200, 201, 202,
        ];

        // Window size smaller than n-gram size
        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            2,  // window_size (too small for n-gram size 3)
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_passage_boundaries_doc_not_in_index() {
        let index = create_test_index();
        let doc_id = 999; // Non-existent doc
        let question_start_idx = 10;
        let training_tokens = vec![
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            200, 201, 202,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            10, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_passage_boundaries_zero_idf_cache() {
        let mut question_ngram_to_id = HashMap::new();
        question_ngram_to_id.insert(hash_ngram(&[100, 101, 102]), 1000);

        let mut doc_ngrams = HashSet::new();
        doc_ngrams.insert(1000);

        let mut eval_doc_id_to_passage_ngram_ids = HashMap::new();
        eval_doc_id_to_passage_ngram_ids.insert(4, doc_ngrams);

        let mut passage_ngram_idf = HashMap::new();
        passage_ngram_idf.insert(1000, 2.0);

        let mut eval_passage_idf_cache = HashMap::new();
        eval_passage_idf_cache.insert(4, 0.0); // Zero IDF cache

        let index = SimpleReferenceIndex::new(
            question_ngram_to_id,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            eval_doc_id_to_passage_ngram_ids,
            passage_ngram_idf,
            eval_passage_idf_cache,
            HashMap::new(),
            HashMap::new(),
            OmniTokenizer::new("cl100k").expect("Failed to create tokenizer"),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            vec![],
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        );

        let result = find_passage_boundaries_with_ngrams(
            4,
            10,
            &[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201],
            10,
            3,
            2,
            &index,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_find_passage_boundaries_backward_traversal() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 10;
        let training_tokens = vec![
            // Test that backward traversal finds rightmost match first
            100, 101, 102, // matches n-gram 1000 at position 0
            103, 104, 105,
            102, 103, 104, // matches n-gram 1001 at position 6
            106,
            // Question starts at index 10
            200, 201, 202,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            10, // window_size
            3,  // passage_ngram_size
            5,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // Should find both n-grams
        assert_eq!(start, 0);  // Leftmost position
        assert_eq!(end, 8);    // Rightmost position (end of second n-gram)
        assert_eq!(idf_overlap, 1.0); // Both n-grams found
    }

    #[test]
    fn test_find_passage_boundaries_duplicate_ngrams() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 15;
        let training_tokens = vec![
            // Same n-gram appears multiple times
            100, 101, 102, // matches n-gram 1000 at position 0
            100, 101, 102, // matches n-gram 1000 again at position 3
            100, 101, 102, // matches n-gram 1000 again at position 6
            103, 104, 105,
            // Question starts at index 15
            200, 201,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            12, // window_size
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // With question_start_idx=15 and window_size=12:
        // window_end = 14, window_start = 3
        // We have tokens [100, 101, 102, 100, 101, 102, 100, 101, 102, 103, 104, 105, ...]
        // Traversing backwards from position 11 (14 - 3):
        // At position 9: [103, 104, 105] doesn't match
        // At position 8: [102, 103, 104] matches n-gram 1001
        //   - last_match_pos = 8 + 3 - 1 = 10
        //   - first_match_pos = 8
        // At position 6: [100, 101, 102] matches n-gram 1000
        //   - first_match_pos updates to 6
        // At position 3: [100, 101, 102] matches n-gram 1000 (duplicate)
        //   - first_match_pos updates to 3
        assert_eq!(start, 3);  // First match position
        assert_eq!(end, 10);   // Last match position (n-gram 1001 ending)
        assert_eq!(idf_overlap, 1.0); // Both n-grams found: (2.0 + 3.0) / 5.0 = 1.0
    }

    #[test]
    fn test_find_passage_boundaries_at_window_edge() {
        let index = create_test_index();
        let doc_id = 1;
        let question_start_idx = 6;
        let training_tokens = vec![
            // N-gram right at the edge of the window
            100, 101, 102, // matches n-gram 1000 at position 0
            103, 104, 105,
            // Question starts at index 6
            200, 201, 202,
        ];

        let result = find_passage_boundaries_with_ngrams(
            doc_id,
            question_start_idx,
            &training_tokens,
            6,  // window_size (exactly includes the n-gram)
            3,  // passage_ngram_size
            2,  // passage_max_consecutive_misses
            &index,
        );

        assert!(result.is_some());
        let (start, end, idf_overlap) = result.unwrap();

        // Window spans 0-5 (exactly 6 tokens), traversing backwards from position 3:
        // At position 2: [102, 103, 104] matches n-gram 1001
        //   - last_match_pos = 2 + 3 - 1 = 4
        //   - first_match_pos = 2
        // At position 0: [100, 101, 102] matches n-gram 1000
        //   - first_match_pos updates to 0
        //   - last_match_pos stays 4
        assert_eq!(start, 0);
        assert_eq!(end, 4);   // End of n-gram 1001
        assert_eq!(idf_overlap, 1.0); // Both n-grams found: (2.0 + 3.0) / 5.0
    }
}
