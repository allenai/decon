use std::collections::{HashMap, HashSet};
use std::time::Instant;
use crate::detect::reference_index::{hash_ngram, QuestionNgramToIdMap, QuestionNgramIdToEvalDocIdsMap};
use crate::detect::stats::StatsContainer;

/// Groups mutable state used during cluster traversal
struct TraversalState<'a> {
    active_documents: &'a mut HashSet<u32>,
    document_matches: &'a mut HashMap<u32, HashSet<u32>>,
    document_misses: &'a mut HashMap<u32, usize>,
    document_boundaries: &'a mut HashMap<u32, (usize, usize)>,
    ngram_doc_frequencies: &'a mut HashMap<u32, usize>,
}

/// Context struct to pass required dependencies for expansion
pub struct ExpansionContext<'a> {
    pub question_ngram_to_id: &'a QuestionNgramToIdMap,
    pub question_ngram_id_to_eval_doc_ids: &'a QuestionNgramIdToEvalDocIdsMap,
    pub ngram_size: usize,
    pub question_max_consecutive_misses: u32,
    pub hot_bucket_id: u32,
    pub question_ngram_id_to_hot_bucket_doc_ids: &'a HashMap<u32, HashSet<u32>>,
}

#[derive(Clone)]
pub(crate) struct SimpleContaminationCluster {
    pub document_matches: HashMap<u32, HashSet<u32>>, // doc_id -> unique_ngram_ids that matched eval
    pub document_boundaries: HashMap<u32, (usize, usize)>, // doc_id -> (start_idx, end_idx)
    pub ngram_doc_frequencies: HashMap<u32, usize>, // ngram_id -> document frequency (cached)
    pub end_idx: usize,
}

/// A contamination cluster is all of the information that stems from a single sampled hit,
/// in other words, when an n-gram is first matched against a training document. From that
/// point we get a set of documents that contain that n-gram, and for each of those documents
/// we record a left and right boundary, set of token ids that matched and their frequencies
/// for idf calculation.
impl SimpleContaminationCluster {
    pub fn new(
        document_matches: HashMap<u32, HashSet<u32>>,
        document_boundaries: HashMap<u32, (usize, usize)>,
        ngram_doc_frequencies: HashMap<u32, usize>,
        end_idx: usize,
    ) -> Self {
        Self {
            document_matches,
            document_boundaries,
            ngram_doc_frequencies,
            end_idx,
        }
    }
}


/// Check a single n-gram for matches, return document IDs that match and the ngram_id.
/// If this is a hot bucket, indicated by the so called sentinal id (max doc id + 1 at index build time),
/// we check the hot bucket map to retrieve the large list of documents. This might seem odd,
/// but it's all about keeping the hot path of "is this n-gram in any eval" as fast as possible, and
/// in the case of hot ngrams, we don't want to start clusters there because the chances of it being
/// a cluster of tokens that match an eval is low, it's just a common token sequence. Note that we
/// shift our token sampling rate to 1, until a not-hot-ngram is detected.
fn check_ngram_for_match<'a>(
    ngram_hash: u64,
    context: &'a ExpansionContext,
) -> Option<(u32, &'a HashSet<u32>)> {
    if let Some(ngram_id) = context.question_ngram_to_id.get(&ngram_hash)
        && let Some(doc_set) = context.question_ngram_id_to_eval_doc_ids.get(ngram_id)
            && !doc_set.is_empty() {
                if doc_set.len() == 1 && doc_set.contains(&context.hot_bucket_id) {
                    // Retrieve the actual doc set from the hot bucket map
                    if let Some(actual_docs) = context.question_ngram_id_to_hot_bucket_doc_ids.get(ngram_id) {
                        return Some((*ngram_id, actual_docs));
                    }
                }
                return Some((*ngram_id, doc_set));
            }
    None
}

/// Expand contamination cluster using intersection-based left/right traversal
/// Note that we don't care if we go over a little, because we are solving a containment
/// problem and using overlap scores. The real problem is when a cluster is shorter than
/// the eval document.
pub(crate) fn expand_simple_contamination_cluster(
    hit_idx: usize,
    word_tokens: &[usize],
    context: &ExpansionContext,
    initial_document_ids: HashSet<u32>,
    initial_training_ngram: &[usize],
    stats: Option<&StatsContainer>,
) -> Option<SimpleContaminationCluster> {
    let start_time = if stats.is_some() {
        Some(Instant::now())
    } else {
        None
    };

    // Return None if we don't have enough tokens to form a proper n-gram
    if word_tokens.len() < context.ngram_size {
        return None;
    }

    // Initialize document match tracking - track consecutive misses for each document
    let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut document_misses: HashMap<u32, usize> = HashMap::new();
    let mut document_boundaries: HashMap<u32, (usize, usize)> = HashMap::new();
    let mut ngram_doc_frequencies: HashMap<u32, usize> = HashMap::new();
    let mut active_documents: HashSet<u32> = initial_document_ids.clone();

    // Get the initial n-gram ID for tracking
    let initial_ngram_hash = hash_ngram(initial_training_ngram);
    let initial_ngram_id = context.question_ngram_to_id
        .get(&initial_ngram_hash).copied()
        .unwrap_or(0);

    // Store the initial ngram's document frequency, used later for idf calculation if cluster reaches scoring
    ngram_doc_frequencies.insert(initial_ngram_id, initial_document_ids.len());

    for doc_id in &initial_document_ids {
        let matched_ngrams = HashSet::from([initial_ngram_id]);
        document_matches.insert(*doc_id, matched_ngrams);
        document_misses.insert(*doc_id, 0);
        document_boundaries.insert(*doc_id, (hit_idx, hit_idx));
    }

    let total_ngrams = word_tokens.len() - context.ngram_size + 1;

    let mut left_state = TraversalState {
        active_documents: &mut active_documents,
        document_matches: &mut document_matches,
        document_misses: &mut document_misses,
        document_boundaries: &mut document_boundaries,
        ngram_doc_frequencies: &mut ngram_doc_frequencies,
    };
    let _left_idx = traverse_left(
        hit_idx,
        word_tokens,
        context,
        &mut left_state,
        stats,
    );

    // Reset active documents and misses for right traversal
    active_documents = initial_document_ids.clone();
    for doc_id in &initial_document_ids {
        document_misses.insert(*doc_id, 0);
    }

    let mut right_state = TraversalState {
        active_documents: &mut active_documents,
        document_matches: &mut document_matches,
        document_misses: &mut document_misses,
        document_boundaries: &mut document_boundaries,
        ngram_doc_frequencies: &mut ngram_doc_frequencies,
    };
    let right_idx = traverse_right(
        hit_idx,
        word_tokens,
        context,
        total_ngrams,
        &mut right_state,
        stats,
    );

    let result = SimpleContaminationCluster::new(
        document_matches,
        document_boundaries,
        ngram_doc_frequencies,
        right_idx,
    );

    // Track timing in stats if available
    if let (Some(stats), Some(start)) = (stats, start_time) {
        let elapsed = start.elapsed().as_micros() as u64;
        stats.add_question_expansion_time(elapsed);
    }

    Some(result)
}

fn traverse_left(
    hit_idx: usize,
    word_tokens: &[usize],
    context: &ExpansionContext,
    state: &mut TraversalState,
    stats: Option<&StatsContainer>,
) -> usize {
    let mut current_idx = hit_idx;

    loop {
        // Boundary check for left traversal
        if state.active_documents.is_empty() || current_idx == 0 {
            break;
        }

        current_idx -= 1;
        if let Some(stats) = stats {
            stats.increment_left_traversals();
        }

        let ngram_tokens = &word_tokens[current_idx..current_idx + context.ngram_size];
        let ngram_hash = hash_ngram(ngram_tokens);

        if let Some((ngram_id, matched_docs)) = check_ngram_for_match(ngram_hash, context) {
            // Cache the document frequency for this ngram
            state.ngram_doc_frequencies.entry(ngram_id).or_insert_with(|| matched_docs.len());

            let intersection: Vec<u32> = state.active_documents
                .intersection(matched_docs)
                .cloned()
                .collect();

            if !intersection.is_empty() {
                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    let is_new_ngram = state.document_matches
                        .entry(*doc_id)
                        .or_default()
                        .insert(ngram_id);

                    // Only reset miss counter if this is a new n-gram for this document
                    if is_new_ngram {
                        state.document_misses.insert(*doc_id, 0);
                    }

                    // Update left boundary
                    if let Some((doc_start, _doc_end)) = state.document_boundaries.get_mut(doc_id) {
                        *doc_start = current_idx;
                    }
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = state.active_documents
                    .difference(matched_docs)
                    .cloned()
                    .collect();
                for doc_id in to_remove {
                    let miss_count = state.document_misses.entry(doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= context.question_max_consecutive_misses as usize {
                        state.active_documents.remove(&doc_id);
                    }
                }

                continue; // Skip to next iteration - we found a match with intersection
            }
        }

        // Increment miss count for all active documents
        // (handles both "no match" and "match with no intersection" cases)
        let mut to_remove = Vec::new();
        for doc_id in state.active_documents.iter() {
            let miss_count = state.document_misses.entry(*doc_id).or_insert(0);
            *miss_count += 1;
            if *miss_count >= context.question_max_consecutive_misses as usize {
                to_remove.push(*doc_id);
            }
        }
        for doc_id in to_remove {
            state.active_documents.remove(&doc_id);
        }
    }

    current_idx
}

fn traverse_right(
    hit_idx: usize,
    word_tokens: &[usize],
    context: &ExpansionContext,
    total_ngrams: usize,
    state: &mut TraversalState,
    stats: Option<&StatsContainer>,
) -> usize {
    let mut current_idx = hit_idx;

    loop {
        // Document boundary check for right traversal
        if state.active_documents.is_empty() || current_idx + 1 >= total_ngrams {
            break;
        }

        current_idx += 1;
        if let Some(stats) = stats {
            stats.increment_right_traversals();
        }

        let ngram_tokens = &word_tokens[current_idx..current_idx + context.ngram_size];
        let ngram_hash = hash_ngram(ngram_tokens);

        if let Some((ngram_id, matched_docs)) = check_ngram_for_match(ngram_hash, context) {
            // Cache the document frequency for this ngram
            state.ngram_doc_frequencies.entry(ngram_id).or_insert_with(|| matched_docs.len());

            let intersection: Vec<u32> = state.active_documents
                .intersection(matched_docs)
                .cloned()
                .collect();

            if !intersection.is_empty() {
                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    let is_new_ngram = state.document_matches
                        .entry(*doc_id)
                        .or_default()
                        .insert(ngram_id);

                    // Only reset miss counter if this is a new n-gram for this document
                    if is_new_ngram {
                        state.document_misses.insert(*doc_id, 0);
                    }

                    // Update right boundary
                    if let Some((_doc_start, doc_end)) = state.document_boundaries.get_mut(doc_id) {
                        *doc_end = current_idx;
                    }
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = state.active_documents
                    .difference(matched_docs)
                    .cloned()
                    .collect();
                for doc_id in to_remove {
                    let miss_count = state.document_misses.entry(doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= context.question_max_consecutive_misses as usize {
                        state.active_documents.remove(&doc_id);
                    }
                }

                continue; // Skip to next iteration - we found a match with intersection
            }
        }

        // Increment miss count for all active documents
        // (handles both "no match" and "match with no intersection" cases)
        let mut to_remove = Vec::new();
        for doc_id in state.active_documents.iter() {
            let miss_count = state.document_misses.entry(*doc_id).or_insert(0);
            *miss_count += 1;
            if *miss_count >= context.question_max_consecutive_misses as usize {
                to_remove.push(*doc_id);
            }
        }
        for doc_id in to_remove {
            state.active_documents.remove(&doc_id);
        }
    }

    current_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    // Helper function to create test maps
    fn create_test_maps() -> (QuestionNgramToIdMap, QuestionNgramIdToEvalDocIdsMap) {
        let mut ngram_to_id = HashMap::new();
        let mut id_to_docs = HashMap::new();

        // Add some test n-grams
        ngram_to_id.insert(1000, 1); // hash 1000 -> id 1
        ngram_to_id.insert(2000, 2); // hash 2000 -> id 2
        ngram_to_id.insert(3000, 3); // hash 3000 -> id 3
        ngram_to_id.insert(4000, 4); // hash 4000 -> id 4

        // Map IDs to document sets
        let mut docs1 = HashSet::new();
        docs1.insert(10);
        docs1.insert(11);
        id_to_docs.insert(1, docs1);

        let mut docs2 = HashSet::new();
        docs2.insert(11);
        docs2.insert(12);
        id_to_docs.insert(2, docs2);

        let mut docs3 = HashSet::new();
        docs3.insert(10);
        docs3.insert(12);
        docs3.insert(13);
        id_to_docs.insert(3, docs3);

        // Empty document set for testing
        id_to_docs.insert(4, HashSet::new());

        (ngram_to_id, id_to_docs)
    }

    #[allow(clippy::type_complexity)]
    fn create_test_context(
        ngram_size: usize,
        question_max_consecutive_misses: u32,
    ) -> (ExpansionContext<'static>, Box<QuestionNgramToIdMap>, Box<QuestionNgramIdToEvalDocIdsMap>, Box<HashMap<u32, HashSet<u32>>>) {
        let (ngram_to_id, id_to_docs) = create_test_maps();
        let hot_bucket_map = HashMap::new();

        // Box the maps to ensure stable memory addresses
        let ngram_to_id_box = Box::new(ngram_to_id);
        let id_to_docs_box = Box::new(id_to_docs);
        let hot_bucket_box = Box::new(hot_bucket_map);

        // Create raw pointers that will be valid for 'static lifetime in tests
        let ngram_to_id_ptr = Box::leak(ngram_to_id_box.clone());
        let id_to_docs_ptr = Box::leak(id_to_docs_box.clone());
        let hot_bucket_ptr = Box::leak(hot_bucket_box.clone());

        let context = ExpansionContext {
            question_ngram_to_id: ngram_to_id_ptr,
            question_ngram_id_to_eval_doc_ids: id_to_docs_ptr,
            ngram_size,
            question_max_consecutive_misses,
            hot_bucket_id: 999999, // Use a high ID that won't conflict with test data
            question_ngram_id_to_hot_bucket_doc_ids: hot_bucket_ptr,
        };

        (context, ngram_to_id_box, id_to_docs_box, hot_bucket_box)
    }

    #[test]
    fn test_check_ngram_for_match_found() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);

        let result = check_ngram_for_match(1000, &context);
        assert!(result.is_some());

        let (ngram_id, doc_set) = result.unwrap();
        assert_eq!(ngram_id, 1);
        assert_eq!(doc_set.len(), 2);
        assert!(doc_set.contains(&10));
        assert!(doc_set.contains(&11));
    }

    #[test]
    fn test_check_ngram_for_match_not_found() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);

        let result = check_ngram_for_match(9999, &context);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_ngram_for_match_empty_docs() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);

        // Hash 4000 maps to ID 4 which has empty document set
        let result = check_ngram_for_match(4000, &context);
        assert!(result.is_none());
    }

    #[test]
    fn test_simple_contamination_cluster_new() {
        let mut doc_matches = HashMap::new();
        let mut match_set = HashSet::new();
        match_set.insert(100);
        match_set.insert(101);
        doc_matches.insert(1, match_set);

        let mut doc_boundaries = HashMap::new();
        doc_boundaries.insert(1, (5, 10));

        let mut ngram_doc_frequencies = HashMap::new();
        ngram_doc_frequencies.insert(100, 5);
        ngram_doc_frequencies.insert(101, 3);

        let cluster = SimpleContaminationCluster::new(
            doc_matches.clone(),
            doc_boundaries.clone(),
            ngram_doc_frequencies,
            15,
        );

        assert_eq!(cluster.document_matches.len(), 1);
        assert_eq!(cluster.document_boundaries.len(), 1);
        assert_eq!(cluster.end_idx, 15);
        assert_eq!(cluster.document_matches[&1].len(), 2);
        assert_eq!(cluster.document_boundaries[&1], (5, 10));
    }

    #[test]
    fn test_simple_contamination_cluster_clone() {
        let mut doc_matches = HashMap::new();
        let mut match_set = HashSet::new();
        match_set.insert(100);
        doc_matches.insert(1, match_set);

        let mut doc_boundaries = HashMap::new();
        doc_boundaries.insert(1, (5, 10));

        let mut ngram_doc_frequencies = HashMap::new();
        ngram_doc_frequencies.insert(100, 5);

        let cluster = SimpleContaminationCluster::new(
            doc_matches,
            doc_boundaries,
            ngram_doc_frequencies,
            15,
        );

        let cloned = cluster.clone();
        assert_eq!(cloned.document_matches.len(), cluster.document_matches.len());
        assert_eq!(cloned.document_boundaries.len(), cluster.document_boundaries.len());
        assert_eq!(cloned.end_idx, cluster.end_idx);
    }

    #[test]
    fn test_traverse_left_basic() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        // Create mock word tokens - using simple values that won't hash to our test values
        let word_tokens = vec![100, 101, 102, 103, 104, 105, 106, 107, 108];

        let mut active_documents = HashSet::new();
        active_documents.insert(10);
        active_documents.insert(11);

        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        document_matches.insert(10, HashSet::new());
        document_matches.insert(11, HashSet::new());

        let mut document_misses = HashMap::new();
        document_misses.insert(10, 0);
        document_misses.insert(11, 0);

        let mut document_boundaries = HashMap::new();
        document_boundaries.insert(10, (5, 5));
        document_boundaries.insert(11, (5, 5));

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        let result = traverse_left(
            5, // Start from middle
            &word_tokens,
            &context,
            &mut state,
            Some(&stats),
        );

        // Should traverse left until it hits boundary or runs out of matches
        assert!(result <= 5);
    }

    #[test]
    fn test_traverse_left_boundary() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102];

        let mut active_documents = HashSet::new();
        active_documents.insert(10);

        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut document_misses = HashMap::new();
        let mut document_boundaries = HashMap::new();

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        let result = traverse_left(
            0, // Start at boundary
            &word_tokens,
            &context,
            &mut state,
            Some(&stats),
        );

        assert_eq!(result, 0); // Should return immediately at boundary
    }

    #[test]
    fn test_traverse_right_basic() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102, 103, 104, 105, 106, 107, 108];
        let total_ngrams = word_tokens.len() - context.ngram_size + 1;

        let mut active_documents = HashSet::new();
        active_documents.insert(10);
        active_documents.insert(11);

        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        document_matches.insert(10, HashSet::new());
        document_matches.insert(11, HashSet::new());

        let mut document_misses = HashMap::new();
        document_misses.insert(10, 0);
        document_misses.insert(11, 0);

        let mut document_boundaries = HashMap::new();
        document_boundaries.insert(10, (3, 3));
        document_boundaries.insert(11, (3, 3));

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        let result = traverse_right(
            3, // Start from middle
            &word_tokens,
            &context,
            total_ngrams,
            &mut state,
            Some(&stats),
        );

        // Should traverse right until it hits boundary or runs out of matches
        assert!(result >= 3);
        assert!(result < total_ngrams);
    }

    #[test]
    fn test_traverse_right_boundary() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102];
        let total_ngrams = 1; // Only one n-gram possible

        let mut active_documents = HashSet::new();
        active_documents.insert(10);

        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut document_misses = HashMap::new();
        let mut document_boundaries = HashMap::new();

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        let result = traverse_right(
            0, // Already at the last valid position
            &word_tokens,
            &context,
            total_ngrams,
            &mut state,
            Some(&stats),
        );

        assert_eq!(result, 0); // Should return immediately at boundary
    }

    #[test]
    fn test_expand_contamination_cluster_insufficient_tokens() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(5, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![1, 2, 3]; // Less than ngram_size
        let initial_docs = HashSet::new();
        let initial_ngram = vec![1, 2, 3];

        let result = expand_simple_contamination_cluster(
            0,
            &word_tokens,
            &context,
            initial_docs,
            &initial_ngram,
            Some(&stats),
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_expand_contamination_cluster_basic() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102, 103, 104, 105, 106];

        let mut initial_docs = HashSet::new();
        initial_docs.insert(10);
        initial_docs.insert(11);

        let initial_ngram = vec![102, 103, 104];

        let result = expand_simple_contamination_cluster(
            2, // Start position for the initial n-gram
            &word_tokens,
            &context,
            initial_docs.clone(),
            &initial_ngram,
            Some(&stats),
        );

        assert!(result.is_some());
        let cluster = result.unwrap();

        // Check that document matches were initialized
        for doc_id in &initial_docs {
            assert!(cluster.document_matches.contains_key(doc_id));
            assert!(cluster.document_boundaries.contains_key(doc_id));
        }

        // Check boundaries make sense
        for (start, end) in cluster.document_boundaries.values() {
            assert!(start <= end);
            assert!(*start < word_tokens.len());
            assert!(*end < word_tokens.len());
        }
    }

    #[test]
    fn test_expand_contamination_cluster_single_document() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102, 103, 104];

        let mut initial_docs = HashSet::new();
        initial_docs.insert(10); // Single document

        let initial_ngram = vec![101, 102, 103];

        let result = expand_simple_contamination_cluster(
            1,
            &word_tokens,
            &context,
            initial_docs,
            &initial_ngram,
            Some(&stats),
        );

        assert!(result.is_some());
        let cluster = result.unwrap();

        assert_eq!(cluster.document_matches.len(), 1);
        assert!(cluster.document_matches.contains_key(&10));
        assert_eq!(cluster.document_boundaries.len(), 1);
        assert!(cluster.document_boundaries.contains_key(&10));
    }

    #[test]
    fn test_document_removal_on_misses() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2); // max 2 consecutive misses
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102, 103, 104, 105, 106, 107];

        let mut active_documents = HashSet::new();
        active_documents.insert(10);
        active_documents.insert(11);

        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut document_misses = HashMap::new();
        document_misses.insert(10, 1); // Already has 1 miss
        document_misses.insert(11, 0);

        let mut document_boundaries = HashMap::new();

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        // This will accumulate misses since our tokens don't hash to test values
        let _result = traverse_right(
            0,
            &word_tokens,
            &context,
            word_tokens.len() - 2,
            &mut state,
            Some(&stats),
        );

        // Documents should be removed after hitting max consecutive misses
        assert!(active_documents.is_empty() || document_misses.values().all(|&v| v < 2));
    }

    #[test]
    fn test_empty_active_documents() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        let word_tokens = vec![100, 101, 102, 103, 104];

        let mut active_documents = HashSet::new(); // Empty from start
        let mut document_matches: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut document_misses = HashMap::new();
        let mut document_boundaries = HashMap::new();

        let mut ngram_doc_frequencies = HashMap::new();

        let mut state = TraversalState {
            active_documents: &mut active_documents,
            document_matches: &mut document_matches,
            document_misses: &mut document_misses,
            document_boundaries: &mut document_boundaries,
            ngram_doc_frequencies: &mut ngram_doc_frequencies,
        };

        let result = traverse_left(
            2,
            &word_tokens,
            &context,
            &mut state,
            Some(&stats),
        );

        // Should return immediately with empty active documents
        assert_eq!(result, 2);
    }

    #[test]
    fn test_stats_tracking() {
        let (context, _ngram_box, _id_box, _hot_bucket_box) = create_test_context(3, 2);
        let stats = StatsContainer::new();

        // Add a dummy file/line count to make sure stats aren't filtered out during aggregation
        stats.increment_files_processed();
        stats.add_lines_processed(1);

        let word_tokens = vec![100, 101, 102, 103, 104, 105];

        let mut initial_docs = HashSet::new();
        initial_docs.insert(10);

        let initial_ngram = vec![101, 102, 103];

        let _result = expand_simple_contamination_cluster(
            1,
            &word_tokens,
            &context,
            initial_docs,
            &initial_ngram,
            Some(&stats),
        );

        // Stats should have been updated
        let aggregated = stats.aggregate();
        assert_eq!(aggregated.total_question_expansion_calls, 1);
        assert!(aggregated.total_question_expansion_time_us > 0);
        // Left and right traversals should have occurred
        assert!(aggregated.total_left_traversals > 0 || aggregated.total_right_traversals > 0);
    }
}




