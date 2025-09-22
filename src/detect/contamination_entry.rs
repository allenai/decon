use std::collections::HashSet;
use std::time::Instant;

use crate::detect::SimpleReferenceIndex;
use crate::common::Config;
use crate::detect::stats::StatsContainer;
use crate::detect::cluster::SimpleContaminationCluster;
use crate::detect::answer_boundary::{
    find_short_answer_exact_match, find_answer_boundaries_with_ngrams,
};
use crate::detect::passage_boundary::{
    find_passage_boundaries_with_ngrams,
};
use crate::detect::scoring::{
    calculate_combined_contamination_score, ScoringParams,
};

/// Context for matching operations to reduce function parameters
pub(crate) struct MatchingContext<'a> {
    pub cluster: &'a SimpleContaminationCluster,
    pub doc_id: u32,
    pub training_tokens: &'a [usize],
    pub config: &'a Config,
    pub index: &'a SimpleReferenceIndex,
    pub excess_tokens: usize,
}

/// Metrics returned from contamination check functions
struct ContaminationMetrics {
    found: bool,
    overlap_ratio: Option<f32>,
    idf_overlap: Option<f32>,
    boundaries: Option<Vec<(usize, usize)>>,
    token_length: Option<usize>,
}

/// Result from finding matching answers in training data
#[derive(Debug, Clone)]
pub(crate) struct AnswerMatchResult {
    pub overlap_ratio: f32,
    pub idf_overlap: f32,
    pub boundaries: Option<Vec<(usize, usize)>>,
}

/// Result from matching answer tokens
#[derive(Debug)]
pub(crate) struct AnswerTokenMatchResult {
    pub boundaries: Option<Vec<(usize, usize)>>,
    pub overlap_ratio: f32,
    pub idf_overlap: f32,
}

// Struct to return contamination check results
pub(crate) struct ContaminationCheckResult {
    pub is_contaminated: bool,
    pub contamination_score: f32,  // The combined contamination score
    pub eval_has_answer: bool,
    pub answer_overlap_ratio: Option<f32>,
    pub answer_idf_overlap: Option<f32>,
    pub answer_boundaries: Option<Vec<(usize, usize)>>,
    pub answer_token_length: Option<usize>,  // Unique token count for answer
    pub eval_has_passage: bool,
    pub passage_overlap_ratio: Option<f32>,
    pub passage_idf_overlap: Option<f32>,
    pub passage_boundaries: Option<(usize, usize)>,
}

impl Default for ContaminationCheckResult {
    fn default() -> Self {
        Self {
            is_contaminated: false,
            contamination_score: 0.0,
            eval_has_answer: false,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_boundaries: None,
            answer_token_length: None,
            eval_has_passage: false,
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            passage_boundaries: None,
        }
    }
}

#[derive(Clone)]
pub struct SimpleContaminationEntry {
    pub training_line: usize,
    pub(crate) eval_key: String,   // The unique eval dataset identifier from config
    pub(crate) eval_line: usize,
    pub(crate) eval_instance_index: usize,  // Generally the line number from a eval, config, split
    pub(crate) split: Option<String>,  // The split (train/validation/test) of the eval dataset
    pub(crate) idf_overlap: Option<f32>, // IDF overlap ratio: matched_idf_sum / eval_total_idf_sum for question
    // Token indices for position recovery
    pub(crate) contamination_start_idx: Option<usize>, // Start index in token array
    pub(crate) contamination_end_idx: Option<usize>,   // End index in token array
    pub(crate) training_overlap_text: Option<String>,  // For reporting display
    pub(crate) ngram_match_cnt: usize,    // Number of _unique_ n-gram matches
    pub(crate) eval_unique_ngrams: usize, // Total unique n-grams in the eval document
    // Answer contamination fields
    pub(crate) eval_has_answer: bool,              // Whether the eval instance has an answer
    pub(crate) answer_overlap_ratio: Option<f32>, // Overlap ratio for answer tokens
    pub(crate) answer_idf_overlap: Option<f32>,   // IDF overlap ratio for answer tokens
    pub(crate) answer_boundaries: Option<Vec<(usize, usize)>>, // Multiple answer boundary ranges
    // Legacy fields for backward compatibility (will contain first range if exists)
    pub(crate) answer_start_idx: Option<usize>, // Start index of first answer match
    pub(crate) answer_end_idx: Option<usize>,   // End index of last answer match
    // Passage contamination fields
    pub(crate) eval_has_passage: bool,             // Whether the eval instance has a passage
    pub(crate) passage_overlap_ratio: Option<f32>, // Overlap ratio for passage tokens
    pub(crate) passage_idf_overlap: Option<f32>,   // IDF overlap ratio for passage tokens
    pub(crate) passage_start_idx: Option<usize>, // Start index of passage match in training tokens
    pub(crate) passage_end_idx: Option<usize>,   // End index of passage match in training tokens
    // Token length tracking fields
    pub(crate) cluster_token_length: Option<usize>, // Length of the matched cluster in tokens
    pub(crate) eval_token_length: Option<usize>,    // Total length of the eval document in tokens
    pub(crate) answer_token_length: Option<usize>,  // Answer token length from eval (unique count)
    pub(crate) answer_total_token_count: Option<usize>,  // Total answer token count (including duplicates)
    pub(crate) passage_total_token_count: Option<usize>,  // Total passage token count (including duplicates)
    // Separate eval text fields for display
    pub(crate) eval_question_text: Option<String>,  // Original question text from eval
    pub(crate) eval_answer_text: Option<String>,    // Original answer text from eval
    pub(crate) eval_passage_text: Option<String>,   // Original passage text from eval
    // New fields for fingerprint and is_correct
    pub(crate) fingerprint: Option<String>,         // Fingerprint from eval dataset
    pub(crate) is_correct: Option<bool>,            // Whether the answer is correct
    // Reference file tracking
    pub(crate) reference_file: Option<String>,      // Reference filename this match came from
    // Final contamination score used for threshold comparison
    pub(crate) contamination_score: Option<f32>,     // The contamination score calculated during detection
}

impl SimpleContaminationEntry {
    /// Calculate excess tokens for answer/passage lookups. This is a bit of kludge, but essentially corrects
    /// for an edge case where the question has repeated ngrams and they are placed on the boundary of the
    /// match. Because we are going non-comparative, we never actually look at the ngrams or their sequence
    /// in the eval, we're only detecting the boundaries of n-gram presence, and as such, it is possible
    /// to double count. This doesn't have an impact on scoring because we're using unique ngram idf sums, but
    /// it does have an impact on boundaries. In cases where the answer shares n-grams with question at the
    /// start of the answer it can lead to missing the answer. This needs a little work and there might be a better
    /// approach, but we adjust the start of the answer window by this amount and empirically it helps.
    fn calculate_excess_tokens(&self) -> usize {
        if let (Some(cluster_len), Some(eval_len)) =
            (self.cluster_token_length, self.eval_token_length) {
            cluster_len.saturating_sub(eval_len)
        } else {
            0
        }
    }

    /// Check for passage contamination and return the relevant metrics
    fn check_passage_contamination(
        &self,
        context: &MatchingContext,
        stats: Option<&StatsContainer>,
    ) -> ContaminationMetrics {
        if let Some(passage_token_set) = context.index.eval_doc_id_to_passage_tokens.get(&context.doc_id) {
            let passage_total_token_count = self.passage_total_token_count.unwrap_or(0);
            let (_passage_found, p_overlap_ratio, p_idf_overlap, p_boundaries) =
                find_matching_passage(
                    passage_token_set,
                    passage_total_token_count,
                    context,
                    stats,
                );
            // Convert single boundary to vec
            let boundaries_vec = p_boundaries.map(|b| vec![b]);
            ContaminationMetrics {
                found: true,
                overlap_ratio: Some(p_overlap_ratio),
                idf_overlap: Some(p_idf_overlap),
                boundaries: boundaries_vec,
                token_length: self.passage_total_token_count,
            }
        } else {
            ContaminationMetrics {
                found: false,
                overlap_ratio: None,
                idf_overlap: None,
                boundaries: None,
                token_length: None,
            }
        }
    }

    /// Check for answer contamination and return the relevant metrics
    fn check_answer_contamination(
        &self,
        context: &MatchingContext,
        stats: Option<&StatsContainer>,
    ) -> ContaminationMetrics {
        if let Some(answer_token_set) = context.index.eval_doc_id_to_answer_tokens.get(&context.doc_id) {
            let answer_total_token_count = self.answer_total_token_count.unwrap_or(0);
            let answer_token_length = Some(answer_token_set.len());

            let answer_result = find_matching_answer(
                answer_token_set,
                answer_total_token_count,
                context,
                stats,
            );

            ContaminationMetrics {
                found: true,
                overlap_ratio: Some(answer_result.overlap_ratio),
                idf_overlap: Some(answer_result.idf_overlap),
                boundaries: answer_result.boundaries,
                token_length: answer_token_length,
            }
        } else {
            ContaminationMetrics {
                found: false,
                overlap_ratio: None,
                idf_overlap: None,
                boundaries: None,
                token_length: None,
            }
        }
    }

    /// Check if this entry represents contamination (score >= threshold)
    /// Returns a ContaminationCheckResult with contamination status and exclusion reason
    pub(crate) fn is_contaminated(
        &self,
        doc_id: u32,
        cluster: &SimpleContaminationCluster,
        training_tokens: &[usize],
        config: &Config,
        index: &SimpleReferenceIndex,
        stats: Option<&StatsContainer>,
    ) -> ContaminationCheckResult {
        // TODO: Optimize duplicate index lookups - passage_token_length is fetched 3 times,
        // answer_token_length 2 times. Could pass these values down the call chain to avoid
        // redundant HashMap lookups (saves ~4 lookups per contaminated document).

        // Initialize result with default values
        let mut result = ContaminationCheckResult::default();

        // Calculate excess tokens for answer/passage lookups
        let excess_tokens = self.calculate_excess_tokens();

        // Create MatchingContext once for both checks
        let context = MatchingContext {
            cluster,
            doc_id,
            training_tokens,
            config,
            index,
            excess_tokens,
        };

        // Check for passage contamination
        let passage_metrics = self.check_passage_contamination(&context, stats);

        result.eval_has_passage = passage_metrics.found;
        result.passage_overlap_ratio = passage_metrics.overlap_ratio;
        result.passage_idf_overlap = passage_metrics.idf_overlap;
        // Convert vec back to single boundary for passage (take first element)
        result.passage_boundaries = passage_metrics.boundaries.and_then(|v| v.into_iter().next());

        // Check for answer contamination
        let answer_metrics = self.check_answer_contamination(&context, stats);

        result.eval_has_answer = answer_metrics.found;
        result.answer_overlap_ratio = answer_metrics.overlap_ratio;
        result.answer_idf_overlap = answer_metrics.idf_overlap;
        result.answer_boundaries = answer_metrics.boundaries;
        result.answer_token_length = answer_metrics.token_length;

        // Create scoring parameters with the calculated scores for combined scoring
        let scoring_params = ScoringParams {
            question_idf: self.idf_overlap.unwrap_or(0.0),
            eval_unique_ngrams: self.eval_unique_ngrams,
            eval_token_length: self.eval_token_length,
            eval_has_answer: result.eval_has_answer,
            answer_idf_overlap: result.answer_idf_overlap,
            answer_token_length: result.answer_token_length,
            eval_has_passage: result.eval_has_passage,
            passage_idf_overlap: result.passage_idf_overlap,
            passage_token_length: passage_metrics.token_length,
        };

        // Calculate combined contamination score using all available components
        let combined_score = calculate_combined_contamination_score(&scoring_params, config);

        // Update final contamination fields
        result.contamination_score = combined_score;
        result.is_contaminated = combined_score >= config.contamination_score_threshold;

        result
    }
}


/// Check if the passage matches - only looks before the contamination cluster
/// Returns (is_contaminated, overlap_ratio, passage_idf_overlap, boundaries)
pub(crate) fn find_matching_passage(
    passage_token_set: &HashSet<usize>,
    passage_total_token_count: usize,
    context: &MatchingContext,
    stats: Option<&StatsContainer>,
) -> (bool, f32, f32, Option<(usize, usize)>) {
    // Only track timing if stats are being collected
    let start_time = if stats.is_some() {
        Some(Instant::now())
    } else {
        None
    };

    // Get matching tokens, boundaries and IDF overlap from n-gram based detection
    let (matching_tokens, passage_boundaries, passage_idf_overlap) = passage_tokens_lookup(
        passage_token_set,
        passage_total_token_count,
        context,
    );

    // Calculate token overlap ratio for display (not used for thresholding)
    let passage_overlap_ratio = if passage_token_set.is_empty() {
        0.0
    } else {
        matching_tokens.len() as f32 / passage_token_set.len() as f32
    };

    // Passage checking is always treated as contaminated if found
    let is_contaminated = true;

    // Track timing in stats if available
    if let (Some(stats), Some(start)) = (stats, start_time) {
        let elapsed = start.elapsed().as_micros() as u64;
        stats.add_passage_cluster_time(elapsed);
    }

    (
        is_contaminated,
        passage_overlap_ratio,
        passage_idf_overlap,
        passage_boundaries,
    )
}

/// Check if the answer matches - only looks after the contamination cluster
/// Returns AnswerMatchResult with contamination status and metrics
pub(crate) fn find_matching_answer(
    answer_token_set: &HashSet<usize>,
    answer_total_token_count: usize,
    context: &MatchingContext,
    stats: Option<&StatsContainer>,
) -> AnswerMatchResult {
    // Only track timing if stats are being collected
    let start_time = if stats.is_some() {
        Some(Instant::now())
    } else {
        None
    };

    // Get matching tokens, boundaries, overlap ratio and IDF from match_answer_tokens
    let token_result = match_answer_tokens(
        answer_token_set,
        answer_total_token_count,
        context,
    );

    // Track timing in stats if available
    if let (Some(stats), Some(start)) = (stats, start_time) {
        let elapsed = start.elapsed().as_micros() as u64;
        stats.add_answer_cluster_time(elapsed);
    }

    AnswerMatchResult {
        overlap_ratio: token_result.overlap_ratio,
        idf_overlap: token_result.idf_overlap,
        boundaries: token_result.boundaries,
    }
}

pub(crate) fn passage_tokens_lookup(
    passage_token_set: &HashSet<usize>,
    passage_total_token_count: usize,
    context: &MatchingContext,
) -> (HashSet<usize>, Option<(usize, usize)>, f32) {
    // Use the passed passage_total_token_count instead of looking it up
    let window_size = std::cmp::max(passage_total_token_count * 2, context.config.min_passage_distance);

    // Get document-specific boundaries
    let (doc_start_idx, _doc_end_idx) = context.cluster
        .document_boundaries
        .get(&context.doc_id)
        .copied()
        .expect("Document boundaries should exist for all matched documents");

    // Use n-gram based passage boundary detection
    let (passage_boundaries, idf_overlap) = if let Some((start, end, idf)) =
        find_passage_boundaries_with_ngrams(
            context.doc_id,
            doc_start_idx,
            context.training_tokens,
            window_size,
            context.config.passage_ngram_size,
            context.config.passage_max_consecutive_misses,
            context.index,
        ) {
        (Some((start, end)), idf)
    } else {
        (None, 0.0)
    };

    // Extract actual matching tokens within the boundaries for display
    let actual_matches = if let Some((start, end)) = passage_boundaries {
        // Get tokens within the boundaries and filter for actual matches
        let boundary_tokens: HashSet<usize> = context.training_tokens[start..=end]
            .iter()
            .filter(|&&token| passage_token_set.contains(&token))
            .copied()
            .collect();
        boundary_tokens
    } else {
        // If no boundaries found, return empty set
        HashSet::new()
    };

    (actual_matches, passage_boundaries, idf_overlap)
}

pub(crate) fn match_answer_tokens(
    answer_token_set: &HashSet<usize>,
    answer_total_token_count: usize,
    context: &MatchingContext,
) -> AnswerTokenMatchResult {
    // Use the passed answer_total_token_count instead of looking it up
    let answer_length = answer_total_token_count;

    // Calculate window size based on answer length
    let window_size = if answer_length <= context.config.short_answer_token_threshold {
        context.config.short_answer_window_length
    } else {
        std::cmp::max(answer_length * 2, context.config.min_long_answer_window)
    };

    // Get document-specific boundaries
    let (_doc_start_idx, doc_end_idx) = context.cluster
        .document_boundaries
        .get(&context.doc_id)
        .copied()
        .expect("Document boundaries should exist for all matched documents");

    // Always exclude question tokens, search only in suffix region (after the contamination)
    // Note: doc_end_idx is the last n-gram position, but we need the last token position
    // Adjust for excess tokens when cluster is longer than eval question
    let question_end_idx = (doc_end_idx + context.config.ngram_size - 1).saturating_sub(context.excess_tokens);
    let suffix_search_start = question_end_idx + 1;
    let suffix_search_end = (suffix_search_start + window_size).min(context.training_tokens.len());

    // Collect tokens from suffix region only
    let mut training_token_set = HashSet::new();

    // Add suffix tokens and decode for debug display
    let mut suffix_tokens = Vec::new();
    if suffix_search_start < suffix_search_end && suffix_search_start < context.training_tokens.len() {
        let end = suffix_search_end.min(context.training_tokens.len());
        suffix_tokens.extend(&context.training_tokens[suffix_search_start..end]);
        training_token_set.extend(suffix_tokens.iter().copied());
    }

    // Find matching tokens from suffix only
    let suffix_token_set: HashSet<usize> = suffix_tokens.iter().copied().collect();

    let suffix_matches: HashSet<usize> = answer_token_set
        .iter()
        .filter(|token| suffix_token_set.contains(token))
        .copied()
        .collect();

    // Get ordered answer tokens for exact/bi-gram matching
    let answer_tokens_ordered = context.index.eval_doc_id_to_answer_tokens_ordered
        .get(&context.doc_id)
        .cloned()
        .unwrap_or_default();

    // Find answer boundaries using appropriate method based on answer length
    let (answer_boundaries, answer_idf_overlap) = if answer_length <= context.config.short_answer_token_threshold {
        // Use exact matching for short answers with token-level IDF
        match find_short_answer_exact_match(
            &answer_tokens_ordered,
            answer_token_set,
            question_end_idx,
            context.training_tokens,
            window_size,
            &context.index.answer_token_idf,
        ) {
            Some((boundaries, idf)) => (Some(boundaries), idf),
            None => (None, 0.0),
        }
    } else {
        // Use n-gram matching for longer answers with n-gram IDF
        match find_answer_boundaries_with_ngrams(
            context.doc_id,
            question_end_idx,
            context.training_tokens,
            window_size,
            context.config.answer_ngram_size,
            context.index,
        ) {
            Some((boundaries, idf)) => (Some(boundaries), idf),
            None => (None, 0.0),
        }
    };

    // TODO: consider getting rid of overlap ratio here....
    // Extract actual matching tokens within all boundary ranges
    let actual_matches = if let Some(ref boundaries) = answer_boundaries {
        let mut all_boundary_tokens = HashSet::new();
        // Collect tokens from all boundary ranges
        for &(start, end) in boundaries {
            let boundary_tokens: HashSet<usize> = context.training_tokens[start..=end]
                .iter()
                .filter(|&&token| answer_token_set.contains(&token))
                .copied()
                .collect();
            all_boundary_tokens.extend(boundary_tokens);
        }
        all_boundary_tokens
    } else {
        // If no boundaries found, return the original suffix matches
        suffix_matches
    };

    // Calculate overlap ratio
    let answer_overlap_ratio = if answer_token_set.is_empty() {
        0.0
    } else {
        actual_matches.len() as f32 / answer_token_set.len() as f32
    };

    AnswerTokenMatchResult {
        boundaries: answer_boundaries,
        overlap_ratio: answer_overlap_ratio,
        idf_overlap: answer_idf_overlap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Config;
    use crate::detect::scoring::{calculate_answer_confidence, interpolate_threshold};
    use std::path::PathBuf;

    fn create_test_config() -> Config {
        use crate::detect::scoring::calculate_minimum_question_idf_threshold;
        let contamination_score_threshold = 0.8;
        Config {
            mode: "simple".to_string(),
            ngram_size: 5,
            tokenizer_str: "cl100k".to_string(),
            hash_seed: 0,
            content_key: "text".to_string(),
            training_dir: PathBuf::from("/tmp/input"),
            evals_dir: PathBuf::from("/tmp/ref"),
            report_output_dir: PathBuf::from("/tmp/report"),
            cleaned_output_dir: None,
            exact_override: false,
            sample_every_m_tokens: 1,
            question_max_consecutive_misses: 2,
            ngram_bucket_lru_cache: 10000,
            punctuation_chars: String::new(),
            worker_threads: 4,
            window_size_increment: None,
            num_windows: None,
            window_step_size: None,
            short_answer_window_length: 50,
            min_long_answer_window: 100,
            short_answer_token_threshold: 3,
            answer_ngram_size: 2,
            verbose: false,
            min_passage_distance: 300,
            passage_max_consecutive_misses: 5,
            passage_ngram_size: 3,
            contamination_score_threshold,
            minimum_question_idf_threshold: calculate_minimum_question_idf_threshold(contamination_score_threshold),
            purify: false,
            eval_dedup: true,
            eval_min_token_length: 20,
            eval_min_unique_word_count: 4,
            perfect_match_decay_start: Some(20),
            perfect_match_decay_end: Some(50),
            index_passages: true,
            index_answers: true,
        }
    }

    #[test]
    fn test_get_required_threshold_with_token_length() {
        let config = create_test_config();
        let mut entry = SimpleContaminationEntry {
            training_line: 1,
            eval_key: "test_key".to_string(),
            eval_line: 1,
            eval_instance_index: 0,
            split: None,
            idf_overlap: Some(0.5),
            contamination_start_idx: None,
            contamination_end_idx: None,
            training_overlap_text: None,
            ngram_match_cnt: 10,
            eval_unique_ngrams: 20,
            eval_has_answer: false,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_boundaries: None,
            answer_start_idx: None,
            answer_end_idx: None,
            eval_has_passage: false,
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            passage_start_idx: None,
            passage_end_idx: None,
            cluster_token_length: Some(100),
            eval_token_length: Some(50),
            answer_token_length: None,
            answer_total_token_count: None,
            passage_total_token_count: None,
            eval_question_text: None,
            eval_answer_text: None,
            eval_passage_text: None,
            fingerprint: None,
            is_correct: None,
            reference_file: None,
            contamination_score: None,
        };

        // Test with short eval token length
        entry.eval_token_length = Some(10);
        let threshold = interpolate_threshold(
            entry.eval_token_length.unwrap_or(usize::MAX),
            config.perfect_match_decay_start,
            config.perfect_match_decay_end,
            config.minimum_question_idf_threshold,
        );
        assert!(threshold > 0.0);

        // Test with longer eval token length
        entry.eval_token_length = Some(1000);
        let threshold_long = interpolate_threshold(
            entry.eval_token_length.unwrap_or(usize::MAX),
            config.perfect_match_decay_start,
            config.perfect_match_decay_end,
            config.minimum_question_idf_threshold,
        );
        assert!(threshold_long > 0.0);
        // Longer documents typically have lower thresholds
        assert!(threshold_long <= threshold);

        // Test with None token length - should still return a valid threshold
        entry.eval_token_length = None;
        let threshold_none = interpolate_threshold(
            entry.eval_token_length.unwrap_or(usize::MAX),
            config.perfect_match_decay_start,
            config.perfect_match_decay_end,
            config.minimum_question_idf_threshold,
        );
        assert!(threshold_none > 0.0);
    }

    #[test]
    fn test_calculate_answer_confidence() {
        let mut entry = SimpleContaminationEntry {
            training_line: 1,
            eval_key: "test_key".to_string(),
            eval_line: 1,
            eval_instance_index: 0,
            split: None,
            idf_overlap: Some(0.5),
            contamination_start_idx: None,
            contamination_end_idx: None,
            training_overlap_text: None,
            ngram_match_cnt: 10,
            eval_unique_ngrams: 20,
            eval_has_answer: true,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_boundaries: None,
            answer_start_idx: None,
            answer_end_idx: None,
            eval_has_passage: false,
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            passage_start_idx: None,
            passage_end_idx: None,
            cluster_token_length: None,
            eval_token_length: None,
            answer_token_length: Some(5),
            answer_total_token_count: None,
            passage_total_token_count: None,
            eval_question_text: None,
            eval_answer_text: None,
            eval_passage_text: None,
            fingerprint: None,
            is_correct: None,
            reference_file: None,
            contamination_score: None,
        };

        // Test with short answer (2 tokens - below MIN_INFORMATIVE_ANSWER_TOKENS)
        entry.answer_token_length = Some(2);
        let confidence_short = calculate_answer_confidence(entry.answer_token_length);
        assert!((0.0..=1.0).contains(&confidence_short));
        assert!(confidence_short < 1.0); // Should be less than 1.0 for short answers

        // Test with medium answer (3 tokens - still below threshold)
        entry.answer_token_length = Some(3);
        let confidence_medium = calculate_answer_confidence(entry.answer_token_length);
        assert!((0.0..=1.0).contains(&confidence_medium));
        assert!(confidence_medium > confidence_short);
        assert!(confidence_medium < 1.0);

        // Test with longer answer (5 tokens - above MIN_INFORMATIVE_ANSWER_TOKENS of 4)
        entry.answer_token_length = Some(5);
        let confidence_long = calculate_answer_confidence(entry.answer_token_length);
        assert!((0.0..=1.0).contains(&confidence_long));
        assert_eq!(confidence_long, 1.0); // Should be 1.0 for answers >= 4 tokens

        // Test with None - should return 1.0 (no answer means we're fully confident in the absence)
        entry.answer_token_length = None;
        let confidence_none = calculate_answer_confidence(entry.answer_token_length);
        assert_eq!(confidence_none, 1.0);
    }

    #[test]
    fn test_match_answer_tokens_empty_answer() {
        use std::collections::HashMap;
        use crate::common::tokenizer::OmniTokenizer;

        let answer_token_set = HashSet::new();
        let cluster = SimpleContaminationCluster {
            document_matches: HashMap::new(),
            document_boundaries: {
                let mut map = HashMap::new();
                map.insert(1, (0, 100));
                map
            },
            ngram_doc_frequencies: HashMap::new(),
            end_idx: 100,
        };
        let training_tokens = vec![1, 2, 3, 4, 5];
        let config = create_test_config();

        // Create a minimal SimpleReferenceIndex
        let tokenizer = OmniTokenizer::new("cl100k").expect("Failed to create tokenizer");
        let index = SimpleReferenceIndex::new(
            HashMap::new(), // question_ngram_to_id
            HashMap::new(), // question_ngram_id_to_eval_doc_ids
            HashMap::new(), // eval_doc_id_to_answer_tokens
            HashMap::new(), // eval_doc_id_to_answer_tokens_ordered
            HashMap::new(), // eval_doc_id_to_answer_ngram_ids
            HashMap::new(), // answer_ngram_idf
            HashMap::new(), // eval_doc_id_to_passage_tokens
            HashMap::new(), // eval_doc_id_to_passage_ngram_ids
            HashMap::new(), // passage_ngram_idf
            HashMap::new(), // eval_passage_idf_cache
            HashMap::new(), // eval_documents
            HashMap::new(), // eval_document_metadata
            tokenizer,      // tokenizer
            HashMap::new(), // eval_text_snippets
            HashMap::new(), // eval_answer_text_snippets
            HashMap::new(), // eval_passage_text_snippets
            HashMap::new(), // eval_doc_id_to_question_ngram_ids
            HashMap::new(), // answer_token_idf
            vec![],         // reference_filenames
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        );

        let context = MatchingContext {
            cluster: &cluster,
            doc_id: 1,
            training_tokens: &training_tokens,
            config: &config,
            index: &index,
            excess_tokens: 0,
        };
        let result = match_answer_tokens(
            &answer_token_set,
            0,
            &context,
        );

        assert!(result.boundaries.is_none());
        assert_eq!(result.overlap_ratio, 0.0);
        assert_eq!(result.idf_overlap, 0.0);
    }

    #[test]
    fn test_passage_tokens_lookup_empty_passage() {
        use std::collections::HashMap;
        use crate::common::tokenizer::OmniTokenizer;

        let passage_token_set = HashSet::new();
        let cluster = SimpleContaminationCluster {
            document_matches: HashMap::new(),
            document_boundaries: {
                let mut map = HashMap::new();
                map.insert(1, (0, 100));
                map
            },
            ngram_doc_frequencies: HashMap::new(),
            end_idx: 100,
        };
        let training_tokens = vec![1, 2, 3, 4, 5];
        let config = create_test_config();

        // Create a minimal SimpleReferenceIndex
        let tokenizer = OmniTokenizer::new("cl100k").expect("Failed to create tokenizer");
        let index = SimpleReferenceIndex::new(
            HashMap::new(), // question_ngram_to_id
            HashMap::new(), // question_ngram_id_to_eval_doc_ids
            HashMap::new(), // eval_doc_id_to_answer_tokens
            HashMap::new(), // eval_doc_id_to_answer_tokens_ordered
            HashMap::new(), // eval_doc_id_to_answer_ngram_ids
            HashMap::new(), // answer_ngram_idf
            HashMap::new(), // eval_doc_id_to_passage_tokens
            HashMap::new(), // eval_doc_id_to_passage_ngram_ids
            HashMap::new(), // passage_ngram_idf
            HashMap::new(), // eval_passage_idf_cache
            HashMap::new(), // eval_documents
            HashMap::new(), // eval_document_metadata
            tokenizer,      // tokenizer
            HashMap::new(), // eval_text_snippets
            HashMap::new(), // eval_answer_text_snippets
            HashMap::new(), // eval_passage_text_snippets
            HashMap::new(), // eval_doc_id_to_question_ngram_ids
            HashMap::new(), // answer_token_idf
            vec![],         // reference_filenames
            HashSet::new(), // unique_eval_suites
            None,           // hot_bucket_stats
            0,              // hot_bucket_id
            HashMap::new(), // question_ngram_id_to_hot_bucket_doc_ids
        );

        let context = MatchingContext {
            cluster: &cluster,
            doc_id: 1,
            training_tokens: &training_tokens,
            config: &config,
            index: &index,
            excess_tokens: 0,
        };
        let (matches, boundaries, idf_overlap) = passage_tokens_lookup(
            &passage_token_set,
            0,
            &context,
        );

        assert!(matches.is_empty());
        assert!(boundaries.is_none());
        assert_eq!(idf_overlap, 0.0);
    }
}
