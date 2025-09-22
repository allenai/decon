use std::collections::{HashMap, HashSet};

use crate::detect::reference_index::AnswerTokenIdfMap;
use crate::common::Config;

/// Lightweight struct containing only the parameters needed for contamination scoring
pub struct ScoringParams {
    pub question_idf: f32,
    pub eval_unique_ngrams: usize,
    pub eval_token_length: Option<usize>,
    pub eval_has_answer: bool,
    pub answer_idf_overlap: Option<f32>,
    pub answer_token_length: Option<usize>,
    pub eval_has_passage: bool,
    pub passage_idf_overlap: Option<f32>,
    pub passage_token_length: Option<usize>,
}

// Default weights for Question + Answer + Passage
const DEFAULT_WEIGHT_QAP_QUESTION: f32 = 0.7;
const DEFAULT_WEIGHT_QAP_ANSWER: f32 = 0.2;
const DEFAULT_WEIGHT_QAP_PASSAGE: f32 = 0.1;

// Default weights for Question + Answer
const DEFAULT_WEIGHT_QA_QUESTION: f32 = 0.75;
const DEFAULT_WEIGHT_QA_ANSWER: f32 = 0.25;

// Default weights for Question + Passage
const DEFAULT_WEIGHT_QP_QUESTION: f32 = 0.85;
const DEFAULT_WEIGHT_QP_PASSAGE: f32 = 0.15;

// Note: calculate_eval_document_idf_sum has been removed because we now pre-compute
// question IDF sums during index building and store them in EvalDocuments

/// Calculate question IDF overlap ratio between matched n-grams and eval document
/// Returns the ratio of matched n-gram IDF sum to total eval document IDF sum
pub fn calculate_question_idf_overlap(
    matched_ngram_ids: &HashSet<u32>,
    eval_total_idf: f32,  // Now passed as pre-computed value
    ngram_doc_frequencies: &HashMap<u32, usize>,  // Required cached frequencies
    total_docs: f32,
) -> Option<f32> {
    // Calculate IDF sum for matched n-grams (shared between training and eval)
    let mut matched_idf_sum = 0.0f32;

    // Sort ngram_ids for deterministic iteration order
    let mut sorted_ids: Vec<u32> = matched_ngram_ids.iter().copied().collect();
    sorted_ids.sort_unstable();

    for ngram_id in sorted_ids {
        // Get the cached document frequency
        if let Some(&doc_freq) = ngram_doc_frequencies.get(&ngram_id) {
            // Standard IDF formula: ln(N/doc_freq)
            let idf = (total_docs / doc_freq as f32).ln();
            matched_idf_sum += idf;
        }
        // Skip ngrams not in the cache (shouldn't happen, but safe to ignore)
    }

    // Calculate IDF overlap ratio using the pre-computed eval document total IDF
    if eval_total_idf > 0.0 {
        Some(matched_idf_sum / eval_total_idf)
    } else {
        Some(0.0)
    }
}

/// Generic threshold interpolation based on token length
/// Returns a threshold value between 1.0 (perfect match required) and target_threshold
pub fn interpolate_threshold(
    token_length: usize,
    perfect_start: Option<usize>,
    threshold_end: Option<usize>,
    target_threshold: f32,
) -> f32 {
    match (perfect_start, threshold_end) {
        (Some(perfect_start), Some(threshold_end)) => {
            if token_length <= perfect_start {
                1.0 // Perfect match required
            } else if token_length >= threshold_end {
                target_threshold // Normal threshold
            } else if perfect_start == threshold_end {
                // Step function: immediate transition at the threshold
                target_threshold
            } else {
                // Linear interpolation between 1.0 and target_threshold
                let range = (threshold_end - perfect_start) as f32;
                let position = (token_length - perfect_start) as f32;
                let ratio = position / range;
                // Interpolate: start at 1.0, end at target_threshold
                1.0 - (1.0 - target_threshold) * ratio
            }
        }
        (Some(perfect_start), None) => {
            // Original behavior - hard cutoff
            if token_length <= perfect_start {
                1.0
            } else {
                target_threshold
            }
        }
        _ => target_threshold,
    }
}


/// Calculate the minimum question IDF required for any possible contamination
/// This represents the theoretical minimum with perfect answer/passage scores
pub fn calculate_minimum_question_idf_threshold(contamination_score_threshold: f32) -> f32 {
    // QAP case gives the lowest possible threshold (most permissive)
    // With perfect answer (1.0) and passage (1.0):
    // question_idf * 0.7 + 0.3 >= contamination_score_threshold
    // question_idf >= (contamination_score_threshold - 0.3) / 0.7
    let max_other_contribution = DEFAULT_WEIGHT_QAP_ANSWER + DEFAULT_WEIGHT_QAP_PASSAGE;
    let min_threshold = (contamination_score_threshold - max_other_contribution) / DEFAULT_WEIGHT_QAP_QUESTION;

    // Ensure it's not negative (though it shouldn't be with reasonable thresholds)
    min_threshold.max(0.0)
}

/// Calculate answer confidence based on answer token length
pub fn calculate_answer_confidence(answer_token_length: Option<usize>) -> f32 {
    const MIN_INFORMATIVE_ANSWER_TOKENS: f32 = 4.0;

    if let Some(answer_len) = answer_token_length {
        let answer_tokens = answer_len as f32;

        if answer_tokens < MIN_INFORMATIVE_ANSWER_TOKENS {
            // Scale from 0.5 to 1.0 based on answer length
            0.5 + (answer_tokens / MIN_INFORMATIVE_ANSWER_TOKENS) * 0.5
        } else {
            1.0
        }
    } else {
        1.0
    }
}

/// Calculate combined contamination score using question, answer, and passage IDF overlaps.
/// It supports four main scenarios, QAP, QA, QP, Q.
///
/// It starts with a default weighting for each scenario, and will adjust the weights based on lengths
/// in decreasing order of important.
///
/// In general the question (aka prompt) component is the most important by far.
///
/// Because evals are often derived from common source material, the shorter the prompt, the more likely
/// it is common, making the presence of substantial portions of the answer and passage important
/// to avoid false positives.
///
/// Returns a score between 0.0 and 1.0, where higher scores indicate more likely contamination
pub fn calculate_combined_contamination_score(params: &ScoringParams, config: &Config) -> f32 {
    let question_idf = params.question_idf;
    let answer_idf = params.answer_idf_overlap.unwrap_or(0.0);
    let passage_idf = params.passage_idf_overlap.unwrap_or(0.0);

    // Calculate question informativeness factor based on unique n-grams
    // Short questions (few n-grams) are less informative and should rely more on answer/passage
    let question_ngrams = params.eval_unique_ngrams as f32;
    let min_informative_ngrams = 20.0;  // Below this, question needs strong support

    // Scale confidence based on question length
    let question_confidence = if question_ngrams < min_informative_ngrams {
        // Very short questions have reduced confidence
        // Scale from 0.5 (at 1 n-gram) to 1.0 (at 20 n-grams)
        0.5 + (question_ngrams / min_informative_ngrams) * 0.5
    } else {
        // Questions with 20+ n-grams have full confidence
        1.0
    };

    // Adjust weights based on question confidence and available components
    let base_score = if params.eval_has_answer && params.eval_has_passage {
        let answer_confidence = calculate_answer_confidence(params.answer_token_length);

        // Handle redistribution based on both question and answer confidence
        if question_confidence < 1.0 || answer_confidence < 1.0 { // QAP
            // Stage 1: Apply question confidence
            let q_weight = DEFAULT_WEIGHT_QAP_QUESTION * question_confidence;
            let q_redistribution = DEFAULT_WEIGHT_QAP_QUESTION * (1.0 - question_confidence);

            // Question's lost weight goes 50% to answer, 50% to passage
            let a_weight_after_q = DEFAULT_WEIGHT_QAP_ANSWER + q_redistribution * 0.5;
            let p_weight_after_q = DEFAULT_WEIGHT_QAP_PASSAGE + q_redistribution * 0.5;

            // Stage 2: Apply answer confidence
            let final_a_weight = a_weight_after_q * answer_confidence;
            let a_redistribution = a_weight_after_q * (1.0 - answer_confidence);

            // Answer's lost weight goes entirely to passage
            let final_p_weight = p_weight_after_q + a_redistribution;

            question_idf * q_weight + answer_idf * final_a_weight + passage_idf * final_p_weight
        } else {
            // Standard weights for long questions and answers
            question_idf * DEFAULT_WEIGHT_QAP_QUESTION + answer_idf * DEFAULT_WEIGHT_QAP_ANSWER + passage_idf * DEFAULT_WEIGHT_QAP_PASSAGE
        }
    } else if params.eval_has_answer { //QA
        if question_confidence < 1.0 {
            // Reduce question weight, give more to answer
            let q_weight = DEFAULT_WEIGHT_QA_QUESTION * question_confidence;
            let answer_weight = DEFAULT_WEIGHT_QA_ANSWER + DEFAULT_WEIGHT_QA_QUESTION * (1.0 - question_confidence);
            question_idf * q_weight + answer_idf * answer_weight
        } else {
            // Standard weights
            question_idf * DEFAULT_WEIGHT_QA_QUESTION + answer_idf * DEFAULT_WEIGHT_QA_ANSWER
        }
    } else if params.eval_has_passage { //QP
        if question_confidence < 1.0 {
            // Reduce question weight, give more to passage
            let q_weight = DEFAULT_WEIGHT_QP_QUESTION * question_confidence;
            let passage_weight = DEFAULT_WEIGHT_QP_PASSAGE + DEFAULT_WEIGHT_QP_QUESTION * (1.0 - question_confidence);
            question_idf * q_weight + passage_idf * passage_weight
        } else { //Q
            // Standard weights
            question_idf * DEFAULT_WEIGHT_QP_QUESTION + passage_idf * DEFAULT_WEIGHT_QP_PASSAGE
        }
    } else {
        question_idf
    };

    let capped_score = base_score.min(1.0);

    // Apply cumulative length adjustment if configured
    if config.perfect_match_decay_start.is_some() ||
       config.perfect_match_decay_end.is_some() {
        // Calculate cumulative token length
        let question_len = params.eval_token_length.unwrap_or(0);
        let answer_len = if params.eval_has_answer {
            params.answer_token_length.unwrap_or(0)
        } else {
            0
        };
        // Use the passage token length from params
        let passage_len = if params.eval_has_passage {
            params.passage_token_length.unwrap_or(0)
        } else {
            0
        };

        let cumulative_length = question_len + answer_len + passage_len;

        // Get required threshold based on cumulative length
        let required_threshold = interpolate_threshold(
            cumulative_length,
            config.perfect_match_decay_start,
            config.perfect_match_decay_end,
            config.contamination_score_threshold,
        );

        // Apply adjustment: scale the score so that it needs to be higher for shorter cumulative lengths
        // If required_threshold is 1.0, only a perfect score of 1.0 will pass
        // If required_threshold is 0.8, scores >= 0.8 will pass
        // We scale the score to make it harder to reach the threshold for shorter lengths
        if required_threshold > config.contamination_score_threshold {
            // Need to make it harder - scale down scores that aren't perfect
            let scale_factor = config.contamination_score_threshold / required_threshold;
            capped_score * scale_factor + (1.0 - scale_factor) * (capped_score == 1.0) as i32 as f32
        } else {
            capped_score
        }
    } else {
        capped_score
    }
}

/// Helper function to calculate IDF overlap for answer cluster selection
/// Used during answer boundary detection to ensure consistent metric usage
pub fn calculate_answer_cluster_idf_overlap(
    cluster_tokens: &HashSet<usize>,
    reference_tokens: &HashSet<usize>,
    answer_token_idf: &AnswerTokenIdfMap,
) -> f32 {
    // Sort reference tokens for deterministic iteration order
    let mut sorted_reference: Vec<usize> = reference_tokens.iter().copied().collect();
    sorted_reference.sort_unstable();

    // Calculate IDF sum for all reference tokens
    let mut reference_idf_sum = 0.0f32;
    for token in sorted_reference.iter() {
        if let Some(idf) = answer_token_idf.get(token) {
            reference_idf_sum += idf;
        }
    }

    // Sort cluster tokens for deterministic iteration order
    let mut sorted_cluster: Vec<usize> = cluster_tokens.iter().copied().collect();
    sorted_cluster.sort_unstable();

    // Calculate IDF sum for matched tokens only
    let mut matched_idf_sum = 0.0f32;
    for token in sorted_cluster {
        if reference_tokens.contains(&token)
            && let Some(idf) = answer_token_idf.get(&token) {
                matched_idf_sum += idf;
            }
    }

    // Calculate overlap ratio
    if reference_idf_sum > 0.0 {
        matched_idf_sum / reference_idf_sum
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn create_test_ngram_doc_frequencies() -> HashMap<u32, usize> {
        let mut map = HashMap::new();

        // ngram_id 1 appears in 3 docs
        map.insert(1, 3);

        // ngram_id 2 appears in 1 doc
        map.insert(2, 1);

        // ngram_id 3 appears in 2 docs
        map.insert(3, 2);

        map
    }

    #[test]
    fn test_question_idf_sum_calculation() {
        let ngram_doc_frequencies = create_test_ngram_doc_frequencies();
        let total_docs = 10.0;

        // Test with ngrams 1 and 2
        let mut ngram_ids = HashSet::new();
        ngram_ids.insert(1); // appears in 3 docs
        ngram_ids.insert(2); // appears in 1 doc

        // Calculate IDF sum directly using cached frequencies
        let mut idf_sum = 0.0f32;
        let mut sorted_ids: Vec<u32> = ngram_ids.iter().copied().collect();
        sorted_ids.sort_unstable();
        for ngram_id in sorted_ids {
            if let Some(&doc_freq) = ngram_doc_frequencies.get(&ngram_id) {
                let idf = (total_docs / doc_freq as f32).ln();
                idf_sum += idf;
            }
        }

        // IDF for ngram 1: ln(10/3) ≈ 1.204
        // IDF for ngram 2: ln(10/1) ≈ 2.303
        // Sum ≈ 3.507
        assert!((idf_sum - 3.507).abs() < 0.01);
    }

    #[test]
    fn test_question_idf_overlap_empty() {
        let ngram_doc_frequencies = create_test_ngram_doc_frequencies();
        let total_docs = 10.0;
        let ngram_ids = HashSet::new();

        // Empty set should have 0 overlap
        let overlap = calculate_question_idf_overlap(&ngram_ids, 5.0, &ngram_doc_frequencies, total_docs);
        assert!(overlap.is_some());
        assert_eq!(overlap.unwrap(), 0.0);
    }

    #[test]
    fn test_question_idf_overlap_partial() {
        let ngram_doc_frequencies = create_test_ngram_doc_frequencies();
        let total_docs = 10.0;

        // Doc 1 has ngrams 1, 2, 3
        // We'll match ngrams 1 and 2
        let mut matched_ngrams = HashSet::new();
        matched_ngrams.insert(1);
        matched_ngrams.insert(2);

        // Calculate the total IDF for doc 1 (has ngrams 1, 2, 3)
        // IDF: ln(10/3) + ln(10/1) + ln(10/2) ≈ 1.204 + 2.303 + 1.609 = 5.116
        let doc1_total_idf = 5.116f32;

        let overlap = calculate_question_idf_overlap(
            &matched_ngrams,
            doc1_total_idf,
            &ngram_doc_frequencies,
            total_docs,
        );

        // Matched IDF: ln(10/3) + ln(10/1) ≈ 1.204 + 2.303 = 3.507
        // Total IDF: ln(10/3) + ln(10/1) + ln(10/2) ≈ 1.204 + 2.303 + 1.609 = 5.116
        // Ratio: 3.507 / 5.116 ≈ 0.685
        assert!(overlap.is_some());
        let overlap_val = overlap.unwrap();
        assert!((overlap_val - 0.685).abs() < 0.01);
    }

    #[test]
    fn test_question_idf_overlap_no_match() {
        let ngram_doc_frequencies = create_test_ngram_doc_frequencies();
        let total_docs = 10.0;

        let matched_ngrams = HashSet::new(); // No matches

        // Calculate the total IDF for doc 1 (has ngrams 1, 2, 3)
        let doc1_total_idf = 5.116f32;

        let overlap = calculate_question_idf_overlap(
            &matched_ngrams,
            doc1_total_idf,
            &ngram_doc_frequencies,
            total_docs,
        );

        assert!(overlap.is_some());
        assert_eq!(overlap.unwrap(), 0.0);
    }

    #[test]
    fn test_question_idf_overlap_consistency() {
        let ngram_doc_frequencies = create_test_ngram_doc_frequencies();
        let total_docs = 10.0;

        let mut matched_ngrams = HashSet::new();
        matched_ngrams.insert(1);

        // Calculate the total IDF for doc 1 (has ngrams 1, 2, 3)
        let doc1_total_idf = 5.116f32;

        // First call with cached frequencies
        let overlap1 = calculate_question_idf_overlap(
            &matched_ngrams,
            doc1_total_idf,
            &ngram_doc_frequencies,
            total_docs,
        );

        // Second call with same cached frequencies (consistent results)
        let overlap2 = calculate_question_idf_overlap(
            &matched_ngrams,
            doc1_total_idf,
            &ngram_doc_frequencies,
            total_docs,
        );

        assert!(overlap2.is_some());
        assert_eq!(overlap1, overlap2); // Results should be identical
    }

    #[test]
    fn test_calculate_minimum_question_idf_threshold() {
        // Test with default threshold of 0.8
        let threshold = calculate_minimum_question_idf_threshold(0.8);
        // (0.8 - 0.3) / 0.7 = 0.5 / 0.7 = 0.714...
        assert!((threshold - 0.714285).abs() < 0.0001);

        // Test with threshold of 1.0 (perfect match required)
        let threshold_perfect = calculate_minimum_question_idf_threshold(1.0);
        // (1.0 - 0.3) / 0.7 = 0.7 / 0.7 = 1.0
        assert!((threshold_perfect - 1.0).abs() < 0.0001);

        // Test with lower threshold
        let threshold_low = calculate_minimum_question_idf_threshold(0.5);
        // (0.5 - 0.3) / 0.7 = 0.2 / 0.7 = 0.2857...
        assert!((threshold_low - 0.2857).abs() < 0.001);

        // Test edge case where threshold is very low (should not go negative)
        let threshold_edge = calculate_minimum_question_idf_threshold(0.2);
        // (0.2 - 0.3) / 0.7 = -0.1 / 0.7 = -0.142... -> max(0.0)
        assert_eq!(threshold_edge, 0.0);
    }
}
