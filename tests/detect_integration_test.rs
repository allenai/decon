use anyhow::Result;

// Import from the decon crate
use decon::detect::config::execute_detect;

// Use the shared test utilities
mod common;

#[test]
fn test_basic_contamination_detection() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;

    // Run detection
    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    // Check results
    let reports = common::read_report_files(&env.report_dir)?;
    assert!(!reports.is_empty(), "Should have contamination reports");

    // We're expecting at least 2 contaminations
    assert!(reports.len() >= 2, "Should detect at least 2 contaminations");

    // Verify Entry 1 contamination (training line 0, eval line 0)
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should detect contamination from Entry 1 (eval line 0) at training line 0"
    );

    // Verify Entry 2 contamination (training line 3, eval line 1)
    assert!(
        common::assert_contamination_found(&reports, 3, 1),
        "Should detect contamination from Entry 2 (eval line 1) at training line 3"
    );

    Ok(())
}

#[test]
fn test_no_contamination() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;

    // Run detection
    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    // Check that no contamination was found
    assert!(
        common::assert_no_contamination(&env.report_dir),
        "Should not detect any contamination in clean data"
    );

    Ok(())
}

#[test]
fn test_purification() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;

    // Get path to training file for later verification
    let training_path = env.training_dir.join("contaminated_mixed.jsonl");

    // Run detection with purification
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.purify = true;  // Override to enable purification
    args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());  // Set cleaned output dir
    execute_detect(&args)?;

    // Check that cleaned file was created (it will be gzipped)
    let cleaned_file = env.cleaned_dir.join("contaminated_mixed.jsonl.gz");
    assert!(cleaned_file.exists(), "Cleaned file should be created");

    // Count lines in the original file
    let original_lines = common::count_lines_in_file(&training_path)?;
    assert_eq!(original_lines, 5, "Original should have 5 lines");

    // Count lines in the cleaned gzipped file
    use flate2::read::GzDecoder;
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(&cleaned_file)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    let cleaned_lines = reader.lines().count();

    assert!(cleaned_lines <= 3, "Cleaned should have at most 3 lines (removed at least 2 contaminated)");

    Ok(())
}

#[test]
fn test_threshold_variations() -> Result<()> {
    // Test with strict thresholds (0.95)
    let env_strict = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_edge_cases.jsonl",
        "strict.yaml",
    )?;

    let args_strict = common::create_default_args(env_strict.config_path.clone());
    execute_detect(&args_strict)?;

    let strict_reports = common::read_report_files(&env_strict.report_dir)?;
    let strict_count = strict_reports.len();

    // Test with lenient thresholds (0.5)
    let env_lenient = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_edge_cases.jsonl",
        "lenient.yaml",
    )?;

    let args_lenient = common::create_default_args(env_lenient.config_path.clone());
    execute_detect(&args_lenient)?;

    let lenient_reports = common::read_report_files(&env_lenient.report_dir)?;
    let lenient_count = lenient_reports.len();

    // Lenient thresholds should detect more or equal contamination
    assert!(
        lenient_count >= strict_count,
        "Lenient thresholds should detect at least as much contamination as strict"
    );

    Ok(())
}

#[test]
fn test_3gram_size_variation() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;

    let mut args = common::create_default_args(env.config_path.clone());
    args.common.ngram_size = Some(3);  // Override ngram_size to 3
    execute_detect(&args)?;

    let reports_3gram = common::read_report_files(&env.report_dir)?;
    assert!(!reports_3gram.is_empty(), "Should detect contamination with 3-grams");

    Ok(())
}

#[test]
fn test_match_in_middle_of_document() -> Result<()> {
    // Test that contamination is detected when it appears in the middle of a longer document
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_middle_position.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;

    // Should detect the Eiffel Tower contamination embedded in the Lorem ipsum text
    assert!(
        !reports.is_empty(),
        "Should detect contamination even when embedded in middle of document"
    );

    // Verify it found the contamination on line 0 (the only line in our fixture)
    assert!(
        reports.iter().any(|r| r.get("training_line").and_then(|v| v.as_u64()) == Some(0)),
        "Should find contamination on line 0 of the training document"
    );

    // Verify it matched the Eiffel Tower reference (eval line 0)
    assert!(
        reports.iter().any(|r| r.get("eval_line").and_then(|v| v.as_u64()) == Some(0)),
        "Should match the Eiffel Tower reference (eval line 0)"
    );

    Ok(())
}

#[test]
fn test_match_at_beginning_of_document() -> Result<()> {
    // Test that contamination is detected when it appears at the beginning of a document
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_at_beginning.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(
        !reports.is_empty(),
        "Should detect contamination at beginning of document"
    );

    // Verify it found the contamination
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should find contamination at beginning of training document"
    );

    Ok(())
}

#[test]
fn test_match_at_end_of_document() -> Result<()> {
    // Test that contamination is detected when it appears at the end of a document
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_at_end.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(
        !reports.is_empty(),
        "Should detect contamination at end of document"
    );

    // Verify it found the contamination
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should find contamination at end of training document"
    );

    Ok(())
}

#[test]
fn test_multiple_matches_in_document() -> Result<()> {
    // Test detection of multiple contaminations within a single document
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_multiple_in_one.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    
    // Should detect multiple contaminations from the same document
    assert!(
        reports.len() >= 2,
        "Should detect multiple contaminations in same document"
    );

    // All should be from training line 0 but matching different eval lines
    let line_0_reports: Vec<_> = reports
        .iter()
        .filter(|r| r.get("training_line").and_then(|v| v.as_u64()) == Some(0))
        .collect();
    
    assert!(
        line_0_reports.len() >= 2,
        "Should have multiple matches from the same training document"
    );

    Ok(())
}

#[test]
fn test_punctuation_cleaning() -> Result<()> {
    // Test that punctuation differences don't prevent contamination detection
    let env = common::setup_test_with_fixtures(
        "question_reference.jsonl",
        "contaminated_punctuation_variation.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(
        !reports.is_empty(),
        "Should detect contamination despite punctuation differences"
    );

    Ok(())
}

#[test]
fn test_single_edit_tolerance() -> Result<()> {
    // Test detection with single character substitution
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_single_edit.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    
    // This test verifies the tolerance level - may or may not detect depending on config
    // Just verify the test runs without error
    println!("Single edit tolerance test completed with {} reports", reports.len());

    Ok(())
}

#[test]
fn test_single_insertion_tolerance() -> Result<()> {
    // Test detection with single character insertion
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_single_insertion.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    
    // This test verifies the tolerance level - may or may not detect depending on config
    println!("Single insertion tolerance test completed with {} reports", reports.len());

    Ok(())
}

#[test]
fn test_single_deletion_tolerance() -> Result<()> {
    // Test detection with single character deletion
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_single_deletion.jsonl",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    
    // This test verifies the tolerance level - may or may not detect depending on config
    println!("Single deletion tolerance test completed with {} reports", reports.len());

    Ok(())
}

// Note: Sampling tests have been moved to sampling_test.rs for better control
// and deterministic testing with long documents

#[test]
fn test_gzip_compressed_files() -> Result<()> {
    // Test detection with gzip compressed training data
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl.gz",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(!reports.is_empty(), "Should detect contamination in gzip file");

    // Should find same contaminations as uncompressed
    assert!(reports.len() >= 2, "Should detect at least 2 contaminations");
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should detect contamination at training line 0, eval line 0"
    );

    Ok(())
}

#[test]
fn test_zstd_compressed_files() -> Result<()> {
    // Test detection with zstd compressed training data
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl.zst",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(!reports.is_empty(), "Should detect contamination in zstd file");

    // Should find same contaminations as uncompressed
    assert!(reports.len() >= 2, "Should detect at least 2 contaminations");
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should detect contamination at training line 0, eval line 0"
    );

    Ok(())
}

#[test]
fn test_bz2_compressed_files() -> Result<()> {
    // Test detection with bzip2 compressed training data
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl.bz2",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(!reports.is_empty(), "Should detect contamination in bz2 file");

    // Should find same contaminations as uncompressed
    assert!(reports.len() >= 2, "Should detect at least 2 contaminations");
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should detect contamination at training line 0, eval line 0"
    );

    Ok(())
}

#[test]
fn test_xz_compressed_files() -> Result<()> {
    // Test detection with xz compressed training data
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl.xz",
        "default.yaml",
    )?;

    let args = common::create_default_args(env.config_path.clone());
    execute_detect(&args)?;

    let reports = common::read_report_files(&env.report_dir)?;
    assert!(!reports.is_empty(), "Should detect contamination in xz file");

    // Should find same contaminations as uncompressed
    assert!(reports.len() >= 2, "Should detect at least 2 contaminations");
    assert!(
        common::assert_contamination_found(&reports, 0, 0),
        "Should detect contamination at training line 0, eval line 0"
    );

    Ok(())
}
