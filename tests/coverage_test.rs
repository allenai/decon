use anyhow::Result;
use decon::detect::config::execute_detect;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

mod common;

fn read_coverage(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(GzDecoder::new(File::open(path)?))
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

#[test]
fn coverage_accounts_for_skips_without_changing_matches() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;
    let input = env.training_dir.join("contaminated_mixed.jsonl");
    let mut file = OpenOptions::new().append(true).open(&input)?;
    if !fs::read(&input)?.ends_with(b"\n") {
        writeln!(file)?;
    }
    for line in [
        r#"{"text":"x"}"#,
        "not json",
        "",
        r#"{"wrong":"field"}"#,
        r#"{"text":17}"#,
        r#"{"text":""}"#,
        r#"{"text":"Whimsical purple penguins invent unrelated numbers beside seventeen green satellites."}"#,
    ] {
        writeln!(file, "{line}")?;
    }
    drop(file);
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.evals_dir = Some(env.reference_dir.clone());
    execute_detect(&args)?;
    let mut before: Vec<String> = common::read_report_files(&env.report_dir)?
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<_, _>>()?;
    let coverage = env.report_dir.with_file_name("coverage");
    args.coverage_output_dir = Some(coverage.clone());
    execute_detect(&args)?;
    let mut after: Vec<String> = common::read_report_files(&env.report_dir)?
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<_, _>>()?;
    before.sort();
    after.sort();
    assert!(!before.is_empty());
    assert_eq!(before, after);
    let rows = read_coverage(&coverage.join("contaminated_mixed.jsonl.coverage.jsonl.gz"))?;
    assert_eq!(rows.len(), 12);
    for (line, row) in rows.iter().enumerate() {
        assert_eq!(row["schema_version"], 1);
        assert_eq!(row["training_line"], line);
        assert_eq!(row["source_file"], "contaminated_mixed.jsonl");
        assert_eq!(row["training_file"], "contaminated_mixed.jsonl");
    }
    let expected = [
        "insufficient_tokens",
        "invalid_json",
        "blank_line",
        "missing_or_nonstring_content",
        "missing_or_nonstring_content",
        "insufficient_tokens",
    ];
    for (row, reason) in rows[5..11].iter().zip(expected) {
        assert_eq!(row["status"], "skipped");
        assert_eq!(row["reason"], reason);
    }
    assert_eq!(rows[11]["status"], "examined");
    assert!(rows[11]["reason"].is_null());
    assert!(rows[11]["token_count"].as_u64().unwrap() >= 10);
    assert!(
        execute_detect(&args).is_err(),
        "Existing coverage cannot be silently mixed with a new run"
    );
    Ok(())
}

#[test]
fn compressed_and_nested_inputs_have_distinct_receipts() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    fs::create_dir(env.training_dir.join("nested"))?;
    let mut gzip = GzEncoder::new(
        File::create(env.training_dir.join("nested/clean.jsonl.gz"))?,
        Compression::fast(),
    );
    writeln!(gzip, "{}", json!({"text":"x"}))?;
    gzip.finish()?;
    let coverage = env.report_dir.with_file_name("coverage");
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.evals_dir = Some(env.reference_dir.clone());
    args.coverage_output_dir = Some(coverage.clone());
    execute_detect(&args)?;
    assert!(coverage.join("clean.jsonl.coverage.jsonl.gz").exists());
    let rows = read_coverage(&coverage.join("nested/clean.jsonl.gz.coverage.jsonl.gz"))?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["source_file"], "nested/clean.jsonl.gz");
    assert_eq!(rows[0]["training_file"], "clean.jsonl");
    assert_eq!(rows[0]["reason"], "insufficient_tokens");
    Ok(())
}

#[test]
fn training_read_failures_propagate() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    fs::write(env.training_dir.join("broken.jsonl.gz"), "not gzip")?;
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.evals_dir = Some(env.reference_dir.clone());
    let error = execute_detect(&args).expect_err("Unreadable input must fail the scan");
    assert!(format!("{error:#}").contains("Error processing training file"));
    Ok(())
}

#[test]
fn match_report_write_failures_propagate() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;
    fs::create_dir(env.training_dir.join("blocked"))?;
    fs::rename(
        env.training_dir.join("contaminated_mixed.jsonl"),
        env.training_dir.join("blocked/contaminated_mixed.jsonl"),
    )?;
    fs::write(env.report_dir.join("blocked"), "not a directory")?;
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.evals_dir = Some(env.reference_dir.clone());
    let error = execute_detect(&args).expect_err("Missing match output must fail the scan");
    assert!(format!("{error:#}").contains("Error saving results"));
    Ok(())
}
