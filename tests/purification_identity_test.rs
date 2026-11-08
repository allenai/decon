use anyhow::Result;
use decon::detect::config::execute_detect;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

mod common;

fn gzip(path: &Path, contents: &str) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let mut encoder = GzEncoder::new(File::create(path)?, Compression::fast());
    encoder.write_all(contents.as_bytes())?;
    encoder.finish()?;
    Ok(())
}

fn gunzip(path: &Path) -> Result<String> {
    let mut contents = String::new();
    GzDecoder::new(File::open(path)?).read_to_string(&mut contents)?;
    Ok(contents)
}

#[test]
fn same_basename_files_keep_their_own_contaminated_lines() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "contaminated_mixed.jsonl",
        "default.yaml",
    )?;
    let source = env.training_dir.join("contaminated_mixed.jsonl");
    let contents = fs::read_to_string(&source)?;
    let rows: Vec<_> = contents.lines().collect();
    let contaminated = rows[0];
    let clean = rows[2];
    fs::remove_file(source)?;

    // Distinct contaminated line numbers in two shards, plus a wholly clean shard.
    let inputs = [
        ("a", format!("{contaminated}\n{clean}\n")),
        ("b", format!("{clean}\n{contaminated}\n")),
        ("c", format!("{clean}\n{clean}\n")),
    ];
    for (directory, contents) in &inputs {
        gzip(
            &env.training_dir.join(directory).join("part-0.jsonl.gz"),
            contents,
        )?;
    }
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.purify = true;
    args.common.worker_threads = Some(4);
    args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
    execute_detect(&args)?;

    for (directory, contaminated_line) in [("a", 0), ("b", 1)] {
        let records = common::read_report_files(&env.report_dir.join(directory))?;
        assert!(!records.is_empty());
        assert!(
            records
                .iter()
                .all(|record| record["training_line"] == contaminated_line)
        );
        assert_eq!(
            gunzip(&env.cleaned_dir.join(directory).join("part-0.jsonl.gz"))?,
            format!("{clean}\n")
        );
    }
    assert_eq!(
        gunzip(&env.cleaned_dir.join("c/part-0.jsonl.gz"))?,
        inputs[2].1
    );
    Ok(())
}

#[test]
fn colliding_output_names_fail_before_training_files_are_read() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    let plain = env.training_dir.join("clean.jsonl");
    let compressed = env.training_dir.join("clean.jsonl.gz");
    // Invalid gzip proves that collision validation precedes training input reads.
    fs::write(&compressed, "not a gzip stream")?;
    fs::write(
        env.training_dir.join("unrelated.jsonl"),
        "{\"text\":\"unrelated\"}\n",
    )?;
    let destination = env.cleaned_dir.join("clean.jsonl.gz");
    fs::write(&destination, "previous output")?;
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.purify = true;
    args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
    let coverage = env.report_dir.with_file_name("coverage");
    args.coverage_output_dir = Some(coverage.clone());
    let error = execute_detect(&args).expect_err("Colliding destinations must fail purification");
    let message = format!("{error:#}");
    assert!(message.contains("Purified output collision"), "{message}");
    for path in [&plain, &compressed, &destination] {
        assert!(message.contains(path.to_str().unwrap()), "{message}");
    }
    assert_eq!(fs::read_to_string(destination)?, "previous output");
    assert!(!env.cleaned_dir.join("unrelated.jsonl.gz").exists());
    assert!(fs::read_dir(&env.report_dir)?.next().is_none());
    assert!(!coverage.exists() || fs::read_dir(&coverage)?.next().is_none());

    // Output-name collisions are irrelevant when purification is disabled.
    gzip(&compressed, &fs::read_to_string(&plain)?)?;
    args.common.purify = false;
    args.coverage_output_dir = None;
    execute_detect(&args)?;
    Ok(())
}
