#![cfg(target_os = "linux")]

use anyhow::Result;
use decon::common::write_purified_file_with_job_id;
use decon::detect::config::execute_detect;
use std::collections::HashSet;
use std::process::Command;

mod common;

#[test]
fn purification_sync_errors_reach_the_cli_and_job_writer() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    // /dev/null accepts writes, flushes, and the gzip trailer, but rejects fsync.
    // This distinguishes filesystem synchronization from encoder finalization.
    std::os::unix::fs::symlink("/dev/null", env.cleaned_dir.join("clean.jsonl.gz"))?;
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.purify = true;
    args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
    let error = execute_detect(&args).expect_err("Synchronization errors must fail detection");
    assert!(format!("{error:#}").contains("Error purifying file"));
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::InvalidInput
    );

    let output = Command::new(env!("CARGO_BIN_EXE_decon"))
        .env_remove("DECON_TEST")
        .env_remove("DECON_QUIET")
        .args(["detect", "--config"])
        .arg(&env.config_path)
        .args(["--purify", "--cleaned-output-dir"])
        .arg(&env.cleaned_dir)
        .output()?;
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Error purifying file"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Purification Summary:"));

    let error = write_purified_file_with_job_id(
        &env.training_dir.join("clean.jsonl"),
        &env.cleaned_dir,
        &HashSet::new(),
        "clean",
    )
    .expect_err("The job writer must also propagate synchronization errors");
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::InvalidInput
    );
    Ok(())
}
