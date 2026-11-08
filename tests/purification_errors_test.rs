use anyhow::Result;
use decon::detect::config::execute_detect;
use std::fs;
use std::process::Command;

mod common;

#[test]
fn purification_output_creation_errors_are_returned() -> Result<()> {
    for training_fixture in ["clean.jsonl", "contaminated_mixed.jsonl"] {
        let env = common::setup_test_with_fixtures(
            "passage_question_answer_reference.jsonl",
            training_fixture,
            "default.yaml",
        )?;
        // A directory at the output filename reliably rejects File::create, even as root.
        fs::create_dir(env.cleaned_dir.join(format!("{training_fixture}.gz")))?;
        let mut args = common::create_default_args(env.config_path.clone());
        args.common.purify = true;
        args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
        let error = execute_detect(&args).expect_err("Missing cleaned output must fail detection");
        let message = format!("{error:#}");
        assert!(message.contains("Error purifying file"), "{message}");
        assert!(message.contains(training_fixture), "{message}");
    }
    Ok(())
}

#[test]
fn purification_failure_exits_nonzero() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    fs::create_dir(env.cleaned_dir.join("clean.jsonl.gz"))?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_decon"));
    command
        .env_remove("DECON_TEST")
        .env_remove("DECON_QUIET")
        .args(["detect", "--config"])
        .arg(&env.config_path)
        .args(["--purify", "--cleaned-output-dir"])
        .arg(&env.cleaned_dir);
    let output = command.output()?;
    assert!(
        !output.status.success(),
        "Purification failure must reach the CLI exit status"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error purifying file"), "{stderr}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("Purification Summary:"));

    // A successful run must print the summary, proving quiet mode cannot make
    // the failure-case assertion pass without exercising summary suppression.
    fs::remove_dir(env.cleaned_dir.join("clean.jsonl.gz"))?;
    let success = command.output()?;
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    assert!(String::from_utf8_lossy(&success.stdout).contains("Purification Summary:"));
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn purification_output_write_errors_are_returned() -> Result<()> {
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "clean.jsonl",
        "default.yaml",
    )?;
    // /dev/full opens successfully but rejects writes: exercise more than path creation.
    std::os::unix::fs::symlink("/dev/full", env.cleaned_dir.join("clean.jsonl.gz"))?;
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.purify = true;
    args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
    let error = execute_detect(&args).expect_err("Write failures must fail detection");
    assert!(format!("{error:#}").contains("Error purifying file"));
    Ok(())
}
