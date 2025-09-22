use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

use decon::detect::{args::DetectArgs, common_args::CommonDetectionArgs};

pub struct TestEnvironment {
    _temp_dir: TempDir,  // Prefixed with _ to indicate it's kept for Drop cleanup
    pub reference_dir: PathBuf,
    pub training_dir: PathBuf,
    pub report_dir: PathBuf,
    pub cleaned_dir: PathBuf,
    pub config_path: PathBuf,
}

impl TestEnvironment {
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let reference_dir = temp_dir.path().join("reference");
        let training_dir = temp_dir.path().join("training");
        let report_dir = temp_dir.path().join("reports");
        let cleaned_dir = temp_dir.path().join("cleaned");
        let config_path = temp_dir.path().join("config.yaml");

        fs::create_dir_all(&reference_dir)?;
        fs::create_dir_all(&training_dir)?;
        fs::create_dir_all(&report_dir)?;
        fs::create_dir_all(&cleaned_dir)?;

        Ok(TestEnvironment {
            _temp_dir: temp_dir,
            reference_dir,
            training_dir,
            report_dir,
            cleaned_dir,
            config_path,
        })
    }
}

/// Copy a fixture file to the test environment
pub fn copy_fixture(fixture_path: &str, dest_dir: &Path) -> Result<PathBuf> {
    let fixture_full_path = PathBuf::from("tests/fixtures").join(fixture_path);
    
    if !fixture_full_path.exists() {
        anyhow::bail!("Fixture file not found: {:?}", fixture_full_path);
    }
    
    let file_name = fixture_full_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("Invalid fixture path"))?;
    
    let dest_path = dest_dir.join(file_name);
    fs::copy(&fixture_full_path, &dest_path)?;
    
    Ok(dest_path)
}

/// Copy a config fixture and update it with the test directories
pub fn copy_and_update_config(
    config_fixture: &str,
    env: &TestEnvironment,
) -> Result<()> {
    let fixture_path = PathBuf::from("tests/fixtures/configs").join(config_fixture);
    
    if !fixture_path.exists() {
        anyhow::bail!("Config fixture not found: {:?}", fixture_path);
    }
    
    // Read the fixture config
    let config_content = fs::read_to_string(&fixture_path)?;
    let mut config: HashMap<String, serde_yaml::Value> = serde_yaml::from_str(&config_content)?;

    config.insert("evals_dir".to_string(),
        serde_yaml::Value::String(env.reference_dir.to_str().unwrap().to_string()));
    config.insert("training_dir".to_string(),
        serde_yaml::Value::String(env.training_dir.to_str().unwrap().to_string()));
    config.insert("report_output_dir".to_string(),
        serde_yaml::Value::String(env.report_dir.to_str().unwrap().to_string()));
    
    // Add cleaned_output_dir if purify is enabled
    if let Some(purify) = config.get("purify") {
        if purify.as_bool() == Some(true) {
            config.insert("cleaned_output_dir".to_string(), 
                serde_yaml::Value::String(env.cleaned_dir.to_str().unwrap().to_string()));
        }
    }
    
    // Write the updated config
    let yaml_content = serde_yaml::to_string(&config)?;
    let mut file = File::create(&env.config_path)?;
    file.write_all(yaml_content.as_bytes())?;
    
    Ok(())
}

/// Setup test environment with fixtures
pub fn setup_test_with_fixtures(
    reference_fixture: &str,
    training_fixture: &str,
    config_fixture: &str,
) -> Result<TestEnvironment> {
    // Enable quiet mode for tests
    unsafe {
        std::env::set_var("DECON_TEST", "1");
    }
    
    let env = TestEnvironment::new()?;
    
    // Copy fixtures to temp directories
    copy_fixture(&format!("reference/{}", reference_fixture), &env.reference_dir)?;
    copy_fixture(&format!("training/{}", training_fixture), &env.training_dir)?;
    copy_and_update_config(config_fixture, &env)?;
    
    Ok(env)
}

pub fn read_report_files(report_dir: &Path) -> Result<Vec<Value>> {
    let mut reports = Vec::new();
    
    if report_dir.exists() {
        for entry in fs::read_dir(report_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let file = File::open(&path)?;
                let reader = BufReader::new(file);
                let report: Value = serde_json::from_reader(reader)?;
                reports.push(report);
            } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                // Read JSONL file line by line
                let content = fs::read_to_string(&path)?;
                for line in content.lines() {
                    if !line.trim().is_empty() {
                        let report: Value = serde_json::from_str(line)?;
                        reports.push(report);
                    }
                }
            }
        }
    }
    
    Ok(reports)
}

#[allow(dead_code)]
pub fn count_lines_in_file(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let content = fs::read_to_string(path)?;
    Ok(content.lines().count())
}

#[allow(dead_code)]
pub fn assert_contamination_found(reports: &[Value], training_line: usize, eval_line: usize) -> bool {
    for report in reports {
        if let Some(training_line_val) = report.get("training_line") {
            if let Some(eval_line_val) = report.get("eval_line") {
                if training_line_val.as_u64() == Some(training_line as u64) 
                    && eval_line_val.as_u64() == Some(eval_line as u64) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn assert_no_contamination(report_dir: &Path) -> bool {
    let reports = read_report_files(report_dir).unwrap_or_default();
    reports.is_empty()
}

pub fn create_default_args(config_path: PathBuf) -> DetectArgs {
    DetectArgs {
        common: CommonDetectionArgs {
            config: Some(config_path),
            training_dir: None,
            content_key: None,
            evals_dir: None,
            report_output_dir: None,
            cleaned_output_dir: None,
            purify: false,
            ngram_size: None,
            tokenizer: None,
            verbose: false,
            contamination_score_threshold: None,
            sample_every_m_tokens: None,
            question_max_consecutive_misses: None,
            passage_max_consecutive_misses: None,
            passage_ngram_size: None,
            worker_threads: None,
            eval_dedup: false,
            eval_min_token_length: Some(0),  // Force to 0 for tests
            eval_min_unique_word_count: Some(0),  // Force to 0 for tests
            perfect_match_decay_start: None,
            perfect_match_decay_end: None,
            short_answer_token_threshold: None,
            short_answer_window_length: None,
            min_long_answer_window: None,
            answer_ngram_size: None,
            index_passages: None,
            index_answers: None,
        },
    }
}