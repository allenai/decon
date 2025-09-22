use anyhow::Result;

// Import from the decon crate
use decon::detect::config::execute_detect;

// Use the shared test utilities
mod common;

#[test]
fn test_sampling_hits_contamination() -> Result<()> {
    // This test verifies that with appropriate sampling intervals,
    // contamination is detected in a long document
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "long_document.jsonl",
        "default.yaml",  // Use default config
    )?;
    
    // Run detection with sampling that will hit the contamination
    // The contamination questions appear at positions 219 and 387
    // Use position 219 where "What year was the Eiffel" starts
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.sample_every_m_tokens = Some(219);  // Sample at position 0, 219, 438...
    args.common.question_max_consecutive_misses = Some(5);   // Allow expansion
    execute_detect(&args)?;
    
    // Check that contamination was found
    let reports = common::read_report_files(&env.report_dir)?;
    
    // We should find contamination in the long document (training line 0)
    // It could match either the Eiffel Tower (eval line 0) or Oxygen (eval line 1)
    assert!(
        !reports.is_empty(),
        "Should detect contamination when sampling interval hits contaminated regions"
    );
    
    // Verify we found contamination in the long document
    let found_in_long_doc = reports.iter().any(|r| 
        r.get("training_line").and_then(|v| v.as_u64()) == Some(0)
    );
    assert!(
        found_in_long_doc,
        "Should find contamination in the long document (training line 0)"
    );
    
    Ok(())
}

#[test]
fn test_sampling_misses_contamination() -> Result<()> {
    // This test verifies that with too-large sampling intervals,
    // contamination can be missed entirely
    let env = common::setup_test_with_fixtures(
        "passage_question_answer_reference.jsonl",
        "long_document.jsonl",
        "default.yaml",  // Use default config
    )?;
    
    // Run detection with sampling that will skip the entire document
    let mut args = common::create_default_args(env.config_path.clone());
    args.common.sample_every_m_tokens = Some(600);  // Sample every 600 tokens (doc is ~500)
    args.common.question_max_consecutive_misses = Some(1);   // Only allow 1 miss
    execute_detect(&args)?;
    
    // Check that no contamination was found (sampling skipped it)
    assert!(
        common::assert_no_contamination(&env.report_dir),
        "Should not detect contamination when sampling interval is larger than document"
    );
    
    Ok(())
}

#[test]
fn test_sampling_with_different_intervals() -> Result<()> {
    // Test that smaller sampling intervals find more contamination instances
    
    // First test with moderate sampling (every 100 tokens)
    let env_moderate = common::TestEnvironment::new()?;
    common::copy_fixture("reference/passage_question_answer_reference.jsonl", &env_moderate.reference_dir)?;
    common::copy_fixture("training/long_document.jsonl", &env_moderate.training_dir)?;
    
    // Create a custom config with 100-token sampling
    let config_content = r#"
mode: simple
content_key: text
tokenizer_str: word
ngram_size: 5
question_threshold: 0.8
answer_threshold: 0.8
passage_threshold: 0.8
sample_every_m_tokens: 100
question_max_consecutive_misses: 3
"#;
    // Write initial config content
    std::fs::write(&env_moderate.config_path, config_content)?;
    
    // Manually add the paths since we're not using a fixture config
    let mut config: std::collections::HashMap<String, serde_yaml::Value> = serde_yaml::from_str(config_content)?;
    config.insert("evals_dir".to_string(),
        serde_yaml::Value::String(env_moderate.reference_dir.to_str().unwrap().to_string()));
    config.insert("reference_dir".to_string(),
        serde_yaml::Value::String(env_moderate.reference_dir.to_str().unwrap().to_string()));
    config.insert("training_dir".to_string(),
        serde_yaml::Value::String(env_moderate.training_dir.to_str().unwrap().to_string()));
    config.insert("report_output_dir".to_string(),
        serde_yaml::Value::String(env_moderate.report_dir.to_str().unwrap().to_string()));
    let yaml_content = serde_yaml::to_string(&config)?;
    std::fs::write(&env_moderate.config_path, yaml_content)?;
    
    let args_moderate = common::create_default_args(env_moderate.config_path.clone());
    execute_detect(&args_moderate)?;
    let moderate_reports = common::read_report_files(&env_moderate.report_dir)?;
    
    // Now test with dense sampling (every 25 tokens)
    let env_dense = common::TestEnvironment::new()?;
    common::copy_fixture("reference/passage_question_answer_reference.jsonl", &env_dense.reference_dir)?;
    common::copy_fixture("training/long_document.jsonl", &env_dense.training_dir)?;
    
    let dense_config = r#"
mode: simple
content_key: text
tokenizer_str: word
ngram_size: 5
question_threshold: 0.8
answer_threshold: 0.8
passage_threshold: 0.8
sample_every_m_tokens: 25
question_max_consecutive_misses: 10
"#;
    // Write initial config content  
    std::fs::write(&env_dense.config_path, dense_config)?;
    
    // Manually add the paths since we're not using a fixture config
    let mut config: std::collections::HashMap<String, serde_yaml::Value> = serde_yaml::from_str(dense_config)?;
    config.insert("evals_dir".to_string(),
        serde_yaml::Value::String(env_dense.reference_dir.to_str().unwrap().to_string()));
    config.insert("reference_dir".to_string(),
        serde_yaml::Value::String(env_dense.reference_dir.to_str().unwrap().to_string()));
    config.insert("training_dir".to_string(),
        serde_yaml::Value::String(env_dense.training_dir.to_str().unwrap().to_string()));
    config.insert("report_output_dir".to_string(),
        serde_yaml::Value::String(env_dense.report_dir.to_str().unwrap().to_string()));
    let yaml_content = serde_yaml::to_string(&config)?;
    std::fs::write(&env_dense.config_path, yaml_content)?;
    
    let args_dense = common::create_default_args(env_dense.config_path.clone());
    execute_detect(&args_dense)?;
    let dense_reports = common::read_report_files(&env_dense.report_dir)?;
    
    // Dense sampling should find at least as much contamination as moderate
    assert!(
        dense_reports.len() >= moderate_reports.len(),
        "Denser sampling should detect at least as much contamination as moderate sampling"
    );
    
    Ok(())
}