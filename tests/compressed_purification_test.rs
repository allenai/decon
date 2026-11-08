use anyhow::Result;
use decon::detect::config::execute_detect;
use flate2::read::GzDecoder;
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, Read, Write};
use std::path::Path;

mod common;

fn compress(bytes: &[u8], extension: &str) -> Result<Vec<u8>> {
    if extension == "bz2" {
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
        encoder.write_all(bytes)?;
        Ok(encoder.finish()?)
    } else {
        let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 1);
        encoder.write_all(bytes)?;
        Ok(encoder.finish()?)
    }
}

fn reports(directory: &Path) -> Result<Vec<String>> {
    let mut records = common::read_report_files(directory)?;
    for record in &mut records {
        // The legacy report name includes bz2/xz suffixes; compare detection content.
        record.as_object_mut().unwrap().remove("training_file");
    }
    let mut records: Vec<_> = records
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<_, _>>()?;
    records.sort();
    Ok(records)
}

fn gunzip(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    GzDecoder::new(File::open(path)?).read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[test]
fn bz2_and_xz_match_plain_input_for_detection_coverage_and_purification() -> Result<()> {
    for extension in ["bz2", "xz"] {
        for concatenated in [false, true] {
            let env = common::setup_test_with_fixtures(
                "passage_question_answer_reference.jsonl",
                "contaminated_mixed.jsonl",
                "default.yaml",
            )?;
            let input = env.training_dir.join("contaminated_mixed.jsonl");
            let contents = fs::read(&input)?;
            let mut args = common::create_default_args(env.config_path.clone());
            args.common.purify = true;
            args.common.cleaned_output_dir = Some(env.cleaned_dir.clone());
            execute_detect(&args)?;
            let expected = reports(&env.report_dir)?;
            assert!(!expected.is_empty());
            let expected_cleaned = gunzip(&env.cleaned_dir.join("contaminated_mixed.jsonl.gz"))?;

            let compressed = if concatenated {
                // A member boundary may fall inside a JSON line.
                [
                    compress(&contents[..7], extension)?,
                    compress(&contents[7..], extension)?,
                ]
                .concat()
            } else {
                compress(&contents, extension)?
            };
            fs::remove_file(&input)?;
            let name = format!("contaminated_mixed.jsonl.{extension}");
            fs::write(env.training_dir.join(&name), compressed)?;
            let report_dir = env.report_dir.with_file_name("compressed-reports");
            let cleaned_dir = env.report_dir.with_file_name("compressed-cleaned");
            let coverage_dir = env.report_dir.with_file_name("coverage");
            args.common.report_output_dir = Some(report_dir.clone());
            args.common.cleaned_output_dir = Some(cleaned_dir.clone());
            args.coverage_output_dir = Some(coverage_dir.clone());
            execute_detect(&args)?;
            assert_eq!(
                reports(&report_dir)?,
                expected,
                "{extension}, concatenated={concatenated}"
            );
            assert_eq!(
                gunzip(&cleaned_dir.join("contaminated_mixed.jsonl.gz"))?,
                expected_cleaned
            );
            let receipts = gunzip(&coverage_dir.join(format!("{name}.coverage.jsonl.gz")))?;
            assert_eq!(receipts.lines().count(), contents.lines().count());
            for (line, receipt) in receipts.lines().enumerate() {
                let row: Value = serde_json::from_str(&receipt?)?;
                assert_eq!(row["training_line"], line);
            }
        }
    }
    Ok(())
}

#[test]
fn truncated_later_bz2_and_xz_members_fail_the_scan() -> Result<()> {
    for extension in ["bz2", "xz"] {
        let env = common::setup_test_with_fixtures(
            "passage_question_answer_reference.jsonl",
            "clean.jsonl",
            "default.yaml",
        )?;
        let mut bytes = compress(b"{\"text\":\"first\"}\n", extension)?;
        let second = compress(b"{\"text\":\"second\"}\n", extension)?;
        bytes.extend_from_slice(&second[..second.len() - 4]);
        fs::write(
            env.training_dir.join(format!("broken.jsonl.{extension}")),
            bytes,
        )?;
        let args = common::create_default_args(env.config_path.clone());
        let error = execute_detect(&args).expect_err("Later-member errors must fail detection");
        assert!(format!("{error:#}").contains("Error processing training file"));
    }
    Ok(())
}
