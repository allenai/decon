use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::Result;
use decon::common::Config;
use decon::detect::contamination_detect;
use decon::detect::reference_index::build_simple_index;
use flate2::read::GzDecoder;
use rayon::ThreadPoolBuilder;
use serde_json::{Value, json};
use tempfile::tempdir;

fn section(kind: &str, document: usize) -> String {
    let mut words = vec![format!("{kind}unique{document}")];
    for divisor in 1..=16 {
        if document % divisor == 0 {
            for term in 0..6 {
                words.push(format!("{kind}segment{divisor}word{term}"));
            }
        }
    }
    words.join(" ")
}

fn reports(root: &Path) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        let file = File::open(&path)?;
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|s| s == "gz") {
            Box::new(BufReader::new(GzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        for line in reader.lines() {
            let value: Value = serde_json::from_str(&line?)?;
            result.push(serde_json::to_string(&value)?);
        }
    }
    result.sort();
    Ok(result)
}

#[test]
fn parallel_index_totals_and_report_scores_are_bitwise_reproducible() -> Result<()> {
    let root = tempdir()?;
    let references = root.path().join("references");
    let training = root.path().join("training");
    fs::create_dir_all(&references)?;
    fs::create_dir_all(&training)?;
    let mut train_file = File::create(training.join("input.jsonl"))?;
    for document in 0..64 {
        let question = section("question", document);
        let answer = section("answer", document);
        let passage = section("passage", document);
        // Separate files make the assignment of interned IDs depend on parallel scheduling.
        let mut reference = File::create(references.join(format!("{document:03}.jsonl")))?;
        writeln!(
            reference,
            "{}",
            json!({
                "doc_id": document, "eval_key": "determinism", "eval_instance_index": document,
                "question": question, "answer": answer, "passage": passage,
            })
        )?;
        if document < 8 {
            writeln!(
                train_file,
                "{}",
                json!({
                    "text": format!("{passage} separator boundary {question} continuation {answer}"),
                })
            )?;
        }
    }
    let mut expected_totals = None;
    let mut expected_reports = None;
    for (attempt, threads) in [1, 4, 8, 4].into_iter().enumerate() {
        let output = root.path().join(format!("reports-{attempt}"));
        fs::create_dir(&output)?;
        let config: Config = serde_json::from_value(json!({
            "training_dir": training, "evals_dir": references, "report_output_dir": output,
            "tokenizer_str": "uniseg", "ngram_size": 2, "answer_ngram_size": 2,
            "passage_ngram_size": 2, "eval_min_token_length": 1, "eval_min_unique_word_count": 1,
            "short_answer_token_threshold": 3, "sample_every_m_tokens": 1,
            "contamination_score_threshold": 0.1, "perfect_match_decay_start": null,
            "perfect_match_decay_end": null,
        }))?;
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        let (index, stats) = pool.install(|| build_simple_index(&config))?;
        assert_eq!(stats.lines_indexed, 64);
        let totals: BTreeMap<_, _> = index
            .eval_documents
            .iter()
            .map(|(id, tuple)| {
                (
                    *id,
                    (
                        tuple.2.to_bits(),
                        index.eval_passage_idf_cache[id].to_bits(),
                    ),
                )
            })
            .collect();
        assert_eq!(
            expected_totals.get_or_insert_with(|| totals.clone()),
            &totals
        );
        pool.install(|| contamination_detect(&config))?;
        let current = reports(&output)?;
        assert!(!current.is_empty());
        assert!(current.iter().any(|r| r.contains("answer_idf_overlap")));
        assert!(current.iter().any(|r| r.contains("passage_idf_overlap")));
        // Report ordering is parallel; compare complete records without rounding scores.
        assert_eq!(
            expected_reports.get_or_insert_with(|| current.clone()),
            &current
        );
    }
    Ok(())
}
