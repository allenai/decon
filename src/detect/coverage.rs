//! Optional per-input receipts. A partial run's receipts are not a successful scan.

use anyhow::{Context, Result};
use flate2::{Compression, write::GzEncoder};
use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Serialize)]
struct CoverageEntry<'a> {
    schema_version: u8,
    source_file: &'a str,
    training_file: &'a str,
    training_line: usize,
    status: &'a str,
    reason: Option<&'a str>,
    token_count: Option<usize>,
}

pub(crate) struct CoverageWriter {
    stream: BufWriter<GzEncoder<std::fs::File>>,
    source_file: String,
    training_file: String,
}

impl CoverageWriter {
    pub(crate) fn new(
        root: &Path,
        input: &Path,
        training_dir: &Path,
        training_file: &str,
    ) -> Result<Self> {
        let relative = input
            .strip_prefix(training_dir)
            .context("Coverage input must be inside the training directory")?;
        let source_file = relative
            .to_str()
            .context("Coverage requires UTF-8 input paths")?
            .to_owned();
        let name = relative
            .file_name()
            .context("Coverage input has no filename")?
            .to_str()
            .context("Coverage requires UTF-8 filenames")?;
        let path = root
            .join(relative)
            .with_file_name(format!("{name}.coverage.jsonl.gz"));
        fs::create_dir_all(path.parent().context("Coverage output has no parent")?)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("Cannot create coverage output {}", path.display()))?;
        Ok(Self {
            stream: BufWriter::new(GzEncoder::new(file, Compression::fast())),
            source_file,
            training_file: training_file.to_owned(),
        })
    }

    pub(crate) fn record(
        &mut self,
        line: usize,
        reason: Option<&str>,
        tokens: Option<usize>,
    ) -> Result<()> {
        let entry = CoverageEntry {
            schema_version: 1,
            source_file: &self.source_file,
            training_file: &self.training_file,
            training_line: line,
            status: if reason.is_some() {
                "skipped"
            } else {
                "examined"
            },
            reason,
            token_count: tokens,
        };
        serde_json::to_writer(&mut self.stream, &entry)?;
        self.stream.write_all(b"\n")?;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        self.stream.into_inner()?.finish()?.sync_all()?;
        Ok(())
    }
}
