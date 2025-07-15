use anyhow::Result;
use axum::{
    extract::{Path, State},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::{write_purified_file, Config};
use std::collections::HashSet;

// Job submission request
#[derive(Debug, Clone, Deserialize)]
struct JobRequest {
    file_path: PathBuf,
}

// Job submission response
#[derive(Debug, Clone, Serialize)]
struct JobResponse {
    job_id: String,
}

// Job status
#[derive(Debug, Clone, Serialize)]
enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
}

// Job information
#[derive(Debug, Clone)]
struct Job {
    id: String,
    file_path: PathBuf,
    status: JobStatus,
    output_path: Option<PathBuf>,
    purified_path: Option<PathBuf>,
}

// Index types enum to support multiple modes
enum IndexType {
    Simple(crate::simple::SimpleIndex),
    Toxic(crate::toxic::ToxicIndex),
    MinHash(crate::minhash::MinHashIndex),
}

// Shared application state
#[derive(Clone)]
struct AppState {
    job_sender: mpsc::Sender<Job>,
    jobs: Arc<Mutex<std::collections::HashMap<String, Job>>>,
    worker_threads: usize,
    config: Config,
}

pub async fn run_daemon(config: Config, port: u16) -> Result<()> {
    // Configuration already loaded
    println!("Running daemon with mode: {}", config.mode);

    // Initialize index based on mode
    println!("Building index...");
    let index = match config.mode.as_str() {
        "simple" => {
            let index = crate::simple::build_simple_index(&config)?;
            println!("Built simple index with {} unique n-grams", index.0.len());
            Arc::new(IndexType::Simple(index))
        }
        "toxic" => {
            // Load embeddings
            println!("Loading embeddings...");
            let embeddings = crate::toxic::load_embeddings(&config.toxic_embedding_path, &config)?;
            println!("Loaded {} word embeddings", embeddings.len());

            // Generate hyperplanes
            println!(
                "Generating {} random hyperplanes...",
                config.toxic_hyperplanes
            );
            let hyperplanes =
                crate::toxic::generate_hyperplanes(config.toxic_hyperplanes, config.hash_seed)?;

            // Build index
            let (
                toxic_buckets,
                hot_buckets,
                eval_documents,
                eval_vocabulary,
                bucket_contents,
                eval_text_snippets,
            ) = crate::toxic::build_toxic_index(&config, &embeddings, &hyperplanes)?;
            println!("Built toxic index with {} buckets", toxic_buckets.len());

            // Create full index tuple
            let index = (
                toxic_buckets,
                hot_buckets,
                eval_documents,
                eval_vocabulary,
                bucket_contents,
                embeddings,
                hyperplanes,
                eval_text_snippets,
            );
            Arc::new(IndexType::Toxic(index))
        }
        "minhash" => {
            let index = crate::minhash::build_reference_index(&config)?;
            println!(
                "Built MinHash index with {} bands and {} signatures",
                index.0.len(),
                index.1.len()
            );
            Arc::new(IndexType::MinHash(index))
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported mode for daemon: {}",
                config.mode
            ));
        }
    };

    // Create job queue channel
    let (job_sender, job_receiver) = mpsc::channel::<Job>(100);

    // Create shared state
    let state = AppState {
        job_sender,
        jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
        worker_threads: config.worker_threads,
        config: config.clone(),
    };

    // Spawn worker threads based on config
    let num_workers = config.worker_threads;
    println!("Starting {} worker threads", num_workers);

    // Wrap receiver in Arc<Mutex> for sharing between workers
    let job_receiver = Arc::new(Mutex::new(job_receiver));

    for worker_id in 0..num_workers {
        let worker_state = state.clone();
        let worker_config = config.clone();
        let worker_index = index.clone();
        let worker_receiver = job_receiver.clone();

        tokio::spawn(async move {
            worker_loop(
                worker_id,
                worker_state,
                worker_receiver,
                worker_config,
                worker_index,
            )
            .await;
        });
    }

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/submit", post(submit_job))
        .route("/status/:job_id", get(get_job_status))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    println!("Daemon listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "worker_threads": state.worker_threads,
        "report_output_dir": state.config.report_output_dir.to_string_lossy(),
        "cleaned_output_dir": state.config.cleaned_output_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        "purify": state.config.purify
    }))
}

async fn submit_job(
    State(state): State<AppState>,
    Json(request): Json<JobRequest>,
) -> Json<JobResponse> {
    let job_id = Uuid::new_v4().to_string();

    let job = Job {
        id: job_id.clone(),
        file_path: request.file_path,
        status: JobStatus::Pending,
        output_path: None,
        purified_path: None,
    };

    // Store job in tracking map
    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(job_id.clone(), job.clone());
    }

    // Send job to queue
    if let Err(e) = state.job_sender.send(job).await {
        eprintln!("Failed to queue job: {}", e);
        // Update status to failed
        let mut jobs = state.jobs.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Failed("Failed to queue job".to_string());
        }
    }

    Json(JobResponse { job_id })
}

async fn get_job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Json<serde_json::Value> {
    let jobs = state.jobs.lock().await;

    if let Some(job) = jobs.get(&job_id) {
        match &job.status {
            JobStatus::Failed(msg) => Json(json!({
                "job_id": job_id,
                "status": "failed",
                "error": msg
            })),
            _ => {
                let mut response = json!({
                    "job_id": job_id,
                    "status": match &job.status {
                        JobStatus::Pending => "pending",
                        JobStatus::Processing => "processing",
                        JobStatus::Completed => "completed",
                        JobStatus::Failed(_) => unreachable!(),
                    }
                });

                // Add output paths if job is completed
                if let JobStatus::Completed = &job.status {
                    if let Some(output_path) = &job.output_path {
                        response["output_path"] = json!(output_path.to_string_lossy());
                    }
                    if let Some(purified_path) = &job.purified_path {
                        response["purified_path"] = json!(purified_path.to_string_lossy());
                    }
                }

                Json(response)
            }
        }
    } else {
        Json(json!({
            "error": "Job not found"
        }))
    }
}

async fn worker_loop(
    worker_id: usize,
    state: AppState,
    job_receiver: Arc<Mutex<mpsc::Receiver<Job>>>,
    config: Config,
    index: Arc<IndexType>,
) {
    println!("Worker {} started", worker_id);

    loop {
        // Lock receiver and try to get a job
        let job = {
            let mut receiver = job_receiver.lock().await;
            receiver.recv().await
        };

        match job {
            Some(job) => {
                println!(
                    "Worker {} processing job {} for file: {:?}",
                    worker_id, job.id, job.file_path
                );

                // Update status to processing
                {
                    let mut jobs = state.jobs.lock().await;
                    if let Some(stored_job) = jobs.get_mut(&job.id) {
                        stored_job.status = JobStatus::Processing;
                    }
                }

                // Process the file
                let job_id = job.id.clone();
                let file_path = job.file_path.clone();
                let config_clone = config.clone();
                let index_clone = index.clone();
                let result = tokio::task::spawn_blocking(move || {
                    process_single_file(&config_clone, &file_path, &index_clone)
                })
                .await;

                // Update status based on result
                let mut jobs = state.jobs.lock().await;
                if let Some(stored_job) = jobs.get_mut(&job_id) {
                    match result {
                        Ok(Ok((output_path, purified_path))) => {
                            stored_job.status = JobStatus::Completed;
                            stored_job.output_path = output_path;
                            stored_job.purified_path = purified_path;
                            println!("Worker {} completed job {} successfully", worker_id, job_id);
                        }
                        Ok(Err(e)) => {
                            stored_job.status =
                                JobStatus::Failed(format!("Processing error: {}", e));
                            eprintln!("Worker {} - job {} failed: {}", worker_id, job_id, e);
                        }
                        Err(e) => {
                            stored_job.status = JobStatus::Failed(format!("Task error: {}", e));
                            eprintln!("Worker {} - job {} task failed: {}", worker_id, job_id, e);
                        }
                    }
                }
            }
            None => {
                println!("Worker {} shutting down - channel closed", worker_id);
                break;
            }
        }
    }
}

// Process a single file using the pre-built index
fn process_single_file(
    config: &Config,
    file_path: &PathBuf,
    index: &IndexType,
) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    match index {
        IndexType::Simple(simple_index) => {
            let (
                ngram_to_id,
                id_to_question_docs,
                id_to_short_answer,
                eval_documents,
                id_to_ngram_tokens,
                tokenizer,
                eval_text_snippets,
                eval_document_idf_cache,
                doc_to_ngram_ids,
                token_doc_freq,
            ) = simple_index;

            // Use the existing detection logic from simple.rs
            let contamination_results = dashmap::DashMap::new();

            // Calculate total documents once
            let total_docs = eval_documents.len() as f32;

            let lines_processed = crate::simple::process_simple_training_file(
                file_path,
                config,
                ngram_to_id,
                id_to_question_docs,
                id_to_short_answer,
                eval_documents,
                &contamination_results,
                tokenizer,
                id_to_ngram_tokens,
                eval_text_snippets,
                total_docs,
                eval_document_idf_cache,
                doc_to_ngram_ids,
                token_doc_freq,
            )?;

            // Only save results if contamination was found
            let output_path = if !contamination_results.is_empty() {
                let unique_filename = crate::get_unique_results_filename(file_path, config);
                Some(crate::simple::save_contamination_results_toxic_format_with_filename_and_eval_text(
                    config,
                    &contamination_results,
                    Some(&unique_filename),
                    eval_text_snippets
                )?)
            } else {
                println!("No contamination found - skipping report file creation");
                None
            };

            println!("Processed {} lines from {:?}", lines_processed, file_path);

            // Create purified file if requested
            let purified_path = if config.purify {
                let cleaned_dir = config
                    .cleaned_output_dir
                    .as_ref()
                    .unwrap_or(&config.report_output_dir);
                let mut contaminated_lines = HashSet::new();

                // Collect all contaminated line numbers
                for entry in contamination_results.iter() {
                    for contamination in entry.value() {
                        contaminated_lines.insert(contamination.training_line);
                    }
                }

                // Always create a purified file when purify mode is enabled
                println!(
                    "Creating purified file (removing {} contaminated lines)",
                    contaminated_lines.len()
                );
                Some(write_purified_file(
                    file_path,
                    cleaned_dir,
                    &contaminated_lines,
                    config,
                )?)
            } else {
                None
            };

            Ok((output_path, purified_path))
        }
        IndexType::Toxic(toxic_index) => {
            let (
                toxic_buckets,
                hot_buckets,
                eval_documents,
                eval_vocabulary,
                _bucket_contents,
                embeddings,
                hyperplanes,
                eval_text_snippets,
            ) = toxic_index;

            // Use the existing detection logic from toxic.rs
            let contamination_results = dashmap::DashMap::new();

            // Extract filename for toxic processing
            let file_name = file_path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");

            let _stats = crate::toxic::process_toxic_training_file(
                file_path,
                file_name,
                config,
                embeddings,
                hyperplanes,
                toxic_buckets,
                hot_buckets,
                eval_documents,
                eval_vocabulary,
                &contamination_results,
                eval_text_snippets,
            )?;

            // Only save results if contamination was found
            let output_path = if !contamination_results.is_empty() {
                let unique_filename = crate::get_unique_results_filename(file_path, config);
                Some(
                    crate::toxic::save_toxic_contamination_results_with_filename(
                        &contamination_results,
                        &config.report_output_dir,
                        Some(&unique_filename),
                        eval_text_snippets,
                    )?,
                )
            } else {
                println!("No contamination found - skipping report file creation");
                None
            };

            println!("Processed file {:?} using toxic mode", file_path);

            // Create purified file if requested
            let purified_path = if config.purify {
                let cleaned_dir = config
                    .cleaned_output_dir
                    .as_ref()
                    .unwrap_or(&config.report_output_dir);
                let mut contaminated_lines = HashSet::new();

                // Collect all contaminated line numbers
                for entry in contamination_results.iter() {
                    for contamination in entry.value() {
                        contaminated_lines.insert(contamination.training_line);
                    }
                }

                // Always create a purified file when purify mode is enabled
                println!(
                    "Creating purified file (removing {} contaminated lines)",
                    contaminated_lines.len()
                );
                Some(write_purified_file(
                    file_path,
                    cleaned_dir,
                    &contaminated_lines,
                    config,
                )?)
            } else {
                None
            };

            Ok((output_path, purified_path))
        }
        IndexType::MinHash(minhash_index) => {
            let (reference_bands, reference_signatures, eval_text_snippets) = minhash_index;

            // Use the existing detection logic from minhash.rs
            let contamination_results = dashmap::DashMap::new();

            // Set up hashing parameters
            let band_seeds: Vec<u32> = crate::minhash::_expand_band_seeds(
                &vec![config.hash_seed as u32],
                config.num_bands,
            )
            .into_iter()
            .map(|x| x as u32)
            .collect();
            let perm_seeds = crate::minhash::_expand_band_seeds(&band_seeds, config.band_size);

            // Extract filename for minhash processing
            let file_name = file_path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");

            crate::minhash::process_training_file(
                file_path,
                file_name,
                &band_seeds,
                &perm_seeds,
                config.band_size,
                config.ngram_size,
                &config.tokenizer_str,
                &config.content_key,
                reference_bands,
                reference_signatures,
                &contamination_results,
                config.exact_override,
                config.jaccard_similarity_threshold,
                &config.punctuation_chars,
                config,
            )?;

            // Only save results if contamination was found
            let output_path = if !contamination_results.is_empty() {
                let unique_filename = crate::get_unique_results_filename(file_path, config);
                Some(crate::minhash::save_contamination_results_with_filename(
                    &contamination_results,
                    &config.report_output_dir,
                    Some(&unique_filename),
                    eval_text_snippets,
                )?)
            } else {
                println!("No contamination found - skipping report file creation");
                None
            };

            println!("Processed file {:?} using minhash mode", file_path);

            // Create purified file if requested
            let purified_path = if config.purify {
                let cleaned_dir = config
                    .cleaned_output_dir
                    .as_ref()
                    .unwrap_or(&config.report_output_dir);
                let mut contaminated_lines = HashSet::new();

                // Collect all contaminated line numbers
                for entry in contamination_results.iter() {
                    for (line_num, _, _, _, _) in entry.value() {
                        contaminated_lines.insert(*line_num);
                    }
                }

                // Always create a purified file when purify mode is enabled
                println!(
                    "Creating purified file (removing {} contaminated lines)",
                    contaminated_lines.len()
                );
                Some(write_purified_file(
                    file_path,
                    cleaned_dir,
                    &contaminated_lines,
                    config,
                )?)
            } else {
                None
            };

            Ok((output_path, purified_path))
        }
    }
}
