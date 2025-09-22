use anyhow::Result;
use axum::{
    extract::{Path, State},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::common::{write_purified_file_with_job_id, Config};
use crate::detect::ContaminationResults;

// Job submission request
#[derive(Debug, Clone, Deserialize)]
struct JobRequest {
    file_path: PathBuf,
}

// Job submission response. Submitters can use to poll for status
#[derive(Debug, Clone, Serialize)]
struct JobResponse {
    job_id: String,
}

#[derive(Debug, Clone, Serialize)]
enum JobStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
}

// Submit a job id and get a status and output file paths, e.g. done?
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
    Simple(crate::detect::reference_index::SimpleReferenceIndex),
}

// Shared application state
#[derive(Clone)]
struct AppState {
    job_sender: mpsc::Sender<Job>,
    jobs: Arc<Mutex<std::collections::HashMap<String, Job>>>,
    worker_threads: usize,
    config: Config,
}

pub async fn run_server(mut config: Config, port: u16) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.worker_threads)
        .build_global()
        .unwrap_or_else(|e| {
            eprintln!("Warning: Failed to set Rayon thread pool size: {}", e);
        });

    // If purify is enabled but cleaned_output_dir is not set, generate a temp directory
    if config.purify && config.cleaned_output_dir.is_none() {
        config.cleaned_output_dir = Some(crate::common::generate_temp_dir("decon-cleaned"));
    }

    // Configuration already loaded
    println!("Running server with mode: {}", config.mode);
    println!("Using {} worker threads", config.worker_threads);

    // Always announce the report output directory at the start
    let report_dir_str = config.report_output_dir.display().to_string();
    print!("Contamination report output directory: {}", report_dir_str);

    // Check if this is an auto-generated temp directory and add hint
    if report_dir_str.contains("/tmp/decon-") || report_dir_str.contains("\\decon-") {
        println!(" (set report directory with --report-output-dir)");
    } else {
        println!();
    }

    // If purify is enabled, announce the cleaned output directory
    if config.purify
        && let Some(ref cleaned_dir) = config.cleaned_output_dir {
            let cleaned_dir_str = cleaned_dir.display().to_string();
            print!("Cleaned output directory: {}", cleaned_dir_str);

            // Check if this is an auto-generated temp directory and add hint
            if cleaned_dir_str.contains("/tmp/decon-cleaned-") || cleaned_dir_str.contains("\\decon-cleaned-") {
                println!(" (set cleaned directory with --cleaned-output-dir)");
            } else {
                println!();
            }
        }

    // Initialize index based on mode
    println!("Building index...");
    let index = match config.mode.as_str() {
        "simple" => {
            let (index, index_stats) = crate::detect::reference_index::build_simple_index(&config)?;

            // Display the index building results in a table
            let index_time = std::time::Duration::from_secs(0); // Server doesn't track build time
            crate::detect::display::display_index_building_results(&index_stats, &index, index_time);

            Arc::new(IndexType::Simple(index))
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported mode for server: {}",
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
    println!("Server listening on http://{}", addr);

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
        "purify": state.config.purify,
        "mode": state.config.mode,
        "contamination_score_threshold": state.config.contamination_score_threshold
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
                let job_id_for_processing = job_id.clone();
                let result = tokio::task::spawn_blocking(move || {
                    process_single_file(
                        &config_clone,
                        &file_path,
                        &index_clone,
                        &job_id_for_processing,
                    )
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
    job_id: &str,
) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    match index {
        IndexType::Simple(simple_index) => {

            // Use the existing detection logic from simple.rs
            let contamination_results: ContaminationResults = dashmap::DashMap::new();

            // Create a stats container for this processing (optional, based on config.verbose)
            let stats = if config.verbose {
                Some(crate::detect::stats::StatsContainer::new())
            } else {
                None
            };

            let lines_processed = crate::detect::detection::process_simple_training_file(
                file_path,
                config,
                simple_index,
                &contamination_results,
                stats.as_ref(),
            )?;

            // Only save results if contamination was found
            let output_path = if !contamination_results.is_empty() {
                // Use job_id for filename
                let unique_filename = format!("{}.report.jsonl", job_id);
                Some(crate::detect::reporting::save_contamination_results(
                    config,
                    &contamination_results,
                    &unique_filename,
                    &simple_index.eval_text_snippets,
                    &simple_index.eval_answer_text_snippets,
                    &simple_index.eval_passage_text_snippets
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
                Some(write_purified_file_with_job_id(
                    file_path,
                    cleaned_dir,
                    &contaminated_lines,
                    job_id,
                )?)
            } else {
                None
            };

            Ok((output_path, purified_path))
        }
    }
}
