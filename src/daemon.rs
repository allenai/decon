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

use crate::{Config, read_config};

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
}

// Shared application state
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    job_sender: mpsc::Sender<Job>,
    jobs: Arc<Mutex<std::collections::HashMap<String, Job>>>,
    index: Arc<crate::simple::SimpleIndex>,
}

pub async fn run_daemon(config_path: PathBuf, port: u16) -> Result<()> {
    // Load configuration
    let config = read_config(&config_path)?;
    println!("Loaded config with mode: {}", config.mode);
    
    // Initialize index based on mode
    println!("Building index...");
    let index = match config.mode.as_str() {
        "simple" => {
            let index = crate::simple::build_simple_index(&config)?;
            println!("Built simple index with {} unique n-grams", index.0.len());
            Arc::new(index)
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported mode for daemon: {}", config.mode));
        }
    };
    
    // Create job queue channel
    let (job_sender, mut job_receiver) = mpsc::channel::<Job>(100);
    
    // Create shared state
    let state = AppState {
        config: Arc::new(config.clone()),
        job_sender,
        jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
        index: index.clone(),
    };
    
    // Spawn worker thread to process jobs
    let worker_state = state.clone();
    tokio::spawn(async move {
        worker_loop(worker_state, &mut job_receiver, config, index).await;
    });
    
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

async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok"
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
            _ => Json(json!({
                "job_id": job_id,
                "status": match &job.status {
                    JobStatus::Pending => "pending",
                    JobStatus::Processing => "processing",
                    JobStatus::Completed => "completed",
                    JobStatus::Failed(_) => unreachable!(),
                }
            }))
        }
    } else {
        Json(json!({
            "error": "Job not found"
        }))
    }
}

async fn worker_loop(
    state: AppState,
    job_receiver: &mut mpsc::Receiver<Job>,
    config: Config,
    index: Arc<crate::simple::SimpleIndex>,
) {
    println!("Worker thread started");
    
    while let Some(job) = job_receiver.recv().await {
        println!("Processing job {} for file: {:?}", job.id, job.file_path);
        
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
        }).await;
        
        // Update status based on result
        let mut jobs = state.jobs.lock().await;
        if let Some(stored_job) = jobs.get_mut(&job_id) {
            match result {
                Ok(Ok(_)) => {
                    stored_job.status = JobStatus::Completed;
                    println!("Job {} completed successfully", job_id);
                }
                Ok(Err(e)) => {
                    stored_job.status = JobStatus::Failed(format!("Processing error: {}", e));
                    eprintln!("Job {} failed: {}", job_id, e);
                }
                Err(e) => {
                    stored_job.status = JobStatus::Failed(format!("Task error: {}", e));
                    eprintln!("Job {} task failed: {}", job_id, e);
                }
            }
        }
    }
    
    println!("Worker thread shutting down");
}

// Process a single file using the pre-built index
fn process_single_file(
    config: &Config,
    file_path: &PathBuf,
    index: &crate::simple::SimpleIndex,
) -> Result<()> {
    let (ngram_to_id, id_to_docs, eval_documents, id_to_ngram_tokens, tokenizer) = index;
    
    // Use the existing detection logic from simple.rs
    let contamination_results = dashmap::DashMap::new();
    
    let lines_processed = crate::simple::process_simple_training_file(
        file_path,
        config,
        ngram_to_id,
        id_to_docs,
        eval_documents,
        &contamination_results,
        tokenizer,
        id_to_ngram_tokens,
    )?;
    
    // Save results
    crate::simple::save_contamination_results_toxic_format(config, &contamination_results)?;
    
    println!("Processed {} lines from {:?}", lines_processed, file_path);
    
    Ok(())
}