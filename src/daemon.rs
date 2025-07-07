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

// Index types enum to support multiple modes
enum IndexType {
    Simple(crate::simple::SimpleIndex),
    Toxic(crate::toxic::ToxicIndex),
}

// Shared application state
#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    job_sender: mpsc::Sender<Job>,
    jobs: Arc<Mutex<std::collections::HashMap<String, Job>>>,
    index: Arc<IndexType>,
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
            Arc::new(IndexType::Simple(index))
        }
        "toxic" => {
            // Load embeddings
            println!("Loading embeddings...");
            let embeddings = crate::toxic::load_embeddings(&config.toxic_embedding_path, &config)?;
            println!("Loaded {} word embeddings", embeddings.len());
            
            // Generate hyperplanes
            println!("Generating {} random hyperplanes...", config.toxic_hyperplanes);
            let hyperplanes = crate::toxic::generate_hyperplanes(config.toxic_hyperplanes, config.hash_seed)?;
            
            // Build index
            let (toxic_buckets, hot_buckets, eval_documents, eval_vocabulary, bucket_contents) = 
                crate::toxic::build_toxic_index(&config, &embeddings, &hyperplanes)?;
            println!("Built toxic index with {} buckets", toxic_buckets.len());
            
            // Create full index tuple
            let index = (toxic_buckets, hot_buckets, eval_documents, eval_vocabulary, bucket_contents, embeddings, hyperplanes);
            Arc::new(IndexType::Toxic(index))
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported mode for daemon: {}", config.mode));
        }
    };
    
    // Create job queue channel
    let (job_sender, job_receiver) = mpsc::channel::<Job>(100);
    
    // Create shared state
    let state = AppState {
        config: Arc::new(config.clone()),
        job_sender,
        jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
        index: index.clone(),
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
            worker_loop(worker_id, worker_state, worker_receiver, worker_config, worker_index).await;
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
                println!("Worker {} processing job {} for file: {:?}", 
                         worker_id, job.id, job.file_path);
                
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
                            println!("Worker {} completed job {} successfully", worker_id, job_id);
                        }
                        Ok(Err(e)) => {
                            stored_job.status = JobStatus::Failed(format!("Processing error: {}", e));
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
) -> Result<()> {
    match index {
        IndexType::Simple(simple_index) => {
            let (ngram_to_id, id_to_docs, eval_documents, id_to_ngram_tokens, tokenizer) = simple_index;
            
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
        }
        IndexType::Toxic(toxic_index) => {
            let (toxic_buckets, hot_buckets, eval_documents, eval_vocabulary, _bucket_contents, embeddings, hyperplanes) = toxic_index;
            
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
            )?;
            
            // Save results
            crate::toxic::save_toxic_contamination_results(&contamination_results, &config.output_dir)?;
            
            println!("Processed file {:?} using toxic mode", file_path);
        }
    }
    
    Ok(())
}