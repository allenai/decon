.PHONY: help detect review stats simple evals embeddings

help:
	@echo "Available targets:"
	@echo "  detect     - Run contamination detection with dev config"
	@echo "  simple     - Run simple contamination detection with dev config"
	@echo "  review     - Run review mode with dev config and step-by-step output (use --full for complete training docs)"
	@echo "  stats      - Run review mode with dev config and statistics output"
	@echo "  tp         - Run review mode to show True Positives"
	@echo "  tn         - Run review mode to show True Negatives"
	@echo "  fp         - Run review mode to show False Positives"
	@echo "  fn         - Run review mode to show False Negatives"
	@echo "  evals      - Download evaluation datasets using Python script"
	@echo "  embeddings - Download and prepare word embeddings using Python script"
	@echo "  daemon     - Start the daemon server on port 8080"
	@echo "  health     - Check daemon health status"
	@echo "  submit     - Submit a file for processing (usage: make submit FILE=<path>)"
	@echo "  status     - Check job status (usage: make status JOB_ID=<id>)"

detect:
	cargo run --release detect --config examples/eval/simple.yaml

review:
	cargo run --release review --config examples/eval/simple.yaml --step

minhash:
	cargo run --release detect --config examples/eval/minhash.yaml

toxic:
	cargo run --release detect --config examples/eval/toxic.yaml

simple:
	cargo run --release detect --config examples/eval/simple.yaml

stats:
	cargo run -- review --config examples/eval/simple.yaml --stats

fn:
	cargo run -- review --config examples/eval/simple.yaml --fn

fp:
	cargo run -- review --config examples/eval/simple.yaml --fp

tp:
	cargo run -- review --config examples/eval/simple.yaml --tp

tn:
	cargo run -- review --config examples/eval/simple.yaml --tn



evals:
	python python/download_evals.py

embeddings:
	python python/prepare_embeddings.py

daemon:
	cargo run --release daemon --config examples/eval/simple.yaml --port 8080

health:
	@curl -s http://localhost:8080/health | jq . || echo "Daemon is not running"

submit:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make submit FILE=<path-to-file>"; \
	else \
		curl -s -X POST http://localhost:8080/submit \
			-H "Content-Type: application/json" \
			-d '{"file_path":"$(FILE)"}' | jq . || echo "Failed to submit job"; \
	fi

status:
	@if [ -z "$(JOB_ID)" ]; then \
		echo "Usage: make status JOB_ID=<job-id>"; \
	else \
		curl -s http://localhost:8080/status/$(JOB_ID) | jq . || echo "Failed to get status"; \
	fi
