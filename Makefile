.PHONY: help detect review stats simple evals evals-s3 embeddings

help:
	@echo "Available targets:"
	@echo "  detect     - Run contamination detection with dev config"
	@echo "  simple     - Run simple contamination detection with dev config"
	@echo "  review     - Run review mode with dev config and step-by-step output (use --full for complete training docs)"
	@echo "  stats      - Display eval dataset statistics from a directory (usage: make stats <directory>)"
	@echo "  tp         - Run review mode to show True Positives"
	@echo "  tn         - Run review mode to show True Negatives"
	@echo "  fp         - Run review mode to show False Positives"
	@echo "  fn         - Run review mode to show False Negatives"
	@echo "  evals      - Download evaluation datasets using Python script"
	@echo "  evals-s3   - Download evaluation datasets from S3 bucket decon-evals"
	@echo "  embeddings - Download and prepare word embeddings using Python script"
	@echo ""
	@echo "Daemon targets:"
	@echo "  daemon     - Start the daemon server on port 8080"
	@echo "  health     - Check daemon health status"
	@echo "  submit     - Submit a file for processing (usage: make submit FILE=<path>)"
	@echo "  status     - Check job status (usage: make status JOB_ID=<id>)"
	@echo ""
	@echo "Orchestration targets:"
	@echo "  orchestrate       - Run distributed orchestration (default: examples/orchestration.yaml, or CONFIG=<path>)"
	@echo "                      Optional parameters:"
	@echo "                        REMOTE_FILE_INPUT=<s3://bucket/path>         - Override input data location"
	@echo "                        REMOTE_REPORT_OUTPUT_DIR=<s3://bucket/path>  - Override report output location"
	@echo "                        REMOTE_CLEANED_OUTPUT_DIR=<s3://bucket/path> - Override cleaned files location"
	@echo "  orchestrate-test  - Test orchestration with example config"
	@echo "  orchestrate-debug - Test with MAX_FILES_DEBUG=5 for development"
	@echo ""
	@echo "Deployment targets (requires poormanray):"
	@echo "  deploy-wizard     - Interactive deployment wizard for setting up Decon on EC2"
	@echo "  deploy-status     - Check status of a deployment (usage: make deploy-status NAME=<cluster-name>)"
	@echo "  deploy-logs       - View deployment logs (usage: make deploy-logs NAME=<cluster-name> [LOG=daemon|orchestrator])"
	@echo "  deploy-terminate  - Terminate a deployment (usage: make deploy-terminate NAME=<cluster-name>)"
	@echo "  polling-auto-terminate - Monitor orchestrator logs and auto-terminate when complete (usage: make polling-auto-terminate NAME=<cluster-name>)"

minhash:
	cargo run --release detect --config examples/minhash.yaml

toxic:
	cargo run --release detect --config examples/toxic.yaml

simple:
	cargo run --release detect --config examples/simple.yaml

review:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: Directory parameter required. Usage: make stats <directory>"; \
		exit 1; \
	fi
	@DIR=$(filter-out $@,$(MAKECMDGOALS)); \
	cargo run --release -- review --step --dir $$DIR

stats:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: Directory parameter required. Usage: make stats <directory>"; \
		exit 1; \
	fi
	@DIR=$(filter-out $@,$(MAKECMDGOALS)); \
	cargo run --release -- review --stats --dir $$DIR

fn:
	cargo run -- review --config examples/simple.yaml --fn

fp:
	cargo run -- review --config examples/simple.yaml --fp

tp:
	cargo run -- review --config examples/simple.yaml --tp

tn:
	cargo run -- review --config examples/simple.yaml --tn



evals:
	python python/download_evals.py

evals-s3:
	mkdir -p fixtures/reference
	s5cmd sync s3://decon-evals/* fixtures/reference

embeddings:
	python python/prepare_embeddings.py

daemon:
	cargo run --release daemon --config examples/simple.yaml --port 8080

daemon-toxic:
	cargo run --release daemon --config examples/toxic.yaml --port 8080

daemon-minhash:
	cargo run --release daemon --config examples/minhash.yaml --port 8080

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

orchestrate:
	@if [ -z "$(CONFIG)" ]; then \
		CONFIG_FILE="examples/orchestration.yaml"; \
	else \
		CONFIG_FILE="$(CONFIG)"; \
	fi; \
	CMD="python python/orchestration.py --config $$CONFIG_FILE"; \
	if [ -n "$(REMOTE_FILE_INPUT)" ]; then \
		CMD="$$CMD --remote-file-input $(REMOTE_FILE_INPUT)"; \
	fi; \
	if [ -n "$(REMOTE_REPORT_OUTPUT_DIR)" ]; then \
		CMD="$$CMD --remote-report-output-dir $(REMOTE_REPORT_OUTPUT_DIR)"; \
	fi; \
	if [ -n "$(REMOTE_CLEANED_OUTPUT_DIR)" ]; then \
		CMD="$$CMD --remote-cleaned-output-dir $(REMOTE_CLEANED_OUTPUT_DIR)"; \
	fi; \
	echo "Running: $$CMD"; \
	$$CMD

orchestrate-debug:
	@echo "Testing orchestration in debug mode (max 5 files)"
	MAX_FILES_DEBUG=100 python python/orchestration.py --config examples/orchestration.yaml

# Deployment targets for managing remote clusters
deploy-wizard:
	@echo "Starting Decon deployment wizard..."
	python python/deploy.py wizard

deploy-status:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make deploy-status NAME=<cluster-name>"; \
		exit 1; \
	fi
	@python python/deploy.py status --name $(NAME)

deploy-logs:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make deploy-logs NAME=<cluster-name> [LOG=daemon|orchestrator]"; \
		exit 1; \
	fi
	@LOG_TYPE=$${LOG:-daemon}; \
	python python/deploy.py logs --name $(NAME) --log-type $$LOG_TYPE

deploy-terminate:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make deploy-terminate NAME=<cluster-name>"; \
		exit 1; \
	fi
	@echo "⚠️  WARNING: This will terminate all instances in cluster '$(NAME)'"
	@python python/deploy.py terminate --name $(NAME)

polling-auto-terminate:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make polling-auto-terminate NAME=<cluster-name>"; \
		exit 1; \
	fi
	@python python/deploy.py polling-auto-terminate --name $(NAME)

# This prevents make from treating the directory argument as a target when using 'make stats <directory>'
%:
	@:
