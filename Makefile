.PHONY: help detect review stats simple evals evals-s3 push-evals embeddings refine

help:
	@echo "Available targets:"
	@echo "  detect     - Run contamination detection with dev config"
	@echo "  simple     - Run simple contamination detection with dev config"
	@echo "  review     - Review contamination cases from a directory (usage: make review [directory], default: fixtures/output)"
	@echo "  stats      - Display eval dataset statistics from a directory (usage: make stats [directory], default: fixtures/output)"
	@echo "  refine     - Run reference data refinement"
	@echo "  reference-stats - Display statistics for reference datasets (usage: make reference-stats <path>)"
	@echo "  evals      - Download evaluation datasets using Python script"
	@echo "  evals-s3   - Download evaluation datasets from S3 bucket decon-evals"
	@echo "  push-evals - Push evaluation datasets to S3 bucket decon-evals (deletes existing content)"
	@echo "  embeddings - Download and prepare word embeddings using Python script"
	@echo ""
	@echo "Server targets:"
	@echo "  server     - Start the server on port 8080"
	@echo "  health     - Check server health status"
	@echo "  submit     - Submit a file for processing (usage: make submit FILE=<path>)"
	@echo "  status     - Check job status (usage: make status JOB_ID=<id>)"
	@echo ""
	@echo "Orchestration targets:"
	@echo "  orchestrate       - Run distributed orchestration (default: config/orchestration.yaml, or CONFIG=<path>)"
	@echo "                      Optional parameters:"
	@echo "                        REMOTE_FILE_INPUT=<s3://bucket/path>         - Override input data location"
	@echo "                        REMOTE_REPORT_OUTPUT_DIR=<s3://bucket/path>  - Override report output location"
	@echo "                        REMOTE_CLEANED_OUTPUT_DIR=<s3://bucket/path> - Override cleaned files location"
	@echo "  orchestrate-test  - Test orchestration with example config"
	@echo "  orchestrate-debug - Test with MAX_FILES_DEBUG=5 for development"
	@echo ""
	@echo "Deployment targets (requires poormanray):"
	@echo "  poormanray-command-generator - Interactive walk to setup Decon on EC2"
	@echo "  deploy-status     - Check status of a deployment (usage: make deploy-status NAME=<cluster-name>)"
	@echo "  deploy-logs       - View deployment logs (usage: make deploy-logs NAME=<cluster-name> [LOG=server|orchestrator])"
	@echo "  deploy-terminate  - Terminate a deployment (usage: make deploy-terminate NAME=<cluster-name>)"
	@echo "  polling-auto-terminate - Monitor orchestrator logs and auto-terminate when complete (usage: make polling-auto-terminate NAME=<cluster-name>)"
	@echo "  polling-auto-status - Check if work is still running (prints 'Running' or 'Finished') (usage: make polling-auto-status NAME=<cluster-name>)"


simple-word:
	cargo run --release detect --config config/simple-word.yaml

simple:
	cargo run --release detect --config config/default.yaml

review:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		DIR="fixtures/output"; \
		echo "Using default directory: $$DIR"; \
	else \
		DIR=$(filter-out $@,$(MAKECMDGOALS)); \
	fi; \
	FILTER_ARGS=""; \
	if [ -n "$(MIN_OVERLAP_RATIO)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-overlap-ratio $(MIN_OVERLAP_RATIO)"; \
	fi; \
	if [ -n "$(MIN_IDF_SCORE)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-idf-score $(MIN_IDF_SCORE)"; \
	fi; \
	if [ -n "$(MIN_LENGTH)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-length $(MIN_LENGTH)"; \
	fi; \
	if [ -n "$(EVAL)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --eval $(EVAL)"; \
	fi; \
	cargo run --release -- review --step --dir $$DIR $$FILTER_ARGS

stats:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		DIR="fixtures/output"; \
		echo "Using default directory: $$DIR"; \
	else \
		DIR=$(filter-out $@,$(MAKECMDGOALS)); \
	fi; \
	FILTER_ARGS=""; \
	if [ -n "$(MIN_OVERLAP_RATIO)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-overlap-ratio $(MIN_OVERLAP_RATIO)"; \
	fi; \
	if [ -n "$(MIN_IDF_SCORE)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-idf-score $(MIN_IDF_SCORE)"; \
	fi; \
	if [ -n "$(MIN_LENGTH)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --min-length $(MIN_LENGTH)"; \
	fi; \
	if [ -n "$(EVAL)" ]; then \
		FILTER_ARGS="$$FILTER_ARGS --eval $(EVAL)"; \
	fi; \
	cargo run --release -- review --stats --dir $$DIR $$FILTER_ARGS

evals:
	python python/evals.py --download

evals-list:
	python python/evals.py --list

evals-s3:
	s5cmd sync s3://decon-evals/* fixtures/

push-evals:
	@echo "Syncing reference files to s3://decon-evals/ with deletion..."
	s5cmd sync --delete fixtures/reference s3://decon-evals/
	@echo "Push to S3 complete!"

embeddings:
	python python/prepare_embeddings.py

refine:
	cargo run --release references --refine

reference-stats:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make reference-stats <path-to-reference-directory>"; \
		echo "Example: make reference-stats fixtures/reference"; \
	else \
		DIR=$(filter-out $@,$(MAKECMDGOALS)); \
		cargo run --release references --stats $$DIR; \
	fi

server:
	cargo run --release server --config config/simple.yaml --port 8080

health:
	@curl -s http://localhost:8080/health | jq . || echo "Server is not running"

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
		CONFIG_FILE="config/orchestration.yaml"; \
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
	MAX_FILES_DEBUG=100 python python/orchestration.py --config config/orchestration.yaml

# Deployment targets for managing remote clusters
poormanray-command-generator:
	python python/deploy.py wizard

generate-orchestrator-command:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Error: S3 input path required. Usage: make generate-orchestrator-command s3://path/to/data"; \
		exit 1; \
	fi
	@python python/generate_orchestrator_command.py $(filter-out $@,$(MAKECMDGOALS))

# Prevent make from treating the S3 path as a target
%:
	@:

deploy-status:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make deploy-status NAME=<cluster-name>"; \
		exit 1; \
	fi
	@python python/deploy.py status --name $(NAME)

deploy-logs:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make deploy-logs NAME=<cluster-name> [LOG=server|orchestrator]"; \
		exit 1; \
	fi
	@LOG_TYPE=$${LOG:-server}; \
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

polling-auto-status:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required. Usage: make polling-auto-status NAME=<cluster-name>"; \
		exit 1; \
	fi
	@python python/deploy.py polling-auto-terminate --name $(NAME) --dry-run

# This prevents make from treating the directory argument as a target when using 'make stats <directory>'
%:
	@:
