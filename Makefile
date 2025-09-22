SERVER_URL ?= localhost:8080

.PHONY: help server health submit status

help:
	@echo "Available targets:"
	@echo "  server     - Start the server on port 8080"
	@echo "  health     - Check server health status"
	@echo "  submit     - Submit a file for processing (usage: make submit FILE=<path>)"
	@echo "  status     - Check job status (usage: make status JOB_ID=<id>)"


server:
	cargo run --release server --port 8080

health:
	@curl -s http://$(SERVER_URL)/health | jq . || echo "Server is not running"

submit:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make submit FILE=<path-to-file>"; \
	else \
		curl -s -X POST http://$(SERVER_URL)/submit \
			-H "Content-Type: application/json" \
			-d '{"file_path":"$(FILE)"}' | jq . || echo "Failed to submit job"; \
	fi

status:
	@if [ -z "$(JOB_ID)" ]; then \
		echo "Usage: make status JOB_ID=<job-id>"; \
	else \
		curl -s http://$(SERVER_URL)/status/$(JOB_ID) | jq . || echo "Failed to get status"; \
	fi
