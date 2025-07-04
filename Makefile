.PHONY: help detect review stats evals embeddings

help:
	@echo "Available targets:"
	@echo "  detect     - Run contamination detection with dev config"
	@echo "  review     - Run review mode with dev config and step-by-step output"
	@echo "  stats      - Run review mode with dev config and statistics output"
	@echo "  evals      - Download evaluation datasets using Python script"
	@echo "  embeddings - Download and prepare word embeddings using Python script"

detect:
	cargo run -- detect --config examples/dev/toxic.yaml

review:
	cargo run -- review --config examples/dev/toxic.yaml --step

stats:
	cargo run -- review --config examples/dev/toxic.yaml --stats

evals:
	python python/download_evals.py

embeddings:
	python python/download_embeddings.py