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

detect:
	cargo run --release detect --config examples/eval/simple.yaml

review:
	cargo run --release review --config examples/eval/simple.yaml --step

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
