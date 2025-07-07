#!/usr/bin/env python3
"""
Download and prepare embeddings for contamination detection.
"""

import requests
import zipfile
import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import argparse


def download_fasttext_embeddings(output_dir="../fixtures/embeddings"):
    """Download FastText wiki-news embeddings and extract to output directory"""
    
    print("Downloading FastText wiki-news-300d-1M embeddings...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # URLs and filenames
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    zip_file = output_path / "wiki-news-300d-1M.vec.zip"
    vec_file = output_path / "wiki-news-300d-1M.vec"
    
    # Check if already downloaded
    if vec_file.exists():
        print(f"Embeddings already exist at {vec_file}")
        return str(vec_file)
    
    # Download the zip file
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Write zip file
    with open(zip_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {zip_file}")
    
    # Extract the zip file
    print("Extracting embeddings...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    # Remove zip file to save space
    zip_file.unlink()
    
    print(f"Embeddings extracted to {vec_file}")
    return str(vec_file)


def load_embeddings(vec_file_path):
    """Load embeddings from .vec file format"""
    
    print(f"Loading embeddings from {vec_file_path}...")
    
    embeddings = {}
    vectors = []
    words = []
    
    with open(vec_file_path, 'r', encoding='utf-8') as f:
        # First line contains vocab size and dimension
        first_line = f.readline().strip()
        vocab_size, dim = map(int, first_line.split())
        print(f"Vocabulary size: {vocab_size}, Dimensions: {dim}")
        
        # Read embeddings
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"Loaded {i}/{vocab_size} embeddings...")
            
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            
            embeddings[word] = vector
            vectors.append(vector)
            words.append(word)
    
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings, np.array(vectors), words


def apply_pca_to_embeddings(input_vec_file, target_dimensions=128, output_dir="../fixtures/embeddings"):
    """Apply PCA to reduce embedding dimensions and save in same format"""
    
    print(f"Applying PCA to reduce dimensions to {target_dimensions}...")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load original embeddings
    embeddings, vectors, words = load_embeddings(input_vec_file)
    original_dim = vectors.shape[1]
    
    print(f"Original dimensions: {original_dim}")
    print(f"Target dimensions: {target_dimensions}")
    
    if target_dimensions >= original_dim:
        print(f"Target dimensions ({target_dimensions}) >= original ({original_dim}), no reduction needed")
        return input_vec_file
    
    # Apply PCA
    print("Fitting PCA...")
    pca = PCA(n_components=target_dimensions)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explained variance ratio: {explained_variance:.4f}")
    
    # Generate output filename
    input_path = Path(input_vec_file)
    output_filename = input_path.stem.replace("-300d-", f"-{target_dimensions}d-") + input_path.suffix
    output_file = Path(output_dir) / output_filename
    
    # Save reduced embeddings in same format
    print(f"Saving reduced embeddings to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header line
        f.write(f"{len(words)} {target_dimensions}\n")
        
        # Write embeddings
        for i, (word, vector) in enumerate(zip(words, reduced_vectors)):
            if i % 100000 == 0:
                print(f"Saved {i}/{len(words)} embeddings...")
            
            vector_str = ' '.join(f'{x:.6f}' for x in vector)
            f.write(f"{word} {vector_str}\n")
    
    print(f"Reduced embeddings saved to {output_file}")
    print(f"Original size: {original_dim}D, Reduced size: {target_dimensions}D")
    print(f"Explained variance: {explained_variance:.4f}")
    
    return str(output_file)


def main():
    """Main function - downloads embeddings and applies PCA"""
    
    parser = argparse.ArgumentParser(description="Download and prepare embeddings for contamination detection")
    parser.add_argument('--dimensions', type=int, default=128,
                       help='Target dimensions for PCA (default: 128)')
    
    args = parser.parse_args()
    
    output_dir = "../fixtures/embeddings"
    
    # Ensure output directory exists at the start
    print(f"Ensuring output directory exists: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download embeddings
    print("=== Step 1: Downloading FastText embeddings ===")
    vec_file = download_fasttext_embeddings(output_dir)
    
    # Step 2: Apply PCA reduction
    print(f"\n=== Step 2: Applying PCA to reduce to {args.dimensions} dimensions ===")
    reduced_file = apply_pca_to_embeddings(vec_file, args.dimensions, output_dir)
    
    print(f"\n=== Complete! ===")
    print(f"Original embeddings: {vec_file}")
    print(f"Reduced embeddings: {reduced_file}")


if __name__ == "__main__":
    main()