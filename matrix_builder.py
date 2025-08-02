#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity Matrix Builder

Builds pre-computed similarity matrices with chunked processing and resume capability.
Designed for background processing of large music collections.

Features:
- Chunked processing to manage memory usage
- Resume capability for interrupted computations
- Progress tracking with ETA calculation
- Multiple algorithm support
- Sparse matrix storage for efficiency

@author: ffx
"""

import os
import json
import numpy as np
from scipy import sparse
import time
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
import sys
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from similarity_analyzer import CoverSongSimilarityAnalyzer, SimilarityAlgorithm
from similarity_matrix import SimilarityMatrixManager

try:
    from src.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s │ %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MatrixBuilder:
    """
    Builds similarity matrices with chunked processing and resume capability.
    """
    
    def __init__(self, chunk_size: int = 200):
        """
        Initialize matrix builder.
        
        Args:
            chunk_size: Size of chunks for processing (200x200 recommended)
        """
        self.chunk_size = chunk_size
        self.matrix_manager = SimilarityMatrixManager()
        self.analyzer = CoverSongSimilarityAnalyzer(str(self.matrix_manager.library_path))
        
        # Progress tracking
        self.progress_file = self.matrix_manager.cache_path / "build_progress.json"
        
        logger.info(f"Matrix builder initialized with chunk size: {chunk_size}x{chunk_size}")
    
    def _get_all_tracks(self) -> List[Tuple[str, str]]:
        """
        Get list of all available tracks.
        
        Returns:
            List of (release_id, track_position) tuples
        """
        tracks = []
        releases = self.analyzer.get_available_releases()
        
        for release in releases:
            for track_position in release['tracks']:
                tracks.append((release['id'], track_position))
        
        return tracks
    
    def _create_track_index(self, tracks: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        Create mapping from track_id to matrix index.
        
        Args:
            tracks: List of (release_id, track_position) tuples
            
        Returns:
            Dictionary mapping track_id -> matrix_index
        """
        track_index = {}
        for idx, (release_id, track_position) in enumerate(tracks):
            track_id = f"{release_id}:{track_position}"
            track_index[track_id] = idx
        
        return track_index
    
    def _load_progress(self, algorithm: SimilarityAlgorithm) -> Dict:
        """Load progress information for resuming computation."""
        if not self.progress_file.exists():
            return {}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            return progress_data.get(algorithm.value, {})
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_progress(self, algorithm: SimilarityAlgorithm, progress_data: Dict):
        """Save progress information for resuming computation."""
        try:
            # Load existing progress file
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    all_progress = json.load(f)
            else:
                all_progress = {}
            
            # Update progress for this algorithm
            all_progress[algorithm.value] = progress_data
            
            # Save updated progress
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(all_progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _calculate_chunk_similarity(self, tracks: List[Tuple[str, str]], 
                                  algorithm: SimilarityAlgorithm,
                                  chunk_i: int, chunk_j: int) -> np.ndarray:
        """
        Calculate similarity for a specific chunk.
        
        Args:
            tracks: List of all tracks
            algorithm: Similarity algorithm to use
            chunk_i: Row chunk index
            chunk_j: Column chunk index
            
        Returns:
            Similarity matrix chunk as numpy array
        """
        start_i = chunk_i * self.chunk_size
        end_i = min(start_i + self.chunk_size, len(tracks))
        start_j = chunk_j * self.chunk_size
        end_j = min(start_j + self.chunk_size, len(tracks))
        
        chunk_rows = end_i - start_i
        chunk_cols = end_j - start_j
        chunk_matrix = np.zeros((chunk_rows, chunk_cols), dtype=np.float32)
        
        for i in range(chunk_rows):
            track_i_idx = start_i + i
            release_id_i, track_pos_i = tracks[track_i_idx]
            
            # Load features for track i
            analysis_i = self.analyzer._load_track_analysis(release_id_i, track_pos_i)
            if analysis_i is None:
                continue
            
            features_i = self.analyzer._extract_all_features(analysis_i)
            if features_i is None:
                continue
            
            for j in range(chunk_cols):
                track_j_idx = start_j + j
                
                # Skip if we're in the lower triangle (matrix is symmetric)
                if track_j_idx < track_i_idx:
                    continue
                
                release_id_j, track_pos_j = tracks[track_j_idx]
                
                # Self-similarity is 1.0
                if track_i_idx == track_j_idx:
                    chunk_matrix[i, j] = 1.0
                    continue
                
                # Load features for track j
                analysis_j = self.analyzer._load_track_analysis(release_id_j, track_pos_j)
                if analysis_j is None:
                    continue
                
                features_j = self.analyzer._extract_all_features(analysis_j)
                if features_j is None:
                    continue
                
                # Calculate similarity based on algorithm
                try:
                    if algorithm == SimilarityAlgorithm.HPCP_CROSS_CORRELATION:
                        if 'hpcp_mean' in features_i and 'hpcp_mean' in features_j:
                            similarity, _ = self.analyzer._calculate_pitch_shifted_similarity(
                                features_i['hpcp_mean'], features_j['hpcp_mean']
                            )
                        else:
                            similarity = 0.0
                    elif algorithm == SimilarityAlgorithm.CHROMA_SIMILARITY:
                        similarity = self.analyzer._calculate_chroma_similarity(features_i, features_j)
                    elif algorithm == SimilarityAlgorithm.HARMONIC_KEY_MATCHING:
                        key_i = self.analyzer._get_best_key(features_i)
                        key_j = self.analyzer._get_best_key(features_j)
                        from similarity_analyzer import CamelotWheel
                        similarity = CamelotWheel.calculate_harmonic_compatibility(key_i, key_j)
                    else:  # COMBINED_FEATURES
                        similarities = self.analyzer._calculate_combined_similarity(features_i, features_j)
                        similarity = similarities['combined_similarity']
                    
                    chunk_matrix[i, j] = similarity
                    
                except Exception as e:
                    logger.debug(f"Error calculating similarity for {track_i_idx}:{track_j_idx}: {e}")
                    chunk_matrix[i, j] = 0.0
        
        return chunk_matrix
    
    def build_matrix(self, algorithm: SimilarityAlgorithm, 
                    min_similarity: float = 0.1,
                    resume: bool = True) -> bool:
        """
        Build complete similarity matrix for given algorithm.
        
        Args:
            algorithm: Similarity algorithm to use
            min_similarity: Minimum similarity to store (for sparsity)
            resume: Whether to resume from previous computation
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Building similarity matrix for {algorithm.value}")
        
        # Get all tracks
        tracks = self._get_all_tracks()
        n_tracks = len(tracks)
        
        if n_tracks == 0:
            logger.error("No tracks found for matrix computation")
            return False
        
        logger.info(f"Building matrix for {n_tracks} tracks ({n_tracks}x{n_tracks})")
        
        # Create track index
        track_index = self._create_track_index(tracks)
        
        # Calculate chunk dimensions
        n_chunks = (n_tracks + self.chunk_size - 1) // self.chunk_size
        total_chunks = (n_chunks * (n_chunks + 1)) // 2  # Only upper triangle
        
        logger.info(f"Processing {n_chunks}x{n_chunks} chunks ({total_chunks} unique chunks)")
        
        # Load progress for resuming
        progress_data = self._load_progress(algorithm) if resume else {}
        completed_chunks = set(tuple(chunk) for chunk in progress_data.get('completed_chunks', []))
        
        # Initialize sparse matrix data
        matrix_data = []
        matrix_rows = []
        matrix_cols = []
        
        # Progress tracking
        start_time = time.time()
        chunks_processed = len(completed_chunks)
        
        with tqdm(total=total_chunks, initial=chunks_processed, 
                 desc=f"Building {algorithm.value} matrix", unit="chunk") as pbar:
            
            for chunk_i in range(n_chunks):
                for chunk_j in range(chunk_i, n_chunks):  # Only upper triangle
                    chunk_key = (chunk_i, chunk_j)
                    
                    # Skip if already completed
                    if chunk_key in completed_chunks:
                        continue
                    
                    # Calculate chunk similarity
                    try:
                        chunk_matrix = self._calculate_chunk_similarity(
                            tracks, algorithm, chunk_i, chunk_j
                        )
                        
                        # Extract non-zero similarities
                        start_i = chunk_i * self.chunk_size
                        start_j = chunk_j * self.chunk_size
                        
                        nonzero_rows, nonzero_cols = np.where(chunk_matrix >= min_similarity)
                        
                        for row_idx, col_idx in zip(nonzero_rows, nonzero_cols):
                            global_row = start_i + row_idx
                            global_col = start_j + col_idx
                            similarity = chunk_matrix[row_idx, col_idx]
                            
                            matrix_data.append(similarity)
                            matrix_rows.append(global_row)
                            matrix_cols.append(global_col)
                            
                            # Add symmetric entry (if not on diagonal)
                            if global_row != global_col:
                                matrix_data.append(similarity)
                                matrix_rows.append(global_col)
                                matrix_cols.append(global_row)
                        
                        # Update progress
                        completed_chunks.add(chunk_key)
                        chunks_processed += 1
                        
                        # Save progress periodically
                        if chunks_processed % 10 == 0:
                            progress_data = {
                                'completed_chunks': list(completed_chunks),
                                'chunks_processed': chunks_processed,
                                'total_chunks': total_chunks,
                                'last_update': datetime.now().isoformat(),
                                'estimated_completion': None
                            }
                            
                            # Calculate ETA
                            elapsed_time = time.time() - start_time
                            if chunks_processed > 0:
                                time_per_chunk = elapsed_time / chunks_processed
                                remaining_chunks = total_chunks - chunks_processed
                                eta_seconds = remaining_chunks * time_per_chunk
                                eta = datetime.now() + timedelta(seconds=eta_seconds)
                                progress_data['estimated_completion'] = eta.isoformat()
                            
                            self._save_progress(algorithm, progress_data)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_key}: {e}")
                        continue
        
        # Create sparse matrix
        logger.info("Creating sparse matrix...")
        similarity_matrix = sparse.csr_matrix(
            (matrix_data, (matrix_rows, matrix_cols)),
            shape=(n_tracks, n_tracks),
            dtype=np.float32
        )
        
        # Calculate matrix statistics
        total_elements = n_tracks * n_tracks
        sparsity = 1.0 - (similarity_matrix.nnz / total_elements)
        
        logger.info(f"Matrix created: {similarity_matrix.shape}, "
                   f"{similarity_matrix.nnz:,} non-zero elements, "
                   f"{sparsity:.1%} sparsity")
        
        # Save matrix
        metadata = {
            "min_similarity_threshold": min_similarity,
            "computation_time_seconds": time.time() - start_time,
            "chunks_processed": chunks_processed,
            "chunk_size": self.chunk_size
        }
        
        self.matrix_manager.save_matrix(algorithm, similarity_matrix, track_index, metadata)
        
        # Clear progress file for this algorithm
        if resume:
            try:
                if self.progress_file.exists():
                    with open(self.progress_file, 'r', encoding='utf-8') as f:
                        all_progress = json.load(f)
                    if algorithm.value in all_progress:
                        del all_progress[algorithm.value]
                    with open(self.progress_file, 'w', encoding='utf-8') as f:
                        json.dump(all_progress, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not clear progress data: {e}")
        
        logger.info(f"Successfully built matrix for {algorithm.value}")
        return True
    
    def get_build_progress(self, algorithm: SimilarityAlgorithm) -> Dict:
        """Get current build progress for an algorithm."""
        progress_data = self._load_progress(algorithm)
        
        if not progress_data:
            return {"status": "not_started"}
        
        chunks_processed = progress_data.get('chunks_processed', 0)
        total_chunks = progress_data.get('total_chunks', 1)
        
        progress = {
            "status": "in_progress",
            "chunks_processed": chunks_processed,
            "total_chunks": total_chunks,
            "percentage": (chunks_processed / total_chunks) * 100,
            "last_update": progress_data.get('last_update'),
            "estimated_completion": progress_data.get('estimated_completion')
        }
        
        return progress


def main():
    """Command-line interface for building similarity matrices."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Similarity Matrices')
    parser.add_argument('--algorithm', required=True,
                       choices=['hpcp', 'chroma', 'combined', 'harmonic'],
                       help='Similarity algorithm to build matrix for')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Chunk size for processing (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.1,
                       help='Minimum similarity threshold (default: 0.1)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from scratch (do not resume)')
    
    args = parser.parse_args()
    
    # Map algorithm names
    algorithm_map = {
        'hpcp': SimilarityAlgorithm.HPCP_CROSS_CORRELATION,
        'chroma': SimilarityAlgorithm.CHROMA_SIMILARITY,
        'combined': SimilarityAlgorithm.COMBINED_FEATURES,
        'harmonic': SimilarityAlgorithm.HARMONIC_KEY_MATCHING
    }
    
    algorithm = algorithm_map[args.algorithm]
    
    # Create builder and build matrix
    builder = MatrixBuilder(chunk_size=args.chunk_size)
    
    logger.info(f"Starting matrix build for {algorithm.value}")
    
    success = builder.build_matrix(
        algorithm=algorithm,
        min_similarity=args.min_similarity,
        resume=not args.no_resume
    )
    
    if success:
        logger.info("Matrix build completed successfully!")
    else:
        logger.error("Matrix build failed!")


if __name__ == '__main__':
    main()