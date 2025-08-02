#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity Matrix Storage and Retrieval System

Manages pre-computed similarity matrices for instant music similarity lookups.
Uses sparse matrix format for efficient storage and provides fast similarity queries.

Features:
- Config-based data path resolution
- Multiple algorithm support (HPCP, Chroma, Combined, Harmonic)
- Sparse matrix storage (HDF5/NPZ format)
- Track index mapping and metadata
- Incremental updates and cache management

@author: ffx
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config import load_config
    from src.logger import logger
except ImportError:
    # Fallback imports
    def load_config():
        return {"LIBRARY_PATH": "/home/ffx/.cache/discogsLibary/discogsLib"}
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s │ %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from similarity_analyzer import SimilarityAlgorithm


class SimilarityMatrixManager:
    """
    Manages similarity matrices for fast music similarity lookups.
    
    Stores matrices in the similarity subfolder of the configured library path.
    """
    
    def __init__(self):
        """Initialize the matrix manager with config-based paths."""
        self.config = load_config()
        self.library_path = Path(self.config.get("LIBRARY_PATH", ""))
        
        # Create similarity data path
        if self.library_path.exists():
            # Go one level up from discogsLib to create similarity folder
            self.similarity_path = self.library_path.parent / "similarity"
        else:
            # Fallback path
            self.similarity_path = Path.home() / ".cache" / "discogsLibary" / "similarity"
        
        # Create directory structure
        self.matrices_path = self.similarity_path / "matrices"
        self.metadata_path = self.similarity_path / "metadata"
        self.cache_path = self.similarity_path / "cache"
        
        self._ensure_directories()
        
        # Matrix filenames for different algorithms
        self.matrix_files = {
            SimilarityAlgorithm.HPCP_CROSS_CORRELATION: "hpcp_matrix.npz",
            SimilarityAlgorithm.CHROMA_SIMILARITY: "chroma_matrix.npz",
            SimilarityAlgorithm.COMBINED_FEATURES: "combined_matrix.npz",
            SimilarityAlgorithm.HARMONIC_KEY_MATCHING: "harmonic_matrix.npz"
        }
        
        # Track index mapping (track_id -> matrix_index)
        self.track_index_file = self.metadata_path / "track_index.json"
        self.computation_log_file = self.metadata_path / "computation_log.json"
        
        logger.info(f"Similarity matrices will be stored in: {self.similarity_path}")
    
    def _ensure_directories(self):
        """Create necessary directory structure."""
        for path in [self.similarity_path, self.matrices_path, 
                     self.metadata_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _get_track_id(self, release_id: str, track_position: str) -> str:
        """Generate unique track ID from release and position."""
        return f"{release_id}:{track_position}"
    
    def load_track_index(self) -> Dict[str, int]:
        """
        Load track index mapping from file.
        
        Returns:
            Dictionary mapping track_id -> matrix_index
        """
        if not self.track_index_file.exists():
            return {}
        
        try:
            with open(self.track_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading track index: {e}")
            return {}
    
    def save_track_index(self, track_index: Dict[str, int]):
        """
        Save track index mapping to file.
        
        Args:
            track_index: Dictionary mapping track_id -> matrix_index
        """
        try:
            with open(self.track_index_file, 'w', encoding='utf-8') as f:
                json.dump(track_index, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved track index with {len(track_index)} tracks")
        except Exception as e:
            logger.error(f"Error saving track index: {e}")
    
    def load_computation_log(self) -> Dict:
        """Load computation history log."""
        if not self.computation_log_file.exists():
            return {"computations": [], "last_update": None}
        
        try:
            with open(self.computation_log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading computation log: {e}")
            return {"computations": [], "last_update": None}
    
    def save_computation_log(self, log_data: Dict):
        """Save computation history log."""
        try:
            with open(self.computation_log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving computation log: {e}")
    
    def matrix_exists(self, algorithm: SimilarityAlgorithm) -> bool:
        """Check if matrix exists for given algorithm."""
        matrix_file = self.matrices_path / self.matrix_files[algorithm]
        return matrix_file.exists()
    
    def save_matrix(self, algorithm: SimilarityAlgorithm, 
                   similarity_matrix: sparse.csr_matrix,
                   track_index: Dict[str, int],
                   metadata: Dict = None):
        """
        Save similarity matrix to disk.
        
        Args:
            algorithm: Similarity algorithm used
            similarity_matrix: Sparse similarity matrix
            track_index: Mapping of track_id -> matrix_index
            metadata: Additional metadata about the computation
        """
        matrix_file = self.matrices_path / self.matrix_files[algorithm]
        
        try:
            # Save sparse matrix
            sparse.save_npz(matrix_file, similarity_matrix)
            
            # Save track index
            self.save_track_index(track_index)
            
            # Update computation log
            log_data = self.load_computation_log()
            computation_entry = {
                "algorithm": algorithm.value,
                "timestamp": datetime.now().isoformat(),
                "matrix_shape": similarity_matrix.shape,
                "num_tracks": len(track_index),
                "sparsity": 1.0 - (similarity_matrix.nnz / (similarity_matrix.shape[0] * similarity_matrix.shape[1])),
                "file_size_mb": matrix_file.stat().st_size / (1024 * 1024) if matrix_file.exists() else 0
            }
            
            if metadata:
                computation_entry.update(metadata)
            
            log_data["computations"].append(computation_entry)
            log_data["last_update"] = datetime.now().isoformat()
            self.save_computation_log(log_data)
            
            logger.info(f"Saved {algorithm.value} matrix: {similarity_matrix.shape}, "
                       f"{similarity_matrix.nnz:,} non-zero elements, "
                       f"{computation_entry['file_size_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error saving matrix for {algorithm.value}: {e}")
    
    def load_matrix(self, algorithm: SimilarityAlgorithm) -> Tuple[Optional[sparse.csr_matrix], Dict[str, int]]:
        """
        Load similarity matrix from disk.
        
        Args:
            algorithm: Similarity algorithm to load
            
        Returns:
            Tuple of (similarity_matrix, track_index) or (None, {}) if not found
        """
        matrix_file = self.matrices_path / self.matrix_files[algorithm]
        
        if not matrix_file.exists():
            logger.warning(f"Matrix file not found for {algorithm.value}")
            return None, {}
        
        try:
            # Load sparse matrix
            similarity_matrix = sparse.load_npz(matrix_file)
            
            # Load track index
            track_index = self.load_track_index()
            
            logger.info(f"Loaded {algorithm.value} matrix: {similarity_matrix.shape}, "
                       f"{similarity_matrix.nnz:,} non-zero elements")
            
            return similarity_matrix, track_index
            
        except Exception as e:
            logger.error(f"Error loading matrix for {algorithm.value}: {e}")
            return None, {}
    
    def get_similarities(self, algorithm: SimilarityAlgorithm, 
                        reference_track_id: str,
                        min_similarity: float = 0.0,
                        max_results: int = None) -> List[Tuple[str, float]]:
        """
        Get similar tracks from pre-computed matrix.
        
        Args:
            algorithm: Similarity algorithm to use
            reference_track_id: Reference track ID (release_id:track_position)
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of (track_id, similarity_score) tuples, sorted by similarity desc
        """
        matrix, track_index = self.load_matrix(algorithm)
        
        if matrix is None or not track_index:
            logger.error(f"Matrix not available for {algorithm.value}")
            return []
        
        if reference_track_id not in track_index:
            logger.error(f"Track {reference_track_id} not found in matrix")
            return []
        
        # Get reference track index
        ref_idx = track_index[reference_track_id]
        
        # Get similarities for reference track
        similarities = matrix[ref_idx].toarray().flatten()
        
        # Create reverse index (matrix_index -> track_id)
        reverse_index = {idx: track_id for track_id, idx in track_index.items()}
        
        # Find similar tracks
        similar_tracks = []
        for idx, similarity in enumerate(similarities):
            if idx == ref_idx:  # Skip self
                continue
            if similarity >= min_similarity:
                track_id = reverse_index.get(idx)
                if track_id:
                    similar_tracks.append((track_id, float(similarity)))
        
        # Sort by similarity (descending)
        similar_tracks.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        if max_results:
            similar_tracks = similar_tracks[:max_results]
        
        return similar_tracks
    
    def get_matrix_info(self, algorithm: SimilarityAlgorithm) -> Dict:
        """
        Get information about a stored matrix.
        
        Args:
            algorithm: Similarity algorithm
            
        Returns:
            Dictionary with matrix information
        """
        matrix_file = self.matrices_path / self.matrix_files[algorithm]
        
        if not matrix_file.exists():
            return {"exists": False}
        
        try:
            # Load matrix to get shape info
            matrix = sparse.load_npz(matrix_file)
            
            # Get file size
            file_size_mb = matrix_file.stat().st_size / (1024 * 1024)
            
            # Get computation log info
            log_data = self.load_computation_log()
            latest_computation = None
            for computation in reversed(log_data.get("computations", [])):
                if computation.get("algorithm") == algorithm.value:
                    latest_computation = computation
                    break
            
            info = {
                "exists": True,
                "algorithm": algorithm.value,
                "shape": matrix.shape,
                "num_tracks": matrix.shape[0],
                "non_zero_elements": matrix.nnz,
                "sparsity": 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
                "file_size_mb": file_size_mb,
                "last_computation": latest_computation
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting matrix info for {algorithm.value}: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_available_matrices(self) -> Dict[str, Dict]:
        """
        List all available similarity matrices.
        
        Returns:
            Dictionary mapping algorithm names to matrix info
        """
        matrices = {}
        
        for algorithm in SimilarityAlgorithm:
            info = self.get_matrix_info(algorithm)
            matrices[algorithm.value] = info
        
        return matrices
    
    def delete_matrix(self, algorithm: SimilarityAlgorithm) -> bool:
        """
        Delete stored matrix for given algorithm.
        
        Args:
            algorithm: Similarity algorithm
            
        Returns:
            True if successfully deleted, False otherwise
        """
        matrix_file = self.matrices_path / self.matrix_files[algorithm]
        
        try:
            if matrix_file.exists():
                matrix_file.unlink()
                logger.info(f"Deleted matrix for {algorithm.value}")
                return True
            else:
                logger.warning(f"Matrix file not found for {algorithm.value}")
                return False
        except Exception as e:
            logger.error(f"Error deleting matrix for {algorithm.value}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics for similarity data.
        
        Returns:
            Dictionary with storage information
        """
        stats = {
            "similarity_path": str(self.similarity_path),
            "total_size_mb": 0,
            "matrices": {},
            "metadata_size_mb": 0,
            "cache_size_mb": 0
        }
        
        try:
            # Calculate matrices size
            for matrix_file in self.matrices_path.glob("*.npz"):
                size_mb = matrix_file.stat().st_size / (1024 * 1024)
                stats["matrices"][matrix_file.name] = size_mb
                stats["total_size_mb"] += size_mb
            
            # Calculate metadata size
            for metadata_file in self.metadata_path.glob("*.json"):
                size_mb = metadata_file.stat().st_size / (1024 * 1024)
                stats["metadata_size_mb"] += size_mb
            
            # Calculate cache size
            for cache_file in self.cache_path.rglob("*"):
                if cache_file.is_file():
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    stats["cache_size_mb"] += size_mb
            
            stats["total_size_mb"] += stats["metadata_size_mb"] + stats["cache_size_mb"]
            
        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")
        
        return stats


def main():
    """Test the similarity matrix manager."""
    manager = SimilarityMatrixManager()
    
    print("=== Similarity Matrix Manager ===")
    print(f"Storage path: {manager.similarity_path}")
    
    # List available matrices
    matrices = manager.list_available_matrices()
    print(f"\nAvailable matrices:")
    for algorithm, info in matrices.items():
        if info.get("exists"):
            print(f"  {algorithm}: {info['shape']}, {info['file_size_mb']:.1f} MB")
        else:
            print(f"  {algorithm}: Not available")
    
    # Storage stats
    stats = manager.get_storage_stats()
    print(f"\nStorage statistics:")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Matrices: {sum(stats['matrices'].values()):.1f} MB")
    print(f"  Metadata: {stats['metadata_size_mb']:.1f} MB")


if __name__ == '__main__':
    main()