#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cover Song Similarity Analyzer

Based on Essentia's cover song similarity tutorial:
https://essentia.upf.edu/tutorial_similarity_cover.html

Uses existing HPCP (Harmonic Pitch Class Profile) analysis data
to find cover songs and similar tracks with pitch shifting tolerance.

@author: ffx
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from enum import Enum
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.logger import logger
except ImportError:
    # Fallback logger if not available
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s │ %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SimilarityAlgorithm(Enum):
    """Available similarity algorithms."""
    HPCP_CROSS_CORRELATION = "hpcp_cross_correlation"
    CHROMA_SIMILARITY = "chroma_similarity"
    COMBINED_FEATURES = "combined_features"
    HARMONIC_KEY_MATCHING = "harmonic_key_matching"


class CamelotWheel:
    """
    Camelot wheel implementation for harmonic key matching.
    
    Maps musical keys to Camelot notation and provides harmonic compatibility.
    """
    
    # Camelot wheel mapping (Key -> Camelot notation)
    KEY_TO_CAMELOT = {
        # Major keys (outer wheel)
        'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
        'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B', 'Ab': '4B',
        'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B',
        
        # Minor keys (inner wheel)
        'Cm': '5A', 'C#m': '12A', 'Dbm': '12A', 'Dm': '7A', 'D#m': '2A', 'Ebm': '2A',
        'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gbm': '11A', 'Gm': '6A', 'G#m': '1A', 'Abm': '1A',
        'Am': '8A', 'A#m': '3A', 'Bbm': '3A', 'Bm': '10A'
    }
    
    # Reverse mapping for display
    CAMELOT_TO_KEY = {v: k for k, v in KEY_TO_CAMELOT.items()}
    
    @classmethod
    def normalize_key(cls, key_str: str) -> str:
        """Normalize key string to standard format."""
        if not key_str:
            return ""
        
        key_str = key_str.strip()
        
        # Handle different minor notations
        if key_str.lower().endswith('minor') or key_str.lower().endswith(' minor'):
            key_str = key_str.replace('minor', 'm').replace('Minor', 'm').replace(' ', '')
        elif key_str.lower().endswith('maj') or key_str.lower().endswith('major'):
            key_str = key_str.replace('maj', '').replace('major', '').replace(' ', '')
        
        # Capitalize first letter
        if len(key_str) > 0:
            key_str = key_str[0].upper() + key_str[1:]
        
        return key_str
    
    @classmethod
    def get_camelot_code(cls, key_str: str) -> str:
        """Get Camelot code for a musical key."""
        normalized_key = cls.normalize_key(key_str)
        return cls.KEY_TO_CAMELOT.get(normalized_key, "")
    
    @classmethod
    def get_harmonic_keys(cls, camelot_code: str) -> List[str]:
        """
        Get harmonically compatible keys for a given Camelot code.
        
        Compatible keys are:
        - Same code (perfect match)
        - Adjacent codes (+/-1)
        - Inner/outer ring (same number, different letter)
        """
        if not camelot_code or len(camelot_code) != 2:
            return []
        
        try:
            number = int(camelot_code[0:-1])
            letter = camelot_code[-1]
        except ValueError:
            return []
        
        compatible = [camelot_code]  # Self
        
        # Adjacent numbers (wrap around 1-12)
        prev_num = 12 if number == 1 else number - 1
        next_num = 1 if number == 12 else number + 1
        
        compatible.extend([
            f"{prev_num}{letter}",  # -1
            f"{next_num}{letter}",  # +1
            f"{number}{'A' if letter == 'B' else 'B'}"  # Inner/outer ring
        ])
        
        return compatible
    
    @classmethod
    def calculate_harmonic_compatibility(cls, key1: str, key2: str) -> float:
        """
        Calculate harmonic compatibility between two keys (0.0 to 1.0).
        
        1.0 = Perfect match (same key)
        0.8 = Adjacent keys or inner/outer ring
        0.0 = Non-compatible keys
        """
        camelot1 = cls.get_camelot_code(key1)
        camelot2 = cls.get_camelot_code(key2)
        
        if not camelot1 or not camelot2:
            return 0.0
        
        if camelot1 == camelot2:
            return 1.0
        
        compatible_keys = cls.get_harmonic_keys(camelot1)
        if camelot2 in compatible_keys:
            return 0.8
        
        return 0.0


class CoverSongSimilarityAnalyzer:
    """
    Analyzes cover song similarity using HPCP features from existing Essentia analysis data.
    
    Implements cross-correlation with pitch shifting as described in:
    https://essentia.upf.edu/tutorial_similarity_cover.html
    """
    
    def __init__(self, library_path: str):
        """
        Initialize the analyzer.
        
        Args:
            library_path: Path to the Discogs library directory
        """
        self.library_path = Path(library_path)
        self.tracks_cache = {}
        logger.info("Cover Song Similarity Analyzer initialized")
    
    def _load_track_analysis(self, release_id: str, track_position: str) -> Optional[Dict]:
        """
        Load existing Essentia analysis data for a track.
        
        Args:
            release_id: Discogs release ID
            track_position: Track position (e.g., 'A1', 'B2')
            
        Returns:
            Analysis data dictionary or None if not found
        """
        # Find release directory
        release_dirs = [d for d in self.library_path.iterdir() 
                       if d.is_dir() and d.name.startswith(f"{release_id}_")]
        
        if not release_dirs:
            logger.error(f"Release directory not found for ID: {release_id}")
            return None
        
        release_dir = release_dirs[0]
        analysis_file = release_dir / f"{track_position}.json"
        
        if not analysis_file.exists():
            logger.error(f"Analysis file not found: {analysis_file}")
            return None
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify required features exist
            if 'tonal' not in data or 'hpcp' not in data['tonal']:
                logger.error(f"HPCP features not found in analysis file: {analysis_file}")
                return None
            
            logger.debug(f"Loaded analysis data for {release_id}:{track_position}")
            return data
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading analysis file {analysis_file}: {e}")
            return None
    
    def _extract_all_features(self, analysis_data: Dict) -> Optional[Dict]:
        """
        Extract all relevant features from analysis data.
        
        Args:
            analysis_data: Essentia analysis data
            
        Returns:
            Dictionary of extracted features or None if extraction fails
        """
        try:
            features = {}
            
            # HPCP features for cover song similarity
            tonal = analysis_data.get('tonal', {})
            if 'hpcp' in tonal:
                features['hpcp_mean'] = np.array(tonal['hpcp']['mean'])
                features['hpcp_std'] = np.array(tonal['hpcp']['stdev'])
            
            # Chroma features (if available)
            if 'thpcp' in tonal:
                features['thpcp'] = np.array(tonal['thpcp'])
            
            # Key information
            features['key_temperley'] = tonal.get('key_temperley', {})
            features['key_krumhansl'] = tonal.get('key_krumhansl', {})
            features['key_edma'] = tonal.get('key_edma', {})
            features['chords_key'] = tonal.get('chords_key', '')
            
            # Rhythm features
            rhythm = analysis_data.get('rhythm', {})
            features['bpm'] = rhythm.get('bpm', 0.0)
            features['danceability'] = rhythm.get('danceability', 0.0)
            features['onset_rate'] = rhythm.get('onset_rate', 0.0)
            features['beats_count'] = rhythm.get('beats_count', 0)
            
            # Low-level features
            lowlevel = analysis_data.get('lowlevel', {})
            features['dynamic_complexity'] = lowlevel.get('dynamic_complexity', 0.0)
            features['average_loudness'] = lowlevel.get('average_loudness', 0.0)
            
            # Spectral features
            if 'spectral_centroid' in lowlevel:
                features['spectral_centroid'] = lowlevel['spectral_centroid'].get('mean', 0.0)
            if 'spectral_rolloff' in lowlevel:
                features['spectral_rolloff'] = lowlevel['spectral_rolloff'].get('mean', 0.0)
            if 'zero_crossing_rate' in lowlevel:
                features['zero_crossing_rate'] = lowlevel['zero_crossing_rate'].get('mean', 0.0)
            
            # Energy and dissonance
            if 'dissonance' in lowlevel:
                features['dissonance'] = lowlevel['dissonance'].get('mean', 0.0)
            
            logger.debug(f"Extracted {len(features)} feature types")
            return features
            
        except (KeyError, TypeError) as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _get_best_key(self, features: Dict) -> str:
        """Get the best key estimate from multiple key detection algorithms."""
        # Priority: Temperley > Krumhansl > EDMA > Chords
        for key_algo in ['key_temperley', 'key_krumhansl', 'key_edma']:
            key_data = features.get(key_algo, {})
            if key_data and isinstance(key_data, dict):
                key = key_data.get('key', '')
                scale = key_data.get('scale', '')
                if key and scale:
                    return f"{key}{'m' if scale.lower() == 'minor' else ''}"
        
        # Fallback to chords key
        chords_key = features.get('chords_key', '')
        return chords_key if chords_key else ''
    
    def _calculate_pitch_shifted_similarity(self, hpcp1: np.ndarray, hpcp2: np.ndarray, 
                                          pitch_range_percent: float = 8.0) -> Tuple[float, int]:
        """
        Calculate similarity with pitch shifting using cross-correlation.
        
        Based on Essentia cover song tutorial approach.
        
        Args:
            hpcp1: HPCP vector for first track
            hpcp2: HPCP vector for second track  
            pitch_range_percent: Pitch shift range in percent (8% or 16%)
            
        Returns:
            Tuple of (max_similarity, best_shift_semitones)
        """
        # Convert pitch range to semitones (approximately)
        # 8% ≈ 1.4 semitones, 16% ≈ 2.8 semitones
        max_shift_semitones = int(np.ceil(pitch_range_percent * 0.175))
        
        # HPCP typically has 36 bins (3 bins per semitone)
        bins_per_semitone = len(hpcp1) // 12
        max_shift_bins = max_shift_semitones * bins_per_semitone
        
        max_similarity = -1.0
        best_shift = 0
        
        # Test different pitch shifts
        for shift_bins in range(-max_shift_bins, max_shift_bins + 1):
            # Circular shift of HPCP2
            hpcp2_shifted = np.roll(hpcp2, shift_bins)
            
            # Calculate cosine similarity
            similarity = np.dot(hpcp1, hpcp2_shifted) / (
                np.linalg.norm(hpcp1) * np.linalg.norm(hpcp2_shifted)
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_shift = shift_bins // bins_per_semitone
        
        return max_similarity, best_shift
    
    def _calculate_chroma_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate chroma-based similarity using THPCP features.
        
        Args:
            features1: Features from first track
            features2: Features from second track
            
        Returns:
            Chroma similarity score (0.0-1.0)
        """
        if 'thpcp' not in features1 or 'thpcp' not in features2:
            return 0.0
        
        thpcp1 = features1['thpcp']
        thpcp2 = features2['thpcp']
        
        if len(thpcp1) == 0 or len(thpcp2) == 0:
            return 0.0
        
        # Simple cosine similarity between normalized THPCP vectors
        thpcp1_norm = thpcp1 / (np.linalg.norm(thpcp1) + 1e-10)
        thpcp2_norm = thpcp2 / (np.linalg.norm(thpcp2) + 1e-10)
        
        similarity = np.dot(thpcp1_norm, thpcp2_norm)
        return max(0.0, min(1.0, similarity))
    
    def _calculate_combined_similarity(self, features1: Dict, features2: Dict, 
                                     pitch_range_percent: float = 8.0) -> Dict:
        """
        Calculate combined similarity using multiple features.
        
        Args:
            features1: Features from first track
            features2: Features from second track
            pitch_range_percent: Pitch shift range for HPCP
            
        Returns:
            Dictionary with various similarity scores
        """
        similarities = {}
        
        # HPCP similarity (cover song detection)
        if 'hpcp_mean' in features1 and 'hpcp_mean' in features2:
            hpcp_sim, pitch_shift = self._calculate_pitch_shifted_similarity(
                features1['hpcp_mean'], features2['hpcp_mean'], pitch_range_percent
            )
            similarities['hpcp_similarity'] = hpcp_sim
            similarities['pitch_shift_semitones'] = pitch_shift
        else:
            similarities['hpcp_similarity'] = 0.0
            similarities['pitch_shift_semitones'] = 0
        
        # Chroma similarity
        similarities['chroma_similarity'] = self._calculate_chroma_similarity(features1, features2)
        
        # Key compatibility
        key1 = self._get_best_key(features1)
        key2 = self._get_best_key(features2)
        similarities['key_compatibility'] = CamelotWheel.calculate_harmonic_compatibility(key1, key2)
        similarities['key1'] = key1
        similarities['key2'] = key2
        similarities['camelot1'] = CamelotWheel.get_camelot_code(key1)
        similarities['camelot2'] = CamelotWheel.get_camelot_code(key2)
        
        # Rhythm similarity
        if features1.get('bpm', 0) > 0 and features2.get('bpm', 0) > 0:
            bpm_diff = abs(features1['bpm'] - features2['bpm'])
            # Similar BPM = high similarity, normalize by max expected difference (100 BPM)
            similarities['bpm_similarity'] = max(0.0, 1.0 - bpm_diff / 100.0)
        else:
            similarities['bpm_similarity'] = 0.0
        
        # Danceability similarity
        dance1 = features1.get('danceability', 0.0)
        dance2 = features2.get('danceability', 0.0)
        similarities['danceability_similarity'] = 1.0 - abs(dance1 - dance2)
        
        # Energy/loudness similarity
        loud1 = features1.get('average_loudness', 0.0)
        loud2 = features2.get('average_loudness', 0.0)
        similarities['energy_similarity'] = 1.0 - min(1.0, abs(loud1 - loud2))
        
        # Combined weighted score
        weights = {
            'hpcp_similarity': 0.4,      # Main cover song detection
            'chroma_similarity': 0.2,    # Additional tonal similarity
            'key_compatibility': 0.2,    # Harmonic compatibility
            'bpm_similarity': 0.1,       # Rhythm compatibility
            'danceability_similarity': 0.05,  # Groove similarity
            'energy_similarity': 0.05    # Energy level similarity
        }
        
        combined_score = sum(
            similarities.get(feature, 0.0) * weight 
            for feature, weight in weights.items()
        )
        similarities['combined_similarity'] = combined_score
        
        return similarities
    
    def get_available_releases(self) -> List[Dict[str, str]]:
        """
        Get list of available releases with analysis data.
        
        Returns:
            List of dictionaries with 'id', 'title', and 'tracks' information
        """
        releases = []
        
        for release_dir in self.library_path.iterdir():
            if not release_dir.is_dir():
                continue
                
            # Extract release ID from directory name
            dir_name = release_dir.name
            if '_' not in dir_name:
                continue
                
            release_id = dir_name.split('_')[0]
            release_title = '_'.join(dir_name.split('_')[1:])
            
            # Check for metadata
            metadata_file = release_dir / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    release_title = metadata.get('title', release_title)
                except:
                    pass
            
            # Find available tracks with analysis
            tracks = []
            for analysis_file in release_dir.glob('*.json'):
                if analysis_file.name == 'metadata.json':
                    continue
                    
                track_position = analysis_file.stem
                # Verify this is a valid track analysis file
                try:
                    with open(analysis_file, 'r') as f:
                        data = json.load(f)
                    if 'tonal' in data and 'hpcp' in data['tonal']:
                        tracks.append(track_position)
                except:
                    continue
            
            if tracks:
                releases.append({
                    'id': release_id,
                    'title': release_title,
                    'tracks': sorted(tracks)
                })
        
        return sorted(releases, key=lambda x: x['title'])
    
    def get_track_info(self, release_id: str, track_position: str) -> Optional[Dict[str, str]]:
        """
        Get track information from metadata.
        
        Args:
            release_id: Discogs release ID
            track_position: Track position
            
        Returns:
            Dictionary with track info or None if not found
        """
        release_dirs = [d for d in self.library_path.iterdir() 
                       if d.is_dir() and d.name.startswith(f"{release_id}_")]
        
        if not release_dirs:
            return None
        
        metadata_file = release_dirs[0] / 'metadata.json'
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Find track in tracklist
            for track in metadata.get('tracklist', []):
                if track.get('position') == track_position:
                    return {
                        'title': track.get('title', ''),
                        'artist': track.get('artist', ''),
                        'duration': track.get('duration', ''),
                        'position': track_position
                    }
        except:
            pass
        
        return {'position': track_position, 'title': '', 'artist': '', 'duration': ''}
    
    def find_similar_tracks(self, reference_release_id: str, reference_track_position: str,
                           algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COMBINED_FEATURES,
                           pitch_range_percent: float = 8.0, 
                           min_similarity: float = 0.7,
                           max_results: int = 20,
                           harmonic_filter: bool = False) -> List[Dict]:
        """
        Find tracks similar to the reference track.
        
        Args:
            reference_release_id: Reference release ID
            reference_track_position: Reference track position
            algorithm: Similarity algorithm to use
            pitch_range_percent: Pitch shift tolerance (8% or 16%)
            min_similarity: Minimum similarity threshold (0.0-1.0)
            max_results: Maximum number of results to return
            harmonic_filter: If True, only show harmonically compatible tracks
            
        Returns:
            List of similar tracks with similarity scores
        """
        logger.info(f"Finding similar tracks to {reference_release_id}:{reference_track_position}")
        logger.info(f"Pitch range: ±{pitch_range_percent}%, Min similarity: {min_similarity}")
        
        # Load reference track analysis
        ref_analysis = self._load_track_analysis(reference_release_id, reference_track_position)
        if ref_analysis is None:
            logger.error("Could not load reference track analysis")
            return []
        
        ref_features = self._extract_all_features(ref_analysis)
        if ref_features is None:
            logger.error("Could not extract features from reference track")
            return []
        
        # Get reference track info
        ref_info = self.get_track_info(reference_release_id, reference_track_position)
        logger.info(f"Reference: {ref_info.get('artist', '')} - {ref_info.get('title', '')}")
        
        similar_tracks = []
        releases = self.get_available_releases()
        
        total_tracks = sum(len(release['tracks']) for release in releases)
        logger.info(f"Comparing against {total_tracks} tracks...")
        
        processed = 0
        for release in releases:
            for track_position in release['tracks']:
                # Skip self-comparison
                if (release['id'] == reference_release_id and 
                    track_position == reference_track_position):
                    continue
                
                # Load comparison track
                comp_analysis = self._load_track_analysis(release['id'], track_position)
                if comp_analysis is None:
                    continue
                
                comp_features = self._extract_all_features(comp_analysis)
                if comp_features is None:
                    continue
                
                # Calculate similarity based on selected algorithm
                if algorithm == SimilarityAlgorithm.HPCP_CROSS_CORRELATION:
                    if 'hpcp_mean' not in ref_features or 'hpcp_mean' not in comp_features:
                        continue
                    similarity, pitch_shift = self._calculate_pitch_shifted_similarity(
                        ref_features['hpcp_mean'], comp_features['hpcp_mean'], pitch_range_percent
                    )
                    similarity_data = {
                        'similarity': similarity,
                        'pitch_shift_semitones': pitch_shift,
                        'algorithm': 'HPCP Cross-Correlation'
                    }
                elif algorithm == SimilarityAlgorithm.CHROMA_SIMILARITY:
                    similarity = self._calculate_chroma_similarity(ref_features, comp_features)
                    similarity_data = {
                        'similarity': similarity,
                        'pitch_shift_semitones': 0,
                        'algorithm': 'Chroma Similarity'
                    }
                elif algorithm == SimilarityAlgorithm.HARMONIC_KEY_MATCHING:
                    ref_key = self._get_best_key(ref_features)
                    comp_key = self._get_best_key(comp_features)
                    similarity = CamelotWheel.calculate_harmonic_compatibility(ref_key, comp_key)
                    similarity_data = {
                        'similarity': similarity,
                        'pitch_shift_semitones': 0,
                        'algorithm': 'Harmonic Key Matching',
                        'ref_key': ref_key,
                        'comp_key': comp_key,
                        'ref_camelot': CamelotWheel.get_camelot_code(ref_key),
                        'comp_camelot': CamelotWheel.get_camelot_code(comp_key)
                    }
                else:  # COMBINED_FEATURES
                    similarities = self._calculate_combined_similarity(ref_features, comp_features, pitch_range_percent)
                    similarity = similarities['combined_similarity']
                    similarity_data = similarities.copy()
                    similarity_data['algorithm'] = 'Combined Features'
                
                # Apply harmonic filter if requested
                if harmonic_filter and algorithm != SimilarityAlgorithm.HARMONIC_KEY_MATCHING:
                    ref_key = self._get_best_key(ref_features)
                    comp_key = self._get_best_key(comp_features)
                    key_compatibility = CamelotWheel.calculate_harmonic_compatibility(ref_key, comp_key)
                    if key_compatibility < 0.8:  # Not harmonically compatible
                        continue
                
                if similarity >= min_similarity:
                    track_info = self.get_track_info(release['id'], track_position)
                    
                    track_result = {
                        'release_id': release['id'],
                        'release_title': release['title'],
                        'track_position': track_position,
                        'track_title': track_info.get('title', ''),
                        'track_artist': track_info.get('artist', ''),
                        'track_duration': track_info.get('duration', ''),
                        'similarity': similarity,
                        'pitch_shift_semitones': similarity_data.get('pitch_shift_semitones', 0),
                        'pitch_shift_percent': similarity_data.get('pitch_shift_semitones', 0) * 5.7,
                        'algorithm': similarity_data.get('algorithm', ''),
                        'bpm': comp_features.get('bpm', 0.0),
                        'danceability': comp_features.get('danceability', 0.0),
                        'key': self._get_best_key(comp_features),
                        'camelot': CamelotWheel.get_camelot_code(self._get_best_key(comp_features))
                    }
                    
                    # Add algorithm-specific data
                    if algorithm == SimilarityAlgorithm.COMBINED_FEATURES:
                        track_result.update({
                            'hpcp_similarity': similarity_data.get('hpcp_similarity', 0.0),
                            'chroma_similarity': similarity_data.get('chroma_similarity', 0.0),
                            'key_compatibility': similarity_data.get('key_compatibility', 0.0),
                            'bpm_similarity': similarity_data.get('bpm_similarity', 0.0),
                            'danceability_similarity': similarity_data.get('danceability_similarity', 0.0),
                        })
                    
                    similar_tracks.append(track_result)
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"Processed {processed}/{total_tracks} tracks...")
        
        # Sort by similarity (descending)
        similar_tracks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit results
        similar_tracks = similar_tracks[:max_results]
        
        logger.info(f"Found {len(similar_tracks)} similar tracks")
        return similar_tracks
    
    def export_results(self, results: List[Dict], output_file: str):
        """
        Export similarity results to CSV file.
        
        Args:
            results: List of similarity results
            output_file: Output CSV file path
        """
        if not results:
            logger.warning("No results to export")
            return
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Results exported to: {output_file}")


def main():
    """Command-line interface for testing the similarity analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cover Song Similarity Analyzer')
    parser.add_argument('--library-path', 
                       default='/home/ffx/.cache/discogsLibary/discogsLib',
                       help='Path to Discogs library directory')
    parser.add_argument('--release-id', required=True,
                       help='Reference release ID')
    parser.add_argument('--track-position', required=True,
                       help='Reference track position (e.g., A1, B2)')
    parser.add_argument('--pitch-range', type=float, default=8.0,
                       choices=[8.0, 16.0],
                       help='Pitch range percentage (8 or 16)')
    parser.add_argument('--min-similarity', type=float, default=0.7,
                       help='Minimum similarity threshold (0.0-1.0)')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Maximum number of results')
    parser.add_argument('--output', 
                       help='Output CSV file for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CoverSongSimilarityAnalyzer(args.library_path)
    
    # Find similar tracks
    results = analyzer.find_similar_tracks(
        args.release_id,
        args.track_position,
        args.pitch_range,
        args.min_similarity,
        args.max_results
    )
    
    # Display results
    print(f"\n📊 Found {len(results)} similar tracks:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['similarity']:.3f} | "
              f"{result['track_artist']} - {result['track_title']} "
              f"({result['release_title']}) "
              f"[{result['track_position']}]")
        if result['pitch_shift_semitones'] != 0:
            print(f"     Pitch shift: {result['pitch_shift_semitones']:+d} semitones "
                  f"({result['pitch_shift_percent']:+.1f}%)")
    
    # Export if requested
    if args.output:
        analyzer.export_results(results, args.output)


if __name__ == '__main__':
    main()