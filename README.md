# Similarity Matrix Analyzer

Advanced music similarity analysis system with pre-computed similarity matrices for instant lookups.

## Features

### 🎵 Multiple Similarity Algorithms
- **HPCP Cross-Correlation** - Cover song detection with pitch shifting
- **Chroma Similarity** - Tonal similarity using THPCP features
- **Harmonic Key Matching** - Camelot wheel compatibility for DJing
- **Combined Features** - Multi-dimensional weighted analysis

### 🚀 Pre-computed Similarity Matrices
- **Instant Results** - No more waiting for similarity computations
- **Sparse Storage** - Efficient storage (1-7 MB for 2000+ tracks)
- **Background Processing** - Build matrices during off-hours
- **Resume Capability** - Continue interrupted computations

### 🎛️ Advanced Features
- **Camelot Wheel Integration** - Harmonic compatibility for DJs
- **Rich Metadata** - BPM, danceability, energy, key information
- **Multiple Formats** - Sparse matrices, HDF5, NPZ storage
- **Configuration-based** - Uses same config as Discogs Label Generator

## Quick Start

### 1. Installation
```bash
# Required Python packages
pip install numpy scipy pandas h5py tqdm

# The system uses existing Essentia analysis data from Discogs Label Generator
```

### 2. Configuration
The system automatically reads the configuration from the Discogs Label Generator:
- Config file: `~/.config/discogsDBLabelGen/discogs.env`
- Similarity data stored in: `{LIBRARY_PATH}/../similarity/`

### 3. Basic Usage

#### Real-time Similarity Analysis
```bash
# Launch GUI for track-by-track analysis
python3 similarity_gui.py
```

#### Build Pre-computed Matrices
```bash
# Build combined features matrix (recommended)
python3 matrix_builder.py --algorithm combined

# Build HPCP matrix for cover song detection
python3 matrix_builder.py --algorithm hpcp

# Build harmonic compatibility matrix
python3 matrix_builder.py --algorithm harmonic
```

#### Matrix-based Instant Lookups
```bash
# Test matrix functionality
python3 similarity_matrix.py
```

## Directory Structure

```
SimilarityMatrixAnalyzer/
├── similarity_analyzer.py     # Core similarity algorithms
├── similarity_gui.py          # Real-time analysis GUI
├── similarity_matrix.py       # Matrix storage and retrieval
├── matrix_builder.py          # Background matrix computation
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── src/
    ├── config.py              # Configuration management
    └── logger.py              # Logging utilities
```

## Data Storage

Similarity data is stored alongside your music library:
```
{LIBRARY_PATH}/../similarity/
├── matrices/
│   ├── hpcp_matrix.npz           # HPCP similarity matrix
│   ├── chroma_matrix.npz         # Chroma similarity matrix
│   ├── combined_matrix.npz       # Combined features matrix
│   └── harmonic_matrix.npz       # Harmonic compatibility matrix
├── metadata/
│   ├── track_index.json          # Track ID mappings
│   └── computation_log.json      # Processing history
└── cache/
    └── build_progress.json       # Resume capability data
```

## Similarity Algorithms

### HPCP Cross-Correlation
- **Purpose**: Cover song detection
- **Features**: Pitch-shifted harmonic analysis
- **Best for**: Finding different versions of the same song
- **Tolerance**: 8% or 16% pitch range

### Chroma Similarity  
- **Purpose**: Tonal similarity
- **Features**: THPCP-based chromagram analysis
- **Best for**: Harmonically similar tracks
- **Speed**: Very fast computation

### Harmonic Key Matching
- **Purpose**: DJ-friendly harmonic compatibility
- **Features**: Camelot wheel mapping
- **Best for**: Seamless mixing and transitions
- **Compatibility**: Perfect (1.0), Adjacent (0.8), or None (0.0)

### Combined Features
- **Purpose**: Multi-dimensional similarity
- **Features**: Weighted combination of:
  - HPCP similarity (40%)
  - Chroma similarity (20%)
  - Key compatibility (20%)
  - BPM similarity (10%)
  - Danceability (5%)
  - Energy similarity (5%)
- **Best for**: Comprehensive music matching

## Matrix Building

### Performance Characteristics
- **Collection size**: 1,887 tracks (example)
- **Matrix size**: 1,887 × 1,887 = 3.6M elements
- **Storage**: 1.7-6.8 MB (depending on format)
- **Build time**: ~15 hours one-time computation
- **Chunk processing**: 200×200 blocks for memory efficiency

### Building Matrices
```bash
# Build with default settings (recommended)
python3 matrix_builder.py --algorithm combined

# Custom chunk size and similarity threshold
python3 matrix_builder.py --algorithm hpcp --chunk-size 100 --min-similarity 0.2

# Resume interrupted computation
python3 matrix_builder.py --algorithm combined  # Resumes automatically

# Start from scratch
python3 matrix_builder.py --algorithm combined --no-resume
```

### Progress Monitoring
The system provides detailed progress tracking:
- Chunks processed vs. total chunks
- Estimated time to completion
- Automatic resume from last completed chunk
- Storage size and sparsity statistics

## API Usage

### Matrix Manager
```python
from similarity_matrix import SimilarityMatrixManager
from similarity_analyzer import SimilarityAlgorithm

# Initialize matrix manager
manager = SimilarityMatrixManager()

# Get similar tracks (instant lookup)
similarities = manager.get_similarities(
    algorithm=SimilarityAlgorithm.COMBINED_FEATURES,
    reference_track_id="13667935:A1",
    min_similarity=0.7,
    max_results=10
)

# Results: [(track_id, similarity_score), ...]
for track_id, score in similarities:
    print(f"{track_id}: {score:.3f}")
```

### Real-time Analysis
```python
from similarity_analyzer import CoverSongSimilarityAnalyzer, SimilarityAlgorithm

# Initialize analyzer
analyzer = CoverSongSimilarityAnalyzer("/path/to/music/library")

# Find similar tracks (real-time computation)
results = analyzer.find_similar_tracks(
    reference_release_id="13667935",
    reference_track_position="A1",
    algorithm=SimilarityAlgorithm.COMBINED_FEATURES,
    min_similarity=0.7,
    harmonic_filter=True  # Only harmonically compatible tracks
)
```

## Camelot Wheel Integration

The system includes complete Camelot wheel support for harmonic mixing:

### Key Features
- **Full mapping**: All major/minor keys → Camelot codes
- **Compatibility rules**: 
  - Same code (1.0) - Perfect match
  - Adjacent numbers (0.8) - Smooth transitions  
  - Inner/outer ring (0.8) - Relative major/minor
  - Non-compatible (0.0) - Clashing keys

### Example Usage
```python
from similarity_analyzer import CamelotWheel

# Get Camelot code for a key
camelot = CamelotWheel.get_camelot_code("Bbm")  # Returns "3A"

# Check compatibility
compatibility = CamelotWheel.calculate_harmonic_compatibility("Bbm", "C")
# Returns 0.0 (not compatible)

# Get harmonically compatible keys
compatible = CamelotWheel.get_harmonic_keys("3A")
# Returns ["3A", "2A", "4A", "3B"]
```

## Performance Benefits

### Matrix vs. Real-time Comparison

| Aspect | Real-time Analysis | Pre-computed Matrix |
|--------|-------------------|-------------------|
| **Initial search** | 15+ minutes | Instant (<1 second) |
| **Subsequent searches** | 15+ minutes each | Instant (<1 second) |
| **Storage overhead** | None | 1-7 MB |
| **Advanced queries** | Not possible | Full collection analysis |
| **Memory usage** | High during computation | Minimal |

### Advanced Queries with Matrices
Once matrices are built, you can perform advanced analyses:
- Find all similar track pairs in collection
- Cluster tracks by similarity
- Generate playlists based on multiple criteria
- Identify outliers and unique tracks
- Create similarity-based recommendations

## Troubleshooting

### Common Issues
- **Matrix file not found**: Run `matrix_builder.py` first
- **Config path errors**: Ensure Discogs Label Generator config exists
- **Memory issues**: Reduce chunk size (e.g., `--chunk-size 100`)
- **Interrupted builds**: System automatically resumes from last chunk

### Storage Cleanup
```bash
# Check storage usage
python3 similarity_matrix.py

# Delete specific matrix
python3 -c "
from similarity_matrix import SimilarityMatrixManager
from similarity_analyzer import SimilarityAlgorithm
manager = SimilarityMatrixManager()
manager.delete_matrix(SimilarityAlgorithm.HPCP_CROSS_CORRELATION)
"
```

## License

Part of the Discogs Record Label Generator project.