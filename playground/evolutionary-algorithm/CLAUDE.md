# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the MergeKit evolutionary algorithm repository, containing tools for merging pre-trained language models using various merging methods including evolutionary optimization. The project uses Python with setuptools and is structured around the main `mergekit` package.

## Key Commands

### Installation and Setup
```bash
# Install in development mode with all features
pip install -e .[evolve,vllm,dev,test]

# Or install with specific feature sets
pip install -e .[evolve]     # For evolutionary merging
pip install -e .[dev]        # For development tools
pip install -e .[test]       # For testing
```

### Main Entry Points
```bash
# Run YAML-based merge
mergekit-yaml path/to/config.yml ./output-dir [--cuda] [--lazy-unpickle]

# Run evolutionary merge optimization  
mergekit-evolve --storage-path /path/to/storage config.yml

# Other specialized tools
mergekit-moe          # Mixture of experts merging
mergekit-tokensurgeon # Tokenizer transplantation
mergekit-extract-lora # LoRA extraction
mergekit-multi        # Multi-stage merging
```

### Development and Testing
```bash
# Run tests
pytest

# Code formatting (if pre-commit is set up)
black .
isort .

# Run specific test
pytest tests/test_basic_merges.py
```

## Architecture Overview

### Core Modules
- **`mergekit/merge.py`**: Main merge orchestration and execution engine
- **`mergekit/config.py`**: Configuration parsing and validation  
- **`mergekit/merge_methods/`**: Implementation of various merge algorithms (linear, TIES, DARE, etc.)
- **`mergekit/evo/`**: Evolutionary optimization system using CMA-ES
- **`mergekit/io/`**: I/O operations including lazy tensor loading and caching
- **`mergekit/architecture/`**: Model architecture definitions and auto-detection

### Evolutionary Merge System (`mergekit/evo/`)
- **`genome.py`**: Parameter space definition and genome encoding/decoding
- **`config.py`**: Evolution configuration with task definitions and validation
- **`strategy.py`**: Different scheduling strategies (pool, buffered, serial)
- **`actors.py`**: Ray-based distributed evaluation actors

### Key Patterns
- Uses lazy tensor loading to handle large models efficiently
- Supports multiple merge methods with pluggable architecture via registry pattern
- Employs Ray for distributed computation in evolutionary merging
- Configuration-driven approach with YAML files for merge specifications
- Supports layer-wise and parameter-wise granular control

## Configuration Files

### Merge Configuration (YAML)
Standard merge configs specify:
- `merge_method`: Algorithm to use (linear, ties, dare_ties, etc.)
- `models` or `slices`: Input models or layer slices
- `parameters`: Method-specific parameters (weights, densities, etc.)
- `base_model`: Base model for task arithmetic methods
- `tokenizer`: Vocabulary handling configuration

### Evolution Configuration (YAML) 
Evolution configs include:
- `genome`: Parameter space definition with models, layer_granularity, filters
- `tasks`: LM evaluation harness tasks with weights and metrics
- Optimization settings like normalize, allow_negative_weights, smooth

## Important Notes

### Virtual Environment
- Always check for existing `.venv` before creating new environments
- Use `.venv` as the standard virtual environment name in project root

### Evolutionary Optimization Ethics
- The system includes warnings against optimizing directly on benchmark test sets
- Use `--i-understand-the-depths-of-the-evils-i-am-unleashing` flag only for legitimate research

### Memory and Storage
- Evolutionary merging can require significant disk space (one fp16 model per GPU)
- Use `--in-memory` flag for faster evaluation but higher memory usage
- Storage path must have sufficient space for model caching

### Ray Configuration
- Supports both single-node and distributed Ray clusters
- Three scheduling strategies available: pool (recommended), buffered, serial
- Pool strategy assigns actors to GPUs and ensures merge/eval on same node