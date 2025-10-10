# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a model soups playground that implements the paper "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time". The repository contains both the original model-soups implementation and Jupyter notebook demonstrations.

### Dependencies

".venv" is used for virtual environment of this project isolated from others

## Key Commands

## Architecture Overview

### Core Components

- **ModelWrapper** (`utils.py`): Wraps CLIP models with linear classification heads
- **Dataset Classes** (`datasets/`): Handles loading of various ImageNet variants
- **Soup Generation** (`main.py`): Implements uniform and greedy soup algorithms

### Model Soups Concept

The repository implements two main soup strategies:

1. **Uniform Soup**: Simple averaging of all 72 fine-tuned model weights
2. **Greedy Soup**: Iteratively adds models to the soup only if they improve validation accuracy

### Key Files

- `main.py`: Primary script with all soup generation logic
- `utils.py`: Model wrapper and evaluation utilities  
- `datasets/`: Dataset loading for ImageNet variants


## Development Notes

- Models are fine-tuned CLIP ViT-B/32 variants
- Default batch size is 256, adjustable with `--batch-size`
- The repository includes pre-computed results, so plotting can work without running evaluations