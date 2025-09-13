# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OMEGA Labs Bittensor Subnet (SN24) is a decentralized multimodal dataset creation system that incentivizes miners to contribute high-quality video content with embeddings. The subnet creates the world's largest AGI dataset through YouTube video scraping, ImageBind embeddings, and a Focus Videos marketplace for task completion recordings.

## Development Commands

### Installation and Setup
```bash
# Clone and install the omega package
git clone https://github.com/omegalabsinc/omegalabs-bittensor-subnet.git
cd omegalabs-bittensor-subnet
pip install -e .

# Install ffmpeg (required for video processing)
apt-get -y update && apt-get install -y ffmpeg

# Install PM2 for process management
npm install pm2 -g
```

### Running Components
```bash
# Run miner with PM2
pm2 start neurons/miner.py --name omega-miner -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --blacklist.force_validator_permit

# Run validator with PM2 (basic)
pm2 start neurons/validator.py --name omega-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace

# Run auto-updating validator (recommended)
pm2 start auto_updating_validator.sh --name omega-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace

# Run Focus Video purchase script
python purchase_focus_video.py
```

### Development and Testing
```bash
# Code linting and formatting
ruff check .
ruff check . --fix
ruff format .

# Run individual test files (no unified test suite)
python TEST_get_emission.py
python test_audio_dataset.py
python neurons/test_miner.py
python omega/test_audio.py
python validator_api/test_search_and_submit.py
```

## Architecture

### Core Components

**Miners (`neurons/miner.py`)**
- Scrape YouTube videos based on validator queries
- Generate ImageBind embeddings for video, audio, and description
- Submit 8 videos at a time with 2-minute maximum clip length
- Enhance descriptions using LLM augmentation (OpenAI, Local LLM, or None)

**Validators (`neurons/validator.py`)**
- Send search topics to miners from curated topics list
- Validate submissions by spot-checking embeddings
- Score videos on relevance, novelty, and detail richness
- Upload validated batches to Hugging Face dataset
- Handle Focus Videos marketplace integration

**Protocol (`omega/protocol.py`)**
- Defines `VideoMetadata` model with embeddings and metadata
- `Videos` synapse for miner-validator communication
- Structured data exchange for video information

### Key Modules

**ImageBind Integration (`omega/imagebind_wrapper.py`)**
- Wrapper around Meta's ImageBind model for multimodal embeddings
- Processes video, audio, and text into unified embedding space
- Core technology enabling multimodal dataset creation

**Scoring Systems**
- YouTube video scoring: relevance + novelty + detail richness
- Focus Videos scoring: LLM evaluation of task completion quality
- Novelty detection via Pinecone similarity search

**Focus Videos Marketplace**
- Users record task completions via Ω Focus app
- Videos scored and listed in marketplace
- Miners purchase videos to boost their subnet scores
- Transaction verification through Bittensor blockchain

### Directory Structure

- `omega/`: Core subnet package with base classes, protocols, utilities
- `neurons/`: Miner and validator entry points
- `validator_api/`: RESTful API for Focus Videos marketplace
- `scripts/`: Utility scripts for development and operations
- `docs/`: Documentation including setup guides

## Configuration

**Environment Variables**
- Standard Bittensor wallet and network configuration
- W&B integration: `export WANDB_API_KEY=<key>` or use `--wandb.off`

**Hardware Requirements**
- **Miners**: 12+ GB VRAM (24 GB for local LLM), GPU required
- **Validators**: 24+ GB VRAM, GPU required
- Both require Ubuntu 20.04+, SSD storage recommended

**Dependencies**
- Bittensor framework (9.10.1+)
- PyTorch ecosystem for model inference
- ImageBind from Omega Labs fork
- YouTube-dl for video downloading
- Pinecone for similarity search
- HuggingFace for dataset uploads

## Important Notes

- No unified test framework - tests are individual scripts
- Uses ruff for linting/formatting (not pytest, mypy, or other common tools)
- Auto-updating validator pulls latest code on each restart
- Competitive environment requires proxy usage and unique content strategies
- Focus Videos currently represents 5% of miner scoring (planned increase)