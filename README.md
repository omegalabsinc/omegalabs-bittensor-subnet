<div align="center">

# OMEGA Labs Bittensor Subnet: The World's Largest Decentralized AGI Multimodal Dataset <!-- omit in toc -->
[![OMEGA](/docs/galacticlandscape.png)](https://omegatron.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Be, and it becomes ... <!-- omit in toc -->

</div>

---
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Contributing](#contributing)
- [License](#license)

---
## Introduction

Welcome to the OMEGA Labs Bittensor subnet, a groundbreaking initiative that aims to create the world's largest decentralized multimodal dataset for accelerating Artificial General Intelligence (AGI) research and development. Our mission is to democratize access to a vast and diverse dataset that captures the landscape of human knowledge and creation, empowering researchers and developers to push the boundaries of AGI.

By harnessing the power of the Bittensor network and a global community of miners and validators, we are building a dataset that surpasses the scale and diversity of existing resources. With over 1 million hours of footage and 30 million+ 2-minute video clips, the OMEGA Labs dataset will enable the development of powerful AGI models and transform various industries.


## Key Features

- 🌍 **Unparalleled Scale and Diversity**: 1 million+ hours of footage, 30 million+ video clips, covering 50+ scenarios and 15,000+ action phrases.
- 🧠 **Latent Representations**: Leveraging state-of-the-art models to translate video components into a unified latent space for efficient processing.
- 💰 **Incentivized Data Collection**: Rewarding miners for contributing high-quality, diverse, and novel videos through a decentralized network.
- 🤖 **Empowering Digital Agents**: Enabling the development of intelligent agents that can navigate complex workflows and assist users across platforms.
- 🎮 **Immersive Gaming Experiences**: Facilitating the creation of realistic gaming environments with rich physics and interactions.

## Miner and Validator Functionality

### Miner

- Performs a simple search on YouTube and retrieves 8 videos at a time.
- Provides a certain clip range (maximum of 2 minutes) and a description (catch) which includes the title, tags, and description of the video.
- Obtains the ImageBind embeddings for the video, audio, and caption.
- Returns the video ID, caption, ImageBind embeddings (video, audio, caption embeddings), and start and end times for the clips (maximum of 2 minutes).

### Validator

- Takes the received videos from the miners and randomly selects one video for validation.
- Computes the ImageBind embeddings for all three modalities (video, audio, caption) of the selected video.
- Compares the quality of the embeddings to ensure they are consistent with the miner's submissions.
- If the selected video passes the validation, assumes all eight videos from the miner are valid.
- Scores the videos based on relevance, novelty, and detail richness:
  - Relevance: Calculated using cosine similarity between the topic embedding and each of the eight videos.
  - Novelty: For each video, finds the closest video in the Pinecone index and computes 1 - similarity.
    - Potential issue: Choosing the second most similar video instead of the most similar one.
  - Detail Richness: Determined by the cosine similarity between the text and video embeddings.
- Collects 1024 validated video entries and pushes them to Hugging Face as a file, which is then concatenated.
  - If a miner submits too frequently, the validator may increase the file threshold accumulation limit.
  - If the API needs to shut down for any reason, it will submit the remaining validated entries.

## Roadmap

### Phase 1: Foundation (Q1 2024)
- [x] Launch OMEGA Labs subnet on Bittensor testnet
- [ ] Reach 100,000 hours of footage and 3 million video clips

### Phase 2: Expansion (Q2 2024)
- [ ] Reach 500,000 hours of footage and 15 million video clips
- [ ] Train and demo any-to-any models on the dataset
- [ ] Build synthetic data pipelines to enhance dataset quality
- [ ] Publish a research paper on the Bittensor-powered Ω AGI dataset
- [ ] Expand into running inference for state-of-the-art any-to-any multimodal models

### Phase 3: Refinement (Q3 2024)
- [ ] Reach 1 million+ hours of footage and 30 million+ video clips
- [ ] Use the dataset to train powerful unified representation models
- [ ] Fine-tune any-to-any models for advanced audio-video synchronized generation
- [ ] Open up an auctioning page for companies and groups to bid on validation topics using various currencies (in addition to TAO)
- [ ] Develop state-of-the-art video processing models for applications such as:
  - Transcription
  - Motion analysis
  - Object detection and tracking
  - Emotion recognition

### Phase 4: Application (Q4 2024)
- [ ] Train desktop & mobile action prediction models on the dataset
- [ ] Develop cross-platform digital agents MVP

### Phase 5: Democratization (Q1 2025)
- [ ] Generalize the subnet for miners to upload videos from any data source
- [ ] Incentivize people to record and label their own data using non-deep learning approaches

## Running Miners and Validators
### Running a Miner
#### Requirements
- Python 3.8+
- Pip
- GPU with at least 12 GB of VRAM or 24 GB if you'd like to run a local LLM
- If running on runpod, `runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04` is a good base template.

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-bittensor-subnet.git
cd omegalabs-bittensor-subnet
```
2. Install ffmpeg. If you're on Ubuntu, just run: `apt-get -y update && apt-get install -y ffmpeg`.
3. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
4. Next, install the `omega` package: `pip install -e .`

#### Run with PM2
```bash
pm2 start neurons/miner.py --name omega-miner -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --blacklist.force_validator_permit
```

#### Tips for Better Incentive
The subnet has become quite competitive, and the basic miner template is no longer sufficient to earn good emissions and avoid deregistration. Here are some tips to consider improving your miner:
1. Use proxies or frequently change your pod.
  a) We've heard good things about [Storm Proxies](https://stormproxies.com/).
2. Make sure your videos are unique. You can de-duplicate your collected video with this [video ID index](https://huggingface.co/datasets/jondurbin/omega-multimodal-ids) graciously offered by Jon, one of the miners on the OMEGA subnet.
3. Improve the descriptions you are submitting alongside your uploaded videos. You can try doing this by using video captioning models or incorporating the transcript. Lots of experimentation room here.
4. You can use the `check_score` endpoint that we offer to check your score breakdown. See [this gist](https://gist.github.com/salmanshah1d/f5a8e83cb4af6444ffdef4325a59b489).

#### Common Troubleshooting Tips
1. If you've been running for several minutes and have not received any requests, make sure your port is open to receiving requests. You can try hitting your IP and port with `curl`. If you get no response, it means your port is not open.
2. You can use our [validator logs W&B](https://wandb.ai/omega-labs/omega-sn24-validator-logs) to see how your miner is scoring in practice.

### Running a Validator
#### Requirements
- Python 3.8+
- Pip
- If running on runpod, `runpod/base:0.5.1-cpu` is a good base template.

#### Setup
1. To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/omegalabsinc/omegalabs-bittensor-subnet.git
cd omegalabs-bittensor-subnet
```
2. Install ffmpeg. If you used the runpod image recommended above, ffmpeg is already installed. Otherwise, if you're on Ubuntu, just run: `apt-get -y update && apt-get install -y ffmpeg`.
3. Install pm2 if you don't already have it: [pm2.io](https://pm2.io/docs/runtime/guide/installation/).
4. Next, install the `omega` package: `pip install -e .`

#### Run auto-updating validator with PM2 (recommended)
```bash
pm2 start auto_updating_validator.sh --name omega-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```
Note: you might need to adjust "python" to "python3" within the `neurons/auto_updating_validator.sh` depending on your preferred system python.

#### Run basic validator with PM2
```bash
pm2 start neurons/validator.py --name omega-validator -- \
    --netuid {netuid} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --axon.port {port} \
    --logging.trace
```

## Contributing

We believe in the power of community and collaboration. Join us in building the world's largest decentralized multimodal dataset for AGI research! Whether you're a researcher, developer, or data enthusiast, there are many ways to contribute:

- Submit high-quality videos and annotations
- Develop and improve data validation and quality control mechanisms
- Train and fine-tune models on the dataset
- Create applications and tools that leverage the dataset
- Provide feedback and suggestions for improvement

To get started, please see our [contribution guidelines](./CONTRIBUTING.md) and join our vibrant community on [Discord](https://discord.gg/opentensor).

## License

The OMEGA Labs Bittensor subnet is released under the [MIT License](./LICENSE).

---

🌟 Together, let's revolutionize AGI research and unlock the full potential of multimodal understanding! 🌟
</div>
