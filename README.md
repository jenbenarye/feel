---
title: Feel
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.8.0
app_file: app/app.py
pinned: false
---

# Feel 

This is a project to create a continuous training application.

Platform being developed at MIT in collaboration with HuggingFace. Aimed at improving performance of existing Large Language Models through real-time human feedback loop.

This repository hosts the development of an automated RLHF platform for Hugging Face, where the community can provide real-time feedback on language models. The feedback is automatically integrated into an RLHF pipeline to continuously fine-tune and improve the models.

## What is Feel?

A community-driven project to improve Multilingual Vision-Language Models (VLMs). Leverages feedback from users and automated RLHF pipelines to continuously improve model performance.

## Why Feel?

Feel is a platform that enables the community to provide real-time feedback on language models. The feedback is automatically integrated into an RLHF pipeline to continuously fine-tune and improve the models.

## Repository Structure

The repository is organized as follows:

```
ml/                # Directory for machine learning code
├── README.md      # Dataset schema and project structure
├── data/          # Directory for dataset files
├── models/        # Directory for model files
app/               # Directory for application code
├── app.py         # Main application file
```

## Installation

The repository uses `uv` for managing virtual environments. To install `uv`, go [here](https://docs.astral.sh/uv/getting-started/installation/).


To install the required dependencies, run the following commands:

### ML Dependencies

```bash
uv install ml
```

### App Dependencies

```bash
uv install app
```
