# german_french_translation_llm

## Table of Contents
* [Introduction](#introduction)
* [Requirements](#requirements)
* [Setup](#setup)
* [Training and Usage](#training-and-usage)
* [Evaluation](#evaluation)
* [Example Usage](#example-usage)
* [Acknowledgments](#acknowledgments)

## Introduction

This project implements a German-to-French language translation system by fine-tuning a large language model (LLM). We used the `microsoft/phi-2` model and applied parameter-efficient fine-tuning techniques like LoRA to make the training feasible on limited resources (Google Colab with T4 GPU). Evaluation was conducted using SacreBLEU to ensure consistent, standardized performance measurement.

## Requirements

- Python version: 3.10+
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- Accelerate
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes
- Evaluate
- Matplotlib
- Google Colab (with T4 GPU)

Install all dependencies with:
```bash
pip install -q transformers datasets accelerate peft bitsandbytes evaluate matplotlib
```
## Setup

Follow these steps to set up and run the project:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/german_french_translation_llm.git
cd german_french_translation_llm
```
2. Ensure you have Python and all required dependencies installed.
3. Open the Jupyter notebook GenAI.ipynb in Google Colab.
4. Follow the notebook cells to execute each phase of the fine-tuning and evaluation.

## Training and Usage

The project includes the following key components:

- Data Preparation
  - Initially used the Tatoeba dataset (~1,000 German-French pairs) but switched to Europarl for richer linguistic variety and context.
  - Additional synthetic data was created using the fine-tuned model to enhance the training corpus.

- Model and Techniques
  - Base Model: microsoft/phi-2 — chosen for its flexibility and modern architecture.
  
  - Fine-Tuning Approach:
    - LoRA (Low-Rank Adaptation): Efficient fine-tuning by updating only a subset of parameters.
    - 8-bit Quantization: Used BitsAndBytes to compress the model and reduce GPU memory load.
    - Gradient Accumulation & Checkpointing: Enabled larger virtual batch sizes and reduced memory usage.

- Training Phases  
  - Model A: Base model evaluation on the test set.  
  - Model B: Fine-tuned on the benchmark training data.  
  - Model C: Fine-tuned on synthetic data generated from Model A.  
  - Model D: Fine-tuned on the combination of benchmark and synthetic data.
 

## Evaluation
- Evaluation was conducted using SacreBLEU, which ensures consistent scoring with fixed tokenization.
- Beam search was used during generation for improved translation quality.

| Model       | SacreBLEU Score |
|-------------|-----------------|
| Base Model  | 39.54           |
| Model B     | 7.56            |
| Model C     | 6.01            |
| Model D     | 7.62            |

- The fine-tuned models underperformed likely due to limited training epochs and constrained GPU memory. Future improvements can include more epochs, better hyperparameters, and distributed training.

## Example Usage
from transformers import pipeline
```bash
from transformers import pipeline

translator = pipeline("translation", model="path/to/your/final-model")
result = translator("Guten Morgen! Wie geht es dir?")
print(result)

```
## Acknowledgments
This project is powered by HuggingFace Transformers, PEFT, and Evaluate libraries. Special thanks to Google Colab for providing GPU access and to the microsoft/phi-2 model, which enabled efficient fine-tuning using LoRA.
