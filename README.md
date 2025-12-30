# Score Before You Speak
<a href="https://arxiv.org/abs/2508.06886"><img src="https://img.shields.io/badge/arXiv-2508.06886-red"></a>
<a href="https://arpita2512.github.io/score_before_you_speak/"><img src="https://img.shields.io/badge/Project%20Page-online-green"></a>
<a href="https://huggingface.co/collections/Arpita1/score-before-you-speak-68a4a9f2b2598c476d35b723"><img src="https://img.shields.io/badge/%F0%9F%A4%97Hugging%20Face-Models-blue"></a>
<br>

Code repository for the ECAI 2025 paper **Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores**

## Installation

Experiments were run on two different setups (see supplementary material). Please use *persona.yml* for DialoGPT and *llama.yml* for Llama 3.1. 

```
git clone https://github.com/arpita2512/score_before_you_speak.git
cd score_before_you_speak
conda env create -f <environment-name>.yml
```

## Data

PERSONA-CHAT and ConvAI2 are avaiable through [ParlAI](https://github.com/facebookresearch/ParlAI). We use the `<split>_self_original.txt` setting for both datasets.

## Data Pre-processing

### Pos-tagging

```
python preprocess\postag.py <path_to_txt_file> # saves pos-tagged file as json
```

### Masking

### Scoring

## Training