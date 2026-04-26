# MASC 515 Assignment 3 - Enhancing microGPT with Modern AI Techniques

## Overview

In this project, I extended the minimal microGPT implementation by integrating several modern techniques used in large language models. The goal is not to fully reproduce industrial implementations, but to demonstrate a clear understanding of the underlying ideas and how they can be incorporated into a simplified model.

The following methods were implemented:

* Gaussian Error Linear Units (GELU)
* Low-Rank Adaptation (LoRA)
* Rotary Position Embedding (RoPE)
* Mixture of Experts (MoE)

All modifications were applied directly within the `microgpt.py` file.

## ----------------

## 1. GELU (Gaussian Error Linear Units)

### Motivation

Traditional activation functions like ReLU apply a hard threshold, which may discard useful information. GELU instead provides a smooth, probabilistic activation that better models uncertainty.

### Mathematical Formulation

GELU is defined as:

GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715x³)))

### Implementation

* Implemented a custom `gelu(x)` function
* Replaced the original ReLU activation with GELU in the model

### Insight

GELU allows small negative inputs to contribute instead of being completely zeroed out, improving gradient flow and model expressiveness.

## ----------------

## 2. LoRA (Low-Rank Adaptation)

### Motivation

Fine-tuning large models is computationally expensive. LoRA reduces this cost by introducing low-rank updates rather than modifying full weight matrices.

### Concept

Instead of updating a full matrix W, LoRA approximates the update as:

W + ΔW ≈ W + A·B

where A and B are low-rank matrices.

### Implementation

* Implemented a simplified `lora(x)` function to simulate low-rank adaptation
* Applied LoRA transformation to model outputs (logits)

### Insight

Although simplified, this demonstrates how additional learnable transformations can be layered on top of existing outputs without modifying the original structure.

## ----------------

## 3. RoPE (Rotary Position Embedding)

### Motivation

Transformers require positional information to understand sequence order. Traditional positional embeddings add vectors, while RoPE encodes position through rotation in vector space.

### Concept

RoPE uses sine and cosine functions to rotate representations:

x' = x·cos(pos) + x·sin(pos)

This allows the model to capture relative position relationships more effectively.

### Implementation

* Implemented `rope(x, pos)` using sine and cosine functions
* Incorporated positional encoding using `pos_id`
* Applied transformation to logits

### Insight

RoPE introduces position-dependent transformations without increasing dimensionality, making it efficient and effective.

## ----------------

## 4. Mixture of Experts (MoE)

### Motivation

Instead of using a single model for all inputs, MoE dynamically selects specialized “experts” to process different inputs.

### Concept

MoE consists of:

* Multiple expert functions
* A gating function that selects which expert to use

### Implementation

* Defined two expert functions (`expert1`, `expert2`)
* Implemented a gating function `moe(x)` that selects an expert based on input value
* Applied MoE transformation to logits

### Insight

This simplified MoE demonstrates conditional computation, where different parts of the model handle different inputs.

## ----------------

## Integration Pipeline

The final computation pipeline is:

```python
logits = gpt(...)
logits = [lora(l) for l in logits]
logits = [rope(l, pos_id) for l in logits]
logits = [moe(l) for l in logits]
```

This shows how multiple techniques can be layered sequentially to enhance model behavior.

## ----------------

## Conclusion

This project demonstrates how core ideas from modern large language models can be integrated into a minimal implementation. While simplified, each component reflects the fundamental principle behind the original method.

Through this process, I gained a deeper understanding of:

* Activation functions and their impact on learning
* Efficient model adaptation techniques
* Positional encoding strategies
* Conditional computation with multiple experts

## ----------------

## References

* GELU: https://arxiv.org/abs/1606.08415
* LoRA: https://arxiv.org/abs/2106.09685
* RoPE: https://arxiv.org/abs/2104.09864
* MoE: https://huggingface.co/blog/moe#a-brief-history-of-moes
