# AMP-Latent-Diffusion

## Dual-Branch Transformer-VAE and Latent Diffusion Framework for Novel Antimicrobial Peptide Generation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Research](https://img.shields.io/badge/Research-Antimicrobial%20Peptides-purple.svg)](https://github.com/anshul-3000/AMP-Latent-Diffusion)

A deep generative framework for **computational antimicrobial peptide (AMP) discovery** that combines Transformer-based Variational Autoencoders (Transformer-VAE), latent diffusion modeling, branch-specific LSTM sequence decoders, AMP activity prediction, hemolysis screening, novelty analysis, and physicochemical characterization.

The framework separately models **short peptides (5–40 amino acids)** and **long peptides (41–60 amino acids)** to explore different regions of peptide sequence space.

> **Research status:** This repository contains a computational research implementation. Generated peptides are computationally predicted candidates and require experimental validation before biological or therapeutic conclusions can be made.

---

# Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Key Contributions](#key-contributions)
- [Framework Overview](#framework-overview)
- [Datasets](#datasets)
- [Sequence Representation](#sequence-representation)
- [AMP Classification](#amp-classification)
- [Hemolysis Classification](#hemolysis-classification)
- [Dual-Branch Generative Framework](#dual-branch-generative-framework)
- [Transformer-VAE](#transformer-vae)
- [Latent Diffusion](#latent-diffusion)
- [LSTM Sequence Decoding](#lstm-sequence-decoding)
- [Candidate Screening](#candidate-screening)
- [Novelty Analysis](#novelty-analysis)
- [Physicochemical Analysis](#physicochemical-analysis)
- [Amino Acid Composition Analysis](#amino-acid-composition-analysis)
- [Embedding-Space Analysis](#embedding-space-analysis)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Computational Requirements](#computational-requirements)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Citation](#citation)
- [Data Sources](#data-sources)
- [License](#license)
- [Author](#author)

---

# Overview

Antimicrobial peptides (AMPs) represent an important class of bioactive peptides with potential applications in addressing the growing problem of antimicrobial resistance.

This project investigates a computational framework for **de novo AMP sequence generation** using deep generative modeling.

The central idea is to learn a continuous latent representation of AMP sequences using a Transformer-based Variational Autoencoder and subsequently model the latent distribution using diffusion.

Rather than performing generation directly in discrete amino-acid sequence space, the framework performs generation in a learned continuous latent space and subsequently reconstructs peptide sequences using sequence decoders.

The complete workflow combines generation with downstream computational screening for antimicrobial activity, predicted hemolysis, sequence novelty, and physicochemical characteristics.

---

# Research Motivation

Conventional experimental AMP discovery can require substantial time, resources, and laboratory screening.

Deep generative models provide a computational alternative by learning patterns from known peptide sequences and generating candidate sequences that can subsequently be prioritized for experimental validation.

However, a useful computational AMP discovery system should not focus only on sequence generation.

Generated candidates should also be evaluated for:

- Predicted antimicrobial activity
- Predicted hemolytic potential
- Novelty relative to training data
- Physicochemical properties
- Amino-acid composition
- Representation-space distribution

This project therefore integrates generation and multi-stage computational screening into a single workflow.

---

# Key Contributions

The framework incorporates the following components:

1. **Dual-branch peptide generation** for short and long peptide regimes.

2. **Transformer-VAE representation learning** for peptide sequences.

3. **Latent diffusion modeling** for generation in continuous latent space.

4. **Branch-specific LSTM decoders** for sequence reconstruction from generated latent representations.

5. **AMP activity prediction** for computational screening of generated sequences.

6. **Hemolysis prediction** to identify candidates with lower predicted hemolytic potential.

7. **Sequence-level novelty analysis** relative to the AMP training dataset.

8. **Physicochemical characterization** of generated candidates.

9. **Amino-acid composition analysis** of generated peptide populations.

10. **UMAP embedding visualization** to examine the representation-space relationship between known and generated peptides.

---

# Framework Overview

The complete computational workflow is:

```text
                         ┌─────────────────────┐
                         │   DRAMP AMP Data    │
                         └──────────┬──────────┘
                                    │
                                    │
                         ┌──────────▼──────────┐
                         │  AMP Preprocessing  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    ESM-2 Embedding  │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
       ┌───────────────────┐                 ┌────────────────────┐
       │   AMP Classifier  │                 │ Hemolysis Classifier│
       └─────────┬─────────┘                 └──────────┬─────────┘
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Length-based Split  │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ Short Branch      │                │ Long Branch       │
        │ 5–40 aa           │                │ 41–60 aa          │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  ▼                                    ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ Transformer-VAE   │                │ Transformer-VAE   │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  ▼                                    ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ Latent Vectors    │                │ Latent Vectors    │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  ▼                                    ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ Latent Diffusion  │                │ Latent Diffusion  │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  ▼                                    ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ Generated Latents │                │ Generated Latents │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  ▼                                    ▼
        ┌───────────────────┐                ┌───────────────────┐
        │ LSTM Decoder      │                │ LSTM Decoder      │
        │ 5–40 aa           │                │ 41–60 aa          │
        └─────────┬─────────┘                └─────────┬─────────┘
                  │                                    │
                  └────────────────┬───────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Generated Peptides  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ AMP Screening       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Hemolysis Screening │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Novelty Analysis    │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
             Physicochemical   Composition       UMAP
                Analysis        Analysis       Analysis


      ▼
Latent Diffusion
      │
      ▼
Generated Latents
      │
      ▼
LSTM Decoder
      │
      ▼
Short Peptide Candidates
```

Relevant notebooks:

```
new_diffusion/
│
├── prepare_sequences.ipynb
│
├── extract_latents/
│   └── extract_latents40.ipynb
│
└── final_model/
    ├── transformer_vae_len40.ipynb
    ├── diffusion__len40_v2.ipynb
    └── lstm_decoder_len40.ipynb
```

---

# Long-Peptide Branch

## Length Range

```
41–60 amino acids
```

The long branch follows:

```
AMP Sequences
      │
      ▼
Transformer-VAE
      │
      ▼
Latent Representation
      │
      ▼
Latent Diffusion
      │
      ▼
Generated Latents
      │
      ▼
LSTM Decoder
      │
      ▼
Long Peptide Candidates
```

Relevant notebooks:

```
new_diffusion/
│
├── prepare_sequences_len60_v2.ipynb
│
├── extract_latents/
│   └── extract_latents60.ipynb
│
└── final_model/
    ├── transformer_vae_len60.ipynb
    ├── diffusion__len60_v2.ipynb
    └── lstm_decoder_len60.ipynb
```

---

# Transformer-VAE

The Transformer-VAE is used to learn continuous latent representations of peptide sequences.

Conceptually:

```
```

```
Peptide Sequence
       │
       ▼
Transformer Encoder
       │
       ▼
Latent Distribution
       │
       ├───────────────┐
       ▼               ▼
       μ               σ
       │               │
       └───────┬───────┘
               │
               ▼
         Latent Vector
               │
               ▼
      Transformer Decoder
               │
               ▼
      Reconstructed Sequence
```

Separate models are trained for the two length regimes.

Implementations:

```
new_diffusion/final_model/transformer_vae_len40.ipynb
new_diffusion/final_model/transformer_vae_len60.ipynb
```

---

# Latent Diffusion

The diffusion process operates on the learned continuous latent representation rather than directly on discrete amino-acid tokens.

The conceptual process is:

```
Training Latents
      │
      ▼
Forward Noise Process
      │
      ▼
Noisy Latent Representation
      │
      ▼
Denoising Network
      │
      ▼
Generated Latent Representation
```

Separate latent diffusion models are used for the short and long peptide branches.

Implementations:

```
new_diffusion/final_model/diffusion__len40_v2.ipynb
new_diffusion/final_model/diffusion__len60_v2.ipynb
```

---

# LSTM Sequence Decoding

The generated latent vectors are converted back into amino-acid sequences using branch-specific LSTM decoders.

```
Generated Latent
       │
       ▼
LSTM Decoder
       │
       ▼
Amino Acid Tokens
       │
       ▼
Generated Peptide
```

Separate decoders are used for:

```
Short branch → 5–40 aa

Long branch → 41–60 aa
```

Implementations:

```
new_diffusion/final_model/lstm_decoder_len40.ipynb
new_diffusion/final_model/lstm_decoder_len60.ipynb
```

---

# Candidate Screening

Generated peptide sequences are evaluated using a multi-stage computational screening pipeline.

## Stage 1 — AMP Activity Screening

The AMP classifier estimates:

```
P(AMP | sequence)
```

Candidates with high predicted antimicrobial probability are prioritized.

---

## Stage 2 — Hemolysis Screening

AMP-positive candidates are evaluated using the hemolysis classifier.

Candidates with lower predicted hemolysis probability are prioritized.

---

## Stage 3 — Candidate Ranking

The generated candidates are ranked using a combined score incorporating antimicrobial activity and predicted hemolytic potential.

The overall screening pipeline is:

```
Generated Peptides
        │
        ▼
AMP Probability
        │
        ▼
AMP-positive Candidates
        │
        ▼
Hemolysis Probability
        │
        ▼
Low-Hemolysis Candidates
        │
        ▼
Final Ranking
```

---

# Novelty Analysis

Novelty is evaluated at the sequence level relative to the AMP training dataset.

A generated peptide is considered **training-set novel** when its exact sequence is absent from the AMP training sequence set.

```
Generated Peptide
       │
       ▼
Compare with AMP Training Set
       │
       ├───────────────┐
       │               │
       ▼               ▼
  Present          Absent
       │               │
       ▼               ▼
 Non-novel          Novel
```

The final analysis is implemented in:

```
final_analysis.ipynb
```

## Novelty Definition

The term "novel" in this project refers specifically to:

> Sequence-level absence from the AMP training dataset used in the generative modeling pipeline.

Therefore, sequence-level novelty should **not** be interpreted as proof that a peptide has never been reported in any external biological database.

---

# Physicochemical Analysis

Selected generated candidates are characterized using:

-  Sequence length 
-  Molecular weight 
-  Isoelectric point (pI) 
-  Aromaticity 
-  Instability index 

Implementation:

```
properties/property_extraction.ipynb
```

These properties provide computational characterization of the generated peptide population.

---

# Amino Acid Composition Analysis

Amino acid composition is analyzed to characterize residue-level patterns in generated peptides.

The generated candidates are compared with known AMP sequences to investigate whether the generated population retains sequence-composition characteristics associated with antimicrobial peptides.

The analysis focuses on the distribution of amino acids and descriptive characteristics such as enrichment of charged and hydrophobic residues.

This analysis is intended as a characterization of generated candidates and does not independently establish antimicrobial activity.

---

# Embedding-Space Analysis

High-dimensional ESM-derived peptide embeddings are projected into two dimensions using **UMAP**.

The visualization is used to examine the representation-space distribution of:

```
Known AMP sequences
        vs.
Generated / Novel peptide sequences
```

The analysis helps investigate whether generated peptides occupy representation-space regions related to known AMPs while also exploring surrounding sequence space.

Implementation:

```
final_analysis.ipynb
```

> UMAP is used as a visualization and exploratory analysis technique. It does not constitute experimental evidence of biological activity.

---

# Experimental Results

The current computational pipeline produced the following candidate statistics:

| MetricResult            |       |
| ----------------------- | ----- |
| AMP training sequences  | 8,400 |
| Short-branch candidates | 5,343 |
| Long-branch candidates  | 1,749 |
| Novel candidates        | 7,011 |

The final generated candidate population was evaluated through:

-  AMP probability screening 
-  Hemolysis probability screening 
-  Novelty analysis 
-  Physicochemical characterization 
-  Amino acid composition analysis 
-  UMAP embedding analysis 

---

# Physicochemical Summary

## Short-Peptide Candidates

The observed mean properties for the analyzed short-peptide population were:

| PropertyMean      |            |
| ----------------- | ---------- |
| Length            | 17.43 aa   |
| Molecular Weight  | 2023.56 Da |
| pI                | 10.15      |
| Aromaticity       | 0.107      |
| Instability Index | 30.09      |

---

## Long-Peptide Candidates

The observed mean properties for the analyzed long-peptide population were:

| PropertyMean      |            |
| ----------------- | ---------- |
| Length            | 47.71 aa   |
| Molecular Weight  | 5291.93 Da |
| pI                | 8.78       |
| Aromaticity       | 0.078      |
| Instability Index | 39.70      |

These values describe computationally generated candidate populations and should not be interpreted as experimentally validated biological measurements.

---

# Repository Structure

```
AMP-Latent-Diffusion/
│
├── .gitignore
├── README.md
├── final_analysis.ipynb
│
├── classifers/
│   ├── amp_classifier.ipynb
│   ├── classifier_ablation.ipynb
│   └── hemolysis_classifier.ipynb
│
├── data/
│   └── processed_data&code/
│       ├── dataset.ipynb
│       ├── hemolysis_dataset.ipynb
│       └── nonamp_dataset.ipynb
│
├── diffusion/
│   ├── autoencoder.ipynb
│   ├── diffusion_generator.ipynb
│   ├── latent_analysis.ipynb
│   ├── latent_diffusion.ipynb
│   └── sequence_decoder.ipynb
│
├── embeddings/
│   └── embedding_generation.ipynb
│
├── new_diffusion/
│   ├── prepare_sequences.ipynb
│   ├── prepare_sequences_len60_v2.ipynb
│   │
│   ├── extract_latents/
│   │   ├── extract_latents40.ipynb
│   │   └── extract_latents60.ipynb
│   │
│   └── final_model/
│       ├── transformer_vae_len40.ipynb
│       ├── transformer_vae_len60.ipynb
│       ├── diffusion__len40_v2.ipynb
│       ├── diffusion__len60_v2.ipynb
│       ├── lstm_decoder_len40.ipynb
│       ├── lstm_decoder_len60.ipynb
│       └── latent_analysis.ipynb
│
└── properties/
    └── property_extraction.ipynb
```

---

# Reproducibility

The recommended execution order is:

## Step 1 — Prepare Datasets

Run:

```
data/processed_data&code/dataset.ipynb
data/processed_data&code/nonamp_dataset.ipynb
data/processed_data&code/hemolysis_dataset.ipynb
```

Prepare and clean the required sequence datasets.

---

## Step 2 — Generate Sequence Embeddings

Run:

```
embeddings/embedding_generation.ipynb
```

Generate the required ESM-based sequence representations.

---

## Step 3 — Train AMP Classifier

Run:

```
classifers/amp_classifier.ipynb
```

---

## Step 4 — Train Hemolysis Classifier

Run:

```
classifers/hemolysis_classifier.ipynb
```

---

## Step 5 — Prepare Length-Specific AMP Sequences

Prepare the two peptide regimes:

```
Short: 5–40 aa

Long: 41–60 aa
```

Use the corresponding sequence preparation notebooks.

---

## Step 6 — Train Transformer-VAE Models

Train the two branch-specific models:

```
new_diffusion/final_model/transformer_vae_len40.ipynb

new_diffusion/final_model/transformer_vae_len60.ipynb
```

---

## Step 7 — Extract Latent Representations

Run:

```
new_diffusion/extract_latents/extract_latents40.ipynb

new_diffusion/extract_latents/extract_latents60.ipynb
```

---

## Step 8 — Train Latent Diffusion Models

Run:

```
new_diffusion/final_model/diffusion__len40_v2.ipynb

new_diffusion/final_model/diffusion__len60_v2.ipynb
```

---

## Step 9 — Generate Latent Samples

Generate new latent representations independently for the short and long branches.

---

## Step 10 — Decode Generated Latents

Run the corresponding LSTM decoder:

```
new_diffusion/final_model/lstm_decoder_len40.ipynb

new_diffusion/final_model/lstm_decoder_len60.ipynb
```

---

## Step 11 — AMP Screening

Apply the trained AMP classifier to generated peptide sequences.

---

## Step 12 — Hemolysis Screening

Apply the trained hemolysis classifier to the AMP-positive candidates.

---

## Step 13 — Novelty Analysis

Compare generated candidates against the AMP training sequence set.

---

## Step 14 — Physicochemical Characterization

Run:

```
properties/property_extraction.ipynb
```

---

## Step 15 — Final Analysis

Run:

```
final_analysis.ipynb
```

for:

-  Candidate statistics 
-  Novelty analysis 
-  Amino-acid composition 
-  Physicochemical analysis 
-  Embedding-space visualization 
-  Final candidate ranking 

---

# Computational Requirements

The project uses Python and Jupyter notebooks.

Major libraries and frameworks include:

-  Python 
-  PyTorch 
-  Transformers 
-  ESM-based protein representations 
-  NumPy 
-  Pandas 
-  scikit-learn 
-  XGBoost 
-  Biopython 
-  UMAP 
-  Matplotlib 
-  Jupyter 

GPU acceleration is recommended for:

-  ESM embedding generation 
-  Transformer-VAE training 
-  Latent diffusion training 
-  LSTM decoder training/inference 

Exact package versions may depend on the computational environment used for the experiments.

---

# Data and Model Artifacts

Large data and model artifacts are intentionally excluded from this repository.

The following file types are ignored:`

```
*.csv
*.xlsx
*.xls

*.npy
*.npz

*.pt
*.pth
*.ckpt
*.safetensors

*.pkl
```

This includes:

-  Raw datasets 
-  Processed sequence libraries 
-  Generated peptide libraries 
-  ESM embeddings 
-  Latent vectors 
-  Generated latent samples 
-  Transformer-VAE checkpoints 
-  Diffusion checkpoints 
-  LSTM decoder checkpoints 
-  AMP classifier checkpoints 
-  Hemolysis classifier checkpoints 

The exclusion keeps the Git repository lightweight and avoids storing large binary artifacts directly in Git history.

---

# Limitations

The current computational framework has several limitations.

### 1. Computational predictions are not experimental validation

AMP activity and hemolysis values are predictions from trained machine-learning classifiers.

They do not replace laboratory assays.

### 2. Novelty is training-set dependent

Novelty is defined as exact sequence-level absence from the AMP training dataset.

A sequence classified as novel by this definition may still exist in another biological database or publication.

### 3. UMAP is exploratory

UMAP is used to visualize embedding-space relationships.

It does not demonstrate antimicrobial activity, structural similarity, or biological function by itself.

### 4. Experimental validation is required

High-scoring candidates require experimental validation before conclusions regarding antimicrobial efficacy or toxicity can be made.

### 5. Biological mechanisms are not established computationally

The current framework does not experimentally establish:

-  Mechanism of action 
-  Membrane interaction 
-  Structural conformation 
-  Binding targets 
-  In-vivo efficacy 
-  Therapeutic safety 

---

# Future Work

Future extensions of the framework include:

-  Experimental validation of high-ranking generated AMP candidates. 
-  In-vitro antimicrobial activity assays. 
-  Experimental hemolysis and cytotoxicity testing. 
-  Structural characterization of selected peptides. 
-  Broader database-based novelty analysis. 
-  Multi-objective optimization of antimicrobial activity and toxicity. 
-  Conditional generation based on desired peptide properties. 
-  Incorporation of structural information into generative modeling. 
-  Improved diversity and sequence-quality evaluation. 
-  Evaluation against additional AMP prediction benchmarks. 
-  Large-scale candidate generation and systematic diversity analysis. 
-  Integration of experimentally validated candidates into subsequent model training. 

---

# Research Status

This repository represents the computational implementation of ongoing research into AI-assisted antimicrobial peptide discovery.

The framework is intended to generate and prioritize candidate peptide sequences for subsequent experimental investigation.

Generated peptide sequences should be regarded as **computational candidates rather than experimentally validated antimicrobial or therapeutic agents**.

---

# Data Sources

## DRAMP

Database of Antimicrobial Activity and Structure of Peptides:

[https://dramp.cpu-bioinfor.org/](https://dramp.cpu-bioinfor.org/)

---

## UniProt

Universal Protein Resource:

[https://www.uniprot.org/](https://www.uniprot.org/)

---

# Citation

If this repository contributes to academic research, please cite the associated publication:

```
Citation information will be added after publication.
```

A formal citation will be added once the associated research paper is published.

---

# License

License information will be added before unrestricted public release.

---

# Author

## Anshul Chaudhary

AI/ML Researcher

Research interests:

-  Artificial Intelligence 
-  Machine Learning 
-  Deep Generative Models 
-  Protein and Peptide Modeling 
-  Antimicrobial Peptide Discovery 
-  Computational Biology 
-  Representation Learning 
-  Diffusion Models 

GitHub:

https://github.com/anshul-3000

---

# Acknowledgements

This work makes use of publicly available biological sequence resources and open-source machine-learning frameworks.

The project uses data derived from DRAMP and UniProt and computational methods based on Transformer representations, variational autoencoders, diffusion modeling, and machine-learning-based biological property prediction.
