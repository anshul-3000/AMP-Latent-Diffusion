# AMP-Latent-Diffusion

## Dual-Branch Latent Diffusion Framework for Novel Antimicrobial Peptide Generation

This repository contains the implementation of a deep generative framework for computational antimicrobial peptide (AMP) discovery. The framework combines Transformer-based Variational Autoencoders (Transformer-VAE), latent diffusion modeling, branch-specific LSTM sequence decoders, AMP activity prediction, hemolysis screening, novelty analysis, and physicochemical characterization.

The proposed approach separately models short and long peptide regimes to explore antimicrobial peptide sequence space while maintaining predicted antimicrobial activity and low hemolytic potential.

---

## Framework Overview

The overall workflow is:

Dataset Preparation
        ↓
ESM-2 Sequence Embeddings
        ↓
AMP / Non-AMP Classification
        ↓
Length-based AMP Partitioning
        ↓
┌───────────────────────┬───────────────────────┐
│ Short Peptide Branch  │ Long Peptide Branch   │
│ 5–40 aa               │ 41–60 aa              │
└───────────┬───────────┴───────────┬───────────┘
            ↓                       ↓
      Transformer-VAE         Transformer-VAE
            ↓                       ↓
      Latent Representation   Latent Representation
            ↓                       ↓
       Latent Diffusion        Latent Diffusion
            ↓                       ↓
       Generated Latents       Generated Latents
            ↓                       ↓
       LSTM Decoder            LSTM Decoder
            └───────────┬───────────┘
                        ↓
                 Generated Peptides
                        ↓
                  AMP Screening
                        ↓
                Hemolysis Screening
                        ↓
                  Novelty Analysis
                        ↓
       Physicochemical & Composition Analysis
                        ↓
                    UMAP Analysis

---

## Key Components

### 1. Dataset Preparation

Two sequence sources were used:

- **DRAMP** — antimicrobial peptide sequences
- **UniProt** — non-antimicrobial protein sequences

The preprocessing notebooks are located in:

```text
data/processed_data&code/
