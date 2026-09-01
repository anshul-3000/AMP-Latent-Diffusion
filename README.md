# AMP-Latent-Diffusion

### Dual-Branch Latent Diffusion Framework for Novel Antimicrobial Peptide Generation

A deep generative framework for computational antimicrobial peptide (AMP) discovery that combines Transformer-based Variational Autoencoders (Transformer-VAE), latent diffusion modeling, branch-specific LSTM sequence decoders, AMP activity prediction, hemolysis screening, novelty analysis, physicochemical characterization, amino-acid composition analysis, and embedding-space visualization.

---

## Overview

Antimicrobial peptides are promising candidates for addressing the growing problem of antimicrobial resistance. However, experimental discovery and screening of large peptide libraries are expensive and time-consuming.

This project investigates a computational approach for generating novel AMP candidates by learning peptide representations in a continuous latent space and performing generative modeling within that space.

The framework uses **two separate generation branches**:

- **Short peptides:** 5–40 amino acids
- **Long peptides:** 41–60 amino acids

Each branch uses its own Transformer-VAE, latent diffusion model, and LSTM sequence decoder.

Generated sequences are subsequently evaluated using AMP activity prediction, hemolysis screening, novelty analysis, physicochemical characterization, amino-acid composition analysis, and embedding-space visualization.

> **Important:** The generated peptides are computational candidates and have not been experimentally validated. Biological activity and toxicity must be confirmed through appropriate wet-lab experiments.

---

# Framework

```text
                    ┌─────────────────────┐
                    │    DRAMP AMP Data   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Length Partition   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Short Branch    │         │ Long Branch     │
        │ 5–40 aa         │         │ 41–60 aa        │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Transformer-VAE │         │ Transformer-VAE │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Latent Space    │         │ Latent Space    │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Latent Diffusion│         │ Latent Diffusion│
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ LSTM Decoder    │         │ LSTM Decoder    │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Generated Peptides  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ AMP Activity Filter │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hemolysis Filter    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Novelty Analysis    │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Biological / Sequence Analysis │
              ├────────────────────────────────┤
              │ • Physicochemical Properties   │
              │ • Amino Acid Composition       │
              │ • Length Distribution          │
              │ • UMAP Embedding Analysis      │
              └────────────────────────────────┘
