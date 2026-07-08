# Multimodal Embedding & Generation System — Training Plan

**Project Goal**  
Build a split, efficient multimodal system consisting of:
- **Vision Embedder** (~1B params, SigLIP-style)
- **Text Embedder** (~1B params, Qwen3-MoE style)
- **Image Generator** (~2B params, MMDiT/DiT-style)

The system produces **unified Matryoshka embeddings** (~1024 dimensions) that are robust to binary and 2-bit quantization. All components operate in the same vector space, enabling strong cross-modal retrieval and high-quality image generation directly from embeddings.

---

## 1. System Architecture & Core Objectives

### Components
| Component          | Architecture          | Target Size | Output                  | Trainable? |
|--------------------|-----------------------|-------------|-------------------------|------------|
| Vision Embedder    | SigLIP Vision         | ~1B         | 1024-dim Matryoshka     | Yes        |
| Text Embedder      | Qwen3-MoE             | ~1B         | 1024-dim Matryoshka     | Yes        |
| Image Generator    | MMDiT / Diffusion Transformer | ~2B   | Image (from embedding)  | Yes        |

### Key Requirements
- Unified vector space across all three models
- Strong Matryoshka property (performance degrades gracefully with dimension truncation)
- Robustness to aggressive quantization of embeddings (binary / 2-bit)
- Embedding-centric interface (minimize token-based conditioning)
- Excellent performance on satellite imagery while retaining general capabilities

---

## 2. Data Strategy

### Core Datasets
| Dataset            | Type          | Size          | Quality     | Primary Use                  | Notes |
|--------------------|---------------|---------------|-------------|------------------------------|-------|
| **ChatEarthNet**   | Satellite     | ~163k         | High        | Reconstruction + Alignment   | Detailed captions |
| **SkyScript**      | Satellite     | ~5.2M         | Medium-High | Pretraining + Specialization | High semantic diversity |
| **RS5M**           | Satellite     | 5M            | Medium      | Domain adaptation            | Large scale |
| **CC12M / CC3M**   | General       | Millions      | High        | General pretraining          | High-quality captions |
| **Visual Genome**  | General       | ~100k         | Very High   | Fine-grained alignment       | Rich annotations |

### Data Mixing Strategy (Curriculum)
- **Stage 1–2**: 50% General (CC12M/CC3M + Visual Genome) + 50% Satellite (ChatEarthNet + SkyScript)
- **Stage 3–4**: 30% General + 70% Satellite + Synthetic
- **Stage 5+**: 15–20% General + 80%+ Satellite + Heavy Synthetic

### Synthetic Data Pipeline
- **Primary tool**: ZitGen (Qwen3.5-4B) + stronger VLMs for captioning unlabeled Sentinel-2 / Landsat data.
- Generate **hierarchical captions** (short + detailed) for Matryoshka training.
- Create hard negatives and preference pairs automatically.
- Use current model outputs for self-improvement loops (Stage 6).

---

## 3. Training Stages

### Stage 1: Seeding & Cross-Modal Alignment
**Goal**: Establish a shared Matryoshka vector space between vision and text embedders.

**Data**: 50/50 mix of general + satellite (ChatEarthNet prioritized for quality).

**Training**:
- Train **both embedders** together.
- **Loss**: Late interaction (ColBERT-style) similarity on positive pairs − similarity to random negatives.
- Random in-batch negatives are sufficient at this stage.
- Add basic Matryoshka loss across multiple dimensions.

**Outcome**: Embedders produce aligned embeddings for the same image.

### Stage 2: Generator Alignment via Reconstruction
**Goal**: Pull the image generator into the same vector space.

**Training**:
- **Freeze** vision embedder.
- Train **only the image generator**.
- Pipeline: `Image → Vision Embedder → Embedding → Generator → Reconstructed Image`
- Losses: Perceptual (LPIPS/DINO) + reconstruction (pixel/latent) + embedding consistency (re-embed generated image).

**Outcome**: Generator can produce images from embeddings in the shared space.

### Stage 3: Joint Unification & Contrastive Refinement
**Goal**: Tighten the unified vector space with richer signals.

**Training**: All three models trained jointly (generator with lower LR or LoRA).

**Losses** (multi-objective):
- Late interaction contrastive (improved hard negative mining)
- Cycle consistency (embed → generate → re-embed)
- Strong Matryoshka loss across multiple prefix dimensions
- Reconstruction + perceptual loss
- Diversity / anti-collapse regularizer on embeddings

**Data**: Add mined hard negatives + initial synthetic data.

### Stage 4: RL / Preference Optimization
**Goal**: Optimize for retrieval quality, generation quality, and robustness.

**Approach** (recommended order):

1. **Offline Preference Optimization (DPO/KTO)** — Preferred starting method
   - Generate preference pairs from Stage 3 model
   - Retrieval preferences (better vs worse embedding pairs)
   - Generation preferences (better vs worse reconstructions from same embedding)

2. **Online RL** (optional follow-up)
   - Reward model combining:
     - Retrieval metrics (Recall@K)
     - Reconstruction quality (perceptual + semantic)
     - Robustness after quantization/truncation

**Components trained**: Primarily generator + light updates to embedders.

### Stage 5: Satellite Specialization
**Goal**: Maximize performance on satellite imagery while preserving generality.

**Data**:
- Heavy SkyScript + RS5M + ChatEarthNet
- Large-scale synthetic captions from unlabeled satellite archives
- Maintain 15–20% general data to prevent forgetting

**Training**:
- Continue reconstruction + contrastive losses
- Increase weight on robustness objectives
- Optional satellite-specific augmentations (multi-scale, spectral-aware)

### Stage 6: Self-Improvement & Bootstrapping (Optional)
**Goal**: Further improve using the model’s own outputs.

**Methods**:
- Generate new training pairs from current embeddings
- Automatic hard negative mining
- Create additional preference data for another RL round
- Consistency training (embed → generate → re-embed)

### Stage 7: Final Calibration & Evaluation
- Light supervised fine-tuning on high-quality held-out data
- Quantization-aware calibration
- Comprehensive evaluation across all metrics

---

## 4. Loss Functions Summary

| Loss                        | Stages     | Purpose                              | Weight Progression      |
|----------------------------|------------|--------------------------------------|-------------------------|
| Late Interaction Contrastive | 1–5       | Cross-modal alignment                | High → Medium           |
| Matryoshka Loss            | 1–7       | Dimension robustness                 | Increasing              |
| Reconstruction + Perceptual| 2–6       | Information richness + generator quality | High             |
| Cycle Consistency          | 3–6       | Unified space enforcement            | Medium → High           |
| Diversity / Anti-collapse  | 3–5       | Prevent embedding collapse           | Medium                  |
| RL / Preference (DPO/KTO)  | 4+        | Optimization beyond supervised loss  | —                       |
| Robustness (Quantization)  | 4–7       | Binary/2-bit embedding performance   | Increasing              |

---

## 5. Evaluation Metrics

**Core Metrics** (tracked every stage):
- Cross-modal retrieval (Recall@1/5/10, nDCG)
- Embedding robustness (performance after 50%/25%/binary quantization and dimension truncation)
- Reconstruction quality (LPIPS, DINO similarity, CLIP score)
- Generation quality from embeddings (FID, CLIP score, human preference)

**Satellite-specific**:
- Zero-shot scene classification / retrieval on held-out satellite benchmarks
- Fine-grained attribute understanding (scale, structure, context)

---

## 6. Implementation Notes

### Hyperparameter Strategy
- Use the configuration search script (previously provided) to identify good starting architectures.
- Prefer **LoRA / QLoRA** on the generator after Stage 2.
- Use `device_map="auto"` + gradient checkpointing for memory efficiency.

### Synthetic Data Generation
- Primary model: ZitGen + stronger open VLMs
- Filtering: CLIP similarity + diversity sampling + LLM-as-judge
- Target: Generate detailed + hierarchical captions at scale

### Hardware Considerations (24GB VRAM)
- Stages 1–3: Feasible with LoRA + checkpointing
- Generator training: Heavy use of PEFT after Stage 2
- RL stage: Can be done with smaller batch sizes or offline methods (DPO)

---

## 7. Recommended Timeline (High-Level)

| Phase          | Stages     | Focus                          |
|----------------|------------|--------------------------------|
| Foundation     | 1–2        | Seeding + basic alignment      |
| Unification    | 3          | Joint training                 |
| Optimization   | 4          | RL / Preference                |
| Specialization | 5–6        | Satellite + Self-improvement   |
| Polish         | 7          | Final calibration              |

---

**Document Version**: 1.0
**Last Updated**: July 2026
**Status**: Ready for implementation

---

This document is designed to be both **high-level strategic** and **actionable**. You can copy it directly into a Notion page, GitHub wiki, or internal docs.

Would you like me to also generate:
- A companion **loss weighting schedule** (with specific numbers per stage)?
- A **data preparation checklist**?
- A **model configuration recommendations** section based on the earlier search script?

Just say the word and I’ll expand it.
