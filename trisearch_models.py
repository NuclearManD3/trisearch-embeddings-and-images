#!/usr/bin/env python3
"""
Utility library that wraps the three resized models under ``models/`` in small,
easy-to-use classes for the runner (smoke-test) scripts:

- ``SiglipEmbedder``  : image  -> list of Matryoshka (1024-dim) patch embeddings.
- ``Qwen3MoeEmbedder``: text   -> list of Matryoshka (1024-dim) token embeddings,
                        with optional merging of similar consecutive embeddings.
- ``MMDiTGenerator``  : text *or* embeddings -> a generated image. Text is
                        tokenized like a normal text-to-image pipeline; external
                        embeddings are pushed through a small transform stage
                        that maps them to the transformer's conditioning shape.

All embeddings are Matryoshka embeddings of ``EMBED_DIM`` (1024) dimensions:
they are L2-normalized so that a prefix of any length is itself a usable
embedding, and so that a dot product equals cosine similarity.

The models were only *resized* from their baselines and are otherwise
un-trained, so outputs are not meaningful yet -- these wrappers exist so the
runner scripts can confirm each model "turns on and runs" (a smoke test), and
so the same code can be reused for evaluation after training.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

warnings.filterwarnings("ignore")

# Every embedding this project produces is a 1024-dim Matryoshka embedding.
EMBED_DIM = 1024

# Defaults: where each model lives on disk and which checkpoint supplies the
# (light) tokenizer / image-processor that the model dir itself does not carry.
SIGLIP_DIR = "models/siglip-vision"
QWEN_DIR = "models/qwen3-moe"
MMDIT_DIR = "models/mmdit"

SIGLIP_PROCESSOR_ID = "google/siglip-so400m-patch14-384"
QWEN_TOKENIZER_ID = "Qwen/Qwen3-1.7B"
# The MMDiT model dir holds only the transformer; the VAE that decodes its
# latents back to full-resolution pixels comes from the SD3 baseline.
MMDIT_VAE_ID = "v2ray/stable-diffusion-3-medium-diffusers"


# ============================ SHARED HELPERS ============================

def matryoshka_normalize(embeddings, dim=None):
    """L2-normalize embeddings (optionally truncated to a Matryoshka prefix).

    embeddings : tensor of shape (..., D).
    dim        : if given, keep only the first ``dim`` components before
                 normalizing (a Matryoshka prefix). Defaults to the full width.
    """
    if dim is not None:
        embeddings = embeddings[..., :dim]
    return F.normalize(embeddings, p=2, dim=-1)


def late_interaction_score(query_embeddings, doc_embeddings):
    """ColBERT-style MaxSim late-interaction score between two token sets.

    query_embeddings / doc_embeddings : (n, D) / (m, D) tensors of normalized
    embeddings. The score is ``sum_i max_j <q_i, d_j>`` -- for each query token
    we take its best-matching document token and sum those similarities.
    """
    q = _stack(query_embeddings).to(torch.float32)
    d = _stack(doc_embeddings).to(torch.float32)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if d.ndim == 1:
        d = d.unsqueeze(0)
    if q.numel() == 0 or d.numel() == 0:
        return 0.0
    sim = q @ d.T  # (n, m); dot product == cosine because inputs are normalized.
    return sim.max(dim=1).values.sum().item()


def _stack(embeddings):
    """Turn a list of 1-D embeddings into a single (n, D) tensor."""
    if isinstance(embeddings, torch.Tensor):
        return embeddings
    return torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in embeddings])


# ============================ SIGLIP (IMAGE) ============================

class SiglipEmbedder:
    """Vision embedder: takes a PIL ``Image`` and returns patch embeddings.

    The SigLIP vision tower produces one hidden vector per image patch; each is
    projected to the shared 1024-dim Matryoshka space and normalized. The list
    of per-patch embeddings is what a late-interaction retriever consumes.
    """

    def __init__(self, model_dir=SIGLIP_DIR, processor_id=SIGLIP_PROCESSOR_ID,
                 embed_dim=EMBED_DIM, device="cpu"):
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.device = device
        self.embed_dim = embed_dim
        self.model = SiglipVisionModel.from_pretrained(model_dir).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(processor_id)
        # The processor comes from the 384px baseline, but our resized tower
        # expects `image_size` px (e.g. 540) -> keep the patch grid consistent
        # with the model's learned position embeddings.
        target = self.model.config.image_size
        self.processor.size = {"height": target, "width": target}
        hidden = self.model.config.hidden_size
        # Projection head to the shared Matryoshka embedding space.
        self.projection = nn.Linear(hidden, embed_dim).to(device).eval()

    @torch.no_grad()
    def embed_image(self, image, matryoshka_dim=None):
        """Return a list of 1024-dim patch embeddings for ``image`` (PIL.Image)."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=pixel_values)
        # (1, num_patches, hidden) -> project every patch token.
        tokens = out.last_hidden_state[0]
        projected = self.projection(tokens)
        normed = matryoshka_normalize(projected, dim=matryoshka_dim)
        return [row for row in normed]


# ============================ QWEN3-MOE (TEXT) ============================

class Qwen3MoeEmbedder:
    """Text embedder: takes a string and returns token embeddings.

    Every token's final hidden state is projected to the 1024-dim Matryoshka
    space and normalized. Consecutive embeddings whose cosine similarity is at
    least ``merge_threshold`` are merged (averaged then re-normalized), which
    collapses redundant tokens for cheaper late-interaction search.
    """

    def __init__(self, model_dir=QWEN_DIR, tokenizer_id=QWEN_TOKENIZER_ID,
                 embed_dim=EMBED_DIM, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.embed_dim = embed_dim
        # AutoModel gives the bare transformer whose output is the hidden state.
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        hidden = self.model.config.hidden_size
        self.projection = nn.Linear(hidden, embed_dim).to(device).eval()

    @torch.no_grad()
    def embed_text(self, text, merge_threshold=1.0, matryoshka_dim=None):
        """Return a list of 1024-dim token embeddings for ``text``.

        merge_threshold : minimum cosine similarity for two *consecutive*
                          embeddings to be merged into one. ``1.0`` merges only
                          (near-)identical neighbours; lower values merge more.
        """
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        tokens = out.last_hidden_state[0]
        projected = self.projection(tokens)
        normed = matryoshka_normalize(projected, dim=matryoshka_dim)
        merged = self._merge_similar(normed, merge_threshold)
        return [row for row in merged]

    @staticmethod
    def _merge_similar(embeddings, merge_threshold):
        """Greedily merge consecutive embeddings with cosine sim >= threshold."""
        if merge_threshold is None or embeddings.shape[0] <= 1:
            return embeddings
        groups = [embeddings[0:1]]
        for i in range(1, embeddings.shape[0]):
            current = embeddings[i]
            rep = matryoshka_normalize(groups[-1].mean(dim=0))
            if torch.dot(rep, current).item() >= merge_threshold:
                groups[-1] = torch.cat([groups[-1], current.unsqueeze(0)], dim=0)
            else:
                groups.append(current.unsqueeze(0))
        merged = torch.stack([matryoshka_normalize(g.mean(dim=0)) for g in groups])
        return merged


# ============================ MMDiT (IMAGE GENERATION) ============================

class MMDiTGenerator:
    """SD3-style MMDiT image generator wrapper.

    Two ways to condition generation:
      * ``generate(text=...)``       -- the text is tokenized like a normal
        text-to-image pipeline and turned into conditioning embeddings.
      * ``generate(embeddings=...)`` -- externally supplied 1024-dim Matryoshka
        embeddings are pushed through a small transform stage that maps them to
        the transformer's conditioning shapes.

    Since the transformer is only resized (un-trained), the produced image is
    noise; the point is to prove the whole pipeline runs end to end.

    The transformer works in the VAE *latent* space (16 channels, 8x smaller
    than the image). The latents it produces are decoded back to full-resolution
    pixels with the SD3 VAE, so the underlying output is a real HxW image (the
    latent grid upsampled 8x by the decoder) rather than the raw latent grid.
    """

    def __init__(self, model_dir=MMDIT_DIR, tokenizer_id=QWEN_TOKENIZER_ID,
                 vae_id=MMDIT_VAE_ID, embed_dim=EMBED_DIM, device="cpu"):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.models.transformers import SD3Transformer2DModel
        from transformers import AutoTokenizer

        self.device = device
        self.embed_dim = embed_dim
        self.transformer = SD3Transformer2DModel.from_pretrained(model_dir)
        self.transformer = self.transformer.to(device).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # Real SD3 VAE: decodes the 16-channel latents to RGB pixels with an 8x
        # spatial upscale, so a HxW image comes from an (H/8)x(W/8) latent.
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
        self.vae = self.vae.to(device).eval()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        cfg = self.transformer.config
        self.in_channels = cfg.in_channels
        self.joint_dim = cfg.joint_attention_dim
        self.pooled_dim = cfg.pooled_projection_dim

        # Text path: a stand-in "text encoder" (untrained) so the class can be
        # driven by a raw prompt, tokenized like a default pipeline.
        vocab = len(self.tokenizer)
        self.token_embedding = nn.Embedding(vocab, self.joint_dim).to(device).eval()
        self.text_pool = nn.Linear(self.joint_dim, self.pooled_dim).to(device).eval()

        # Embedding path: transform external 1024-dim embeddings to the
        # transformer's conditioning shapes.
        self.embed_to_seq = nn.Linear(embed_dim, self.joint_dim).to(device).eval()
        self.embed_to_pool = nn.Linear(embed_dim, self.pooled_dim).to(device).eval()

    @torch.no_grad()
    def _encode_text(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=77)
        ids = enc["input_ids"].to(self.device)
        seq = self.token_embedding(ids)                # (1, seq, joint_dim)
        pooled = self.text_pool(seq.mean(dim=1))        # (1, pooled_dim)
        return seq, pooled

    @torch.no_grad()
    def _encode_embeddings(self, embeddings):
        embs = _stack(embeddings).to(self.device)
        if embs.ndim == 2:
            embs = embs.unsqueeze(0)                    # (1, n, embed_dim)
        seq = self.embed_to_seq(embs)                   # (1, n, joint_dim)
        pooled = self.embed_to_pool(embs.mean(dim=1))   # (1, pooled_dim)
        return seq, pooled

    @torch.no_grad()
    def generate(self, text=None, embeddings=None, height=640, width=640,
                 num_inference_steps=4, seed=0):
        """Run the denoising loop and return a PIL image.

        Provide exactly one of ``text`` or ``embeddings``.
        ``height`` / ``width`` are the final *image* size in pixels; the
        transformer works on an (H/8)x(W/8) latent, which the VAE decodes back
        to the full HxW resolution. Both must be multiples of ``8 * patch_size``
        (16 by default) so the latent tiles cleanly into transformer patches.
        """
        if (text is None) == (embeddings is None):
            raise ValueError("Provide exactly one of `text` or `embeddings`.")

        patch = self.transformer.config.patch_size
        divisor = self.vae_scale_factor * patch
        if height % divisor or width % divisor:
            raise ValueError(
                f"height/width must be multiples of {divisor} "
                f"(vae_scale_factor {self.vae_scale_factor} x patch_size {patch}); "
                f"got {height}x{width}."
            )

        if text is not None:
            encoder_hidden_states, pooled = self._encode_text(text)
        else:
            encoder_hidden_states, pooled = self._encode_embeddings(embeddings)

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(1, self.in_channels, latent_h, latent_w,
                              generator=generator).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for t in self.scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(self.device)
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled,
                timestep=timestep,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred, t, latents,
                                          return_dict=False)[0]

        return self._latents_to_image(latents)

    @torch.no_grad()
    def _latents_to_image(self, latents):
        """Decode latents to a full-resolution RGB image with the SD3 VAE.

        The VAE upsamples the (H/8)x(W/8) latent by ``vae_scale_factor`` (8x),
        so the returned image is the real HxW picture, not the raw latent grid.
        """
        import numpy as np
        from PIL import Image

        cfg = self.vae.config
        latents = latents / cfg.scaling_factor + getattr(cfg, "shift_factor", 0.0)
        decoded = self.vae.decode(latents, return_dict=False)[0]
        x = (decoded[0] / 2 + 0.5).clamp(0, 1)          # (3, H, W) in [0, 1]
        array = (x * 255.0).round().to(torch.uint8)
        array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")


# ============================ QUERY STORE ============================

class LateInteractionStore:
    """Tiny in-memory store mapping a query label to its token embeddings.

    Used by the runner scripts to keep a history of previous queries and, for
    each new query, report the most similar prior queries by MaxSim score.
    """

    def __init__(self):
        self.entries = []  # list of (label, embeddings tensor)

    def most_similar(self, embeddings, top_k=2):
        """Return the ``top_k`` prior (label, score) pairs, highest score first."""
        scored = [(label, late_interaction_score(embeddings, stored))
                  for label, stored in self.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def add(self, label, embeddings):
        self.entries.append((label, _stack(embeddings)))
