from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel


DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RetrieverConfig:
    model_name: str = DEFAULT_MODEL_NAME
    out_dim: int = 256
    pooling: str = "cls"
    text_pooling: str | None = None
    image_pooling: str | None = None
    max_length: int = 64
    freeze_backbone: bool = True
    device: str = get_default_device()
    local_files_only: bool = True


class PageDualEncoder(nn.Module):
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        self.config = config
        self.backbone = SiglipModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.device == "cuda" else torch.float32,
            local_files_only=config.local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = self.backbone.config.text_config.hidden_size
        self.text_proj = ProjectionHead(hidden_dim, config.out_dim)
        self.image_proj = ProjectionHead(hidden_dim, config.out_dim)

    def _position_weighted_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        weights = torch.linspace(1.0, 2.0, steps=seq_len, device=hidden_states.device)
        weights = weights.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is not None:
            weights = weights * attention_mask.float()

        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled

    def _resolve_pooling(self, explicit_pooling: str | None) -> str:
        return explicit_pooling or self.config.pooling

    def _pool_outputs(self, outputs: Any, attention_mask: torch.Tensor | None = None, pooling: str | None = None) -> torch.Tensor:
        active_pooling = self._resolve_pooling(pooling)

        if active_pooling == "cls":
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state[:, 0]

        if active_pooling == "position_weighted":
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return self._position_weighted_pool(outputs.last_hidden_state, attention_mask)
            if isinstance(outputs, torch.Tensor):
                return outputs

        if active_pooling == "pretrained":
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                return outputs.text_embeds
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                return outputs.image_embeds
            if isinstance(outputs, torch.Tensor):
                return outputs

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]
        raise TypeError(f"Cannot extract tensor features from output type: {type(outputs)}")

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        text_pooling = self._resolve_pooling(self.config.text_pooling)
        if text_pooling == "pretrained":
            outputs = self.backbone.get_text_features(**inputs)
            feats = self._pool_outputs(outputs, pooling="pretrained").float()
        else:
            outputs = self.backbone.text_model(**inputs)
            feats = self._pool_outputs(outputs, inputs.get("attention_mask"), pooling=text_pooling).float()
        feats = self.text_proj(feats)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_images(self, pil_images: list[Any]) -> torch.Tensor:
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        image_pooling = self._resolve_pooling(self.config.image_pooling)
        if image_pooling == "pretrained":
            outputs = self.backbone.get_image_features(**inputs)
            feats = self._pool_outputs(outputs, pooling="pretrained").float()
        else:
            outputs = self.backbone.vision_model(**inputs)
            feats = self._pool_outputs(outputs, pooling=image_pooling).float()
        feats = self.image_proj(feats)
        feats = F.normalize(feats, dim=-1)
        return feats


def info_nce_loss(q: torch.Tensor, p: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = q @ p.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss_q2p = F.cross_entropy(logits, labels)
    loss_p2q = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q2p + loss_p2q)


def hard_negative_margin_loss(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    pos_sim = (q * p).sum(dim=-1)
    neg_sim = (q * n).sum(dim=-1)
    return F.relu(margin - pos_sim + neg_sim).mean()


def save_retriever_checkpoint(
    checkpoint_path: Path,
    model: PageDualEncoder,
    extra_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_name": model.config.model_name,
        "out_dim": model.config.out_dim,
        "pooling": model.config.pooling,
        "text_pooling": model.config.text_pooling,
        "image_pooling": model.config.image_pooling,
        "max_length": model.config.max_length,
        "local_files_only": model.config.local_files_only,
        "text_proj": model.text_proj.state_dict(),
        "image_proj": model.image_proj.state_dict(),
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, checkpoint_path)


def load_retriever_from_checkpoint(
    checkpoint_path: Path,
    device: str | None = None,
    model_name: str | None = None,
) -> PageDualEncoder:
    target_device = device or get_default_device()
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    config = RetrieverConfig(
        model_name=model_name or ckpt.get("model_name", DEFAULT_MODEL_NAME),
        out_dim=int(ckpt.get("out_dim", 256)),
        pooling=ckpt.get("pooling", "cls"),
        text_pooling=ckpt.get("text_pooling"),
        image_pooling=ckpt.get("image_pooling"),
        max_length=int(ckpt.get("max_length", 64)),
        device=target_device,
        local_files_only=bool(ckpt.get("local_files_only", True)),
    )
    model = PageDualEncoder(config).to(target_device)
    model.text_proj.load_state_dict(ckpt["text_proj"])
    model.image_proj.load_state_dict(ckpt["image_proj"])
    model.eval()
    return model
