"""
Audio Spectrogram Transformer (AST) implementation for drum transcription.
Based on the AST architecture with modifications for drum-specific tasks.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .transformer_config import BaseConfig


class PatchEmbedding(nn.Module):
    """Convert 1D spectrogram patches to embeddings."""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 128,  # n_mels
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Use conv1d to extract patches and project to embedding dimension
        self.projection = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrograms [batch_size, channels, time, freq]
        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        # x shape: (batch_size, 1, time, freq)
        # The Conv1d layer expects (batch_size, in_channels, length)
        # where in_channels is freq (n_mels) and length is time.
        # Input x shape: (batch_size, 1, n_mels, time)
        x = x.squeeze(1)  # -> (batch_size, n_mels, time)

        # Extract patches and project
        x = self.projection(x)  # -> (batch_size, embed_dim, num_patches_time)

        # Reshape for the transformer encoder
        x = x.transpose(1, 2)  # -> (batch_size, num_patches_time, embed_dim)
        return x


class PositionalEncoding1D(nn.Module):
    """1D positional encoding for time patches."""

    def __init__(self, embed_dim: int, max_time_patches: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time_patches = max_time_patches

        # Create learnable position embeddings
        self.time_embed = nn.Parameter(torch.randn(1, max_time_patches, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [batch_size, num_patches, embed_dim]
        Returns:
            Position-encoded embeddings with CLS token
        """
        batch_size, num_patches, embed_dim = x.shape

        # Add positional encoding
        x = x + self.time_embed[:, :num_patches, :]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional relative position encoding."""

    def __init__(self, embed_dim: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), attention_mask)

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))

        return x


class DrumTranscriptionTransformer(nn.Module):
    """Audio Spectrogram Transformer for drum transcription."""

    def __init__(self, config: BaseConfig, max_time_patches: Optional[int] = None):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size[0],
            in_channels=config.n_mels,
            embed_dim=config.hidden_size,
        )

        if max_time_patches is None:
            # Calculate maximum patch dimensions with conservative bounds for memory efficiency
            max_time_frames = (
                int(config.max_audio_length * config.sample_rate / config.hop_length)
                + 1
            )
            max_time_patches = (
                max_time_frames + config.patch_size[0] - 1
            ) // config.patch_size[0]

            # Apply conservative limits to prevent memory explosion
            max_time_patches = min(max_time_patches, 256)  # Cap at 256 time patches

        self.pos_encoding = PositionalEncoding1D(
            embed_dim=config.hidden_size,
            max_time_patches=max_time_patches,
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=2.0,  # Reduced from 4.0 to save memory
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_drum_classes)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        spectrograms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spectrograms: Input spectrograms [batch_size, 1, time, freq]
            attention_mask: Optional attention mask
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary containing logits and optionally embeddings
        """
        patch_embeddings = self.patch_embed(spectrograms)
        x = self.pos_encoding(patch_embeddings)
        del patch_embeddings

        layer_embeddings = []
        use_checkpointing = (
            getattr(self.config, "gradient_checkpointing", False) and self.training
        )

        for layer in self.transformer_layers:
            if use_checkpointing:
                x = checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
            if return_embeddings:
                layer_embeddings.append(x.clone())

        x = self.norm(x)

        # Remove CLS token and apply classifier to each time step
        cls_embedding = x[:, 0, :]
        x = x[:, 1:, :]
        logits = self.classifier(x)

        output = {"logits": logits}
        if return_embeddings:
            output["layer_embeddings"] = layer_embeddings
            output["final_embedding"] = x
            output["cls_embedding"] = cls_embedding

        return output

    def get_attention_maps(
        self, spectrograms: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """Extract attention maps from a specific layer."""
        # This would require modifying the attention modules to return attention weights
        # For now, return None as placeholder
        return None


def create_model(
    config: BaseConfig, max_time_patches: Optional[int] = None
) -> DrumTranscriptionTransformer:
    """Factory function to create a drum transcription transformer."""
    model = DrumTranscriptionTransformer(config, max_time_patches=max_time_patches)

    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    from .transformer_config import get_config

    config = get_config("local")
    model = create_model(config)

    # Test forward pass with smaller inputs to avoid memory issues
    batch_size = 1  # Reduced from 2
    time_frames = 128  # Reduced from 256
    freq_bins = 128  # Keep at 128

    dummy_input = torch.randn(batch_size, 1, time_frames, freq_bins)

    with torch.no_grad():
        output = model(dummy_input, return_embeddings=True)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"CLS embedding shape: {output['cls_embedding'].shape}")
    print("Model test passed!")
