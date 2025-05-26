"""
Audio Spectrogram Transformer (AST) implementation for drum transcription.
Based on the AST architecture with modifications for drum-specific tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
from .transformer_config import BaseConfig


class PatchEmbedding(nn.Module):
    """Convert 2D spectrogram patches to embeddings."""
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (16, 16),
                 in_channels: int = 1,
                 embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Use conv2d to extract patches and project to embedding dimension
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrograms [batch_size, channels, time, freq]
        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        # x shape: (batch_size, 1, time, freq)
        batch_size, channels, time, freq = x.shape
        
        # Extract patches and project
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_time, num_patches_freq)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for time-frequency patches."""
    
    def __init__(self, embed_dim: int, max_time_patches: int = 64, max_freq_patches: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time_patches = max_time_patches
        self.max_freq_patches = max_freq_patches
        
        # Create learnable position embeddings
        self.time_embed = nn.Parameter(torch.randn(1, max_time_patches, embed_dim // 2))
        self.freq_embed = nn.Parameter(torch.randn(1, max_freq_patches, embed_dim // 2))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x: torch.Tensor, patch_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [batch_size, num_patches, embed_dim]
            patch_shape: (time_patches, freq_patches)
        Returns:
            Position-encoded embeddings with CLS token
        """
        batch_size, num_patches, embed_dim = x.shape
        time_patches, freq_patches = patch_shape
        
        # Create 2D position embeddings
        time_pos = self.time_embed[:, :time_patches, :].repeat(1, freq_patches, 1)
        freq_pos = self.freq_embed[:, :freq_patches, :].repeat_interleave(time_patches, dim=1)
        
        # Concatenate time and frequency positions
        pos_embed = torch.cat([time_pos, freq_pos], dim=-1)
        
        # Ensure position embedding matches the number of patches
        if pos_embed.shape[1] > num_patches:
            pos_embed = pos_embed[:, :num_patches, :]
        elif pos_embed.shape[1] < num_patches:
            # Pad with zeros if we have fewer position embeddings
            padding = torch.zeros(1, num_patches - pos_embed.shape[1], embed_dim, device=pos_embed.device)
            pos_embed = torch.cat([pos_embed, padding], dim=1)
        
        # Add positional encoding
        x = x + pos_embed
        
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
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-normalization."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
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
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), attention_mask)
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class DrumTranscriptionTransformer(nn.Module):
    """Audio Spectrogram Transformer for drum transcription."""
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size,
            in_channels=1,
            embed_dim=config.hidden_size
        )
        
        # Calculate maximum patch dimensions based on audio length and spectrogram size
        max_time_frames = int(config.max_audio_length * config.sample_rate / config.hop_length) + 1
        max_time_patches = (max_time_frames + config.patch_size[0] - 1) // config.patch_size[0]
        max_freq_patches = (config.n_mels + config.patch_size[1] - 1) // config.patch_size[1]
        
        self.pos_encoding = PositionalEncoding2D(
            embed_dim=config.hidden_size,
            max_time_patches=max_time_patches,
            max_freq_patches=max_freq_patches
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Classification heads
        self.drum_classifier = nn.Linear(config.hidden_size, config.num_drum_classes)
        
        # Temporal aggregation for sequence-level prediction
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def forward(self, spectrograms: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            spectrograms: Input spectrograms [batch_size, 1, time, freq]
            attention_mask: Optional attention mask
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing logits and optionally embeddings
        """
        batch_size, channels, time_frames, freq_bins = spectrograms.shape
        
        # Convert to patches
        patch_embeddings = self.patch_embed(spectrograms)
        
        # Calculate patch shape
        time_patches = time_frames // self.config.patch_size[0]
        freq_patches = freq_bins // self.config.patch_size[1]
        patch_shape = (time_patches, freq_patches)
        
        # Add positional encoding and CLS token
        x = self.pos_encoding(patch_embeddings, patch_shape)
        
        # Store embeddings for each layer if requested
        layer_embeddings = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
            if return_embeddings:
                layer_embeddings.append(x.clone())
        
        # Final norm
        x = self.norm(x)
        
        # Use CLS token for global classification
        cls_embedding = x[:, 0]  # [batch_size, hidden_size]
        
        # Drum classification
        drum_logits = self.drum_classifier(cls_embedding)
        
        # Prepare output
        output = {
            'logits': drum_logits,
            'cls_embedding': cls_embedding
        }
        
        if return_embeddings:
            output['layer_embeddings'] = layer_embeddings
            output['final_embedding'] = x
            
        return output
    
    def get_attention_maps(self, spectrograms: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Extract attention maps from a specific layer."""
        # This would require modifying the attention modules to return attention weights
        # For now, return None as placeholder
        return None


def create_model(config: BaseConfig) -> DrumTranscriptionTransformer:
    """Factory function to create a drum transcription transformer."""
    model = DrumTranscriptionTransformer(config)
    
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
    
    # Test forward pass
    batch_size = 2
    time_frames = 256  # ~5.8 seconds at 22050 Hz with hop_length=512
    freq_bins = 128
    
    dummy_input = torch.randn(batch_size, 1, time_frames, freq_bins)
    
    with torch.no_grad():
        output = model(dummy_input, return_embeddings=True)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"CLS embedding shape: {output['cls_embedding'].shape}")
    print("Model test passed!")