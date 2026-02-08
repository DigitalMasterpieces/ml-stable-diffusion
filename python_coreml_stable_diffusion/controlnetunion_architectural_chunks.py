#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Copyright (C) 2025 Digital Masterpieces GmbH. All Rights Reserved.
#
# Architectural chunking for SDXL ControlNet Union
# Splits ControlNet Union into 6 semantically meaningful chunks for Neural Engine compatibility
# This is specifically for Union ControlNet models which are too large for Neural Engine on mobile devices
#
# Note: This 6-chunk architecture is designed for ControlNet Union ProMax which has 3 down_blocks
# (not 4 like standard SDXL). Total of 10 residual outputs (additional_residual_0 through _9).
#
# Residual breakdown:
#   - 1 for initial sample (controlnet_down_blocks[0])
#   - 3 for down_blocks[0] (2 resnets + downsampler) -> controlnet_down_blocks[1-3]
#   - 3 for down_blocks[1] (2 resnets + downsampler) -> controlnet_down_blocks[4-6]
#   - 2 for down_blocks[2] (2 resnets, NO downsampler - last block) -> controlnet_down_blocks[7-8]
#       - Split into DeltaDown2A (resnet[0] -> residual 7) and DeltaDown2B (resnet[1] -> residual 8)
#   - 1 for mid_block -> controlnet_mid_block
# Total: 1 + 3 + 3 + 2 + 1 = 10

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional


# =============================================================================
# Chunk 0: AlphaTimeEmbed - Time embedding with sin/cos operators
# =============================================================================

class ControlNetAlphaTimeEmbedChunk(nn.Module):
    """
    Chunk 0: AlphaTimeEmbed - Time and control type embedding computation

    This chunk is isolated to allow running on CPU/GPU if Neural Engine
    has issues with sin/cos operators (e.g., on M1 iPad Pro).

    Contains:
        - time_proj: Timestep projection (sin/cos)
        - time_embedding: Timestep MLP
        - control_type_proj: Control type projection (sin/cos)
        - control_add_embedding: Control type MLP

    Inputs:
        - timestep: [B] timestep values
        - control_type: [B, num_control_type] control type flags

    Outputs:
        - emb: [B, 1280, 1, 1] time embedding (4D for Conv2d compatibility)
    """

    def __init__(self, controlnet):
        super().__init__()
        # Time embedding
        self.time_proj = controlnet.time_proj
        self.time_embedding = controlnet.time_embedding

        # Control type embedding
        self.control_type_proj = controlnet.control_type_proj
        self.control_add_embedding = controlnet.control_add_embedding

    def forward(
        self,
        timestep: torch.Tensor,
        control_type: torch.Tensor,
    ):
        # 1. Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=torch.float32)
        emb = self.time_embedding(t_emb)

        # 2. Control type embedding
        control_embeds = self.control_type_proj(control_type.flatten())
        control_embeds = control_embeds.reshape((timestep.shape[0], -1))
        control_embeds = control_embeds.to(emb.dtype)
        control_emb = self.control_add_embedding(control_embeds)
        emb = emb + control_emb

        return (emb,)


# =============================================================================
# Chunk 1: BetaCondFusion - Conditioning fusion (no sin/cos)
# =============================================================================

class ControlNetBetaCondFusionChunkBase(nn.Module):
    """
    Base class for Chunk 1: BetaCondFusion - Conditioning fusion

    Contains:
        - conv_in: Initial convolution on sample
        - controlnet_cond_embedding: Control image embedding
        - task_embedding: Per-control-type learnable embeddings
        - transformer_layes: Feature fusion transformer
        - spatial_ch_projs: Spatial channel projections

    Inputs:
        - sample: [B, 4, H, W] latent input
        - controlnet_cond_0 to _N: [B, 3, H*8, W*8] control images
        - conditioning_scale: [num_conds] per-condition scaling

    Outputs:
        - fused_sample: [B, 320, H, W] sample fused with conditioning
    """

    def __init__(self, controlnet, num_control_type=6):
        super().__init__()
        # Input convolution
        self.conv_in = controlnet.conv_in

        # Conditioning embedding
        self.controlnet_cond_embedding = controlnet.controlnet_cond_embedding
        self.task_embedding = controlnet.task_embedding
        self.transformer_layes = controlnet.transformer_layes
        self.spatial_ch_projs = controlnet.spatial_ch_projs

        self.num_control_type = num_control_type

    def forward_impl(
        self,
        sample: torch.Tensor,
        controlnet_cond: List[torch.Tensor],
        conditioning_scale: torch.Tensor,
    ):
        """
        Internal forward implementation that takes a list of controlnet_cond tensors.

        NOTE: We must use tensor indexing (conditioning_scale[i]) instead of .tolist()
        because .tolist() is not traceable - it converts to Python values at JIT trace time,
        causing the scale values to be baked in as constants.
        """
        num_cond = len(controlnet_cond)

        # 1. Conv_in on sample
        sample = self.conv_in(sample)

        # 2. Conditioning embedding and fusion
        inputs = []
        condition_list = []

        for i, cond in enumerate(controlnet_cond):
            # Use tensor indexing to keep it traceable (NOT .tolist())
            scale = conditioning_scale[i]
            condition = self.controlnet_cond_embedding(cond)
            feat_seq = torch.mean(condition, dim=(2, 3))
            feat_seq = feat_seq + self.task_embedding[i]
            if num_cond == 1:
                inputs.append(feat_seq.unsqueeze(1))
                condition_list.append(condition)
            else:
                # Scale must be reshaped for broadcasting: [1, 1, 1, 1] for 4D condition tensor
                scale_4d = scale.reshape(1, 1, 1, 1)
                inputs.append(feat_seq.unsqueeze(1) * scale)
                condition_list.append(condition * scale_4d)

        # Add sample features
        condition = sample
        feat_seq = torch.mean(condition, dim=(2, 3))
        inputs.append(feat_seq.unsqueeze(1))
        condition_list.append(condition)

        # Transformer fusion
        x = torch.cat(inputs, dim=1)
        for layer in self.transformer_layes:
            x = layer(x)

        # Spatial fusion
        controlnet_cond_fuser = sample * 0.0
        for idx, condition in enumerate(condition_list[:-1]):
            # Use tensor indexing to keep it traceable (NOT .tolist())
            scale = conditioning_scale[idx]
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            if num_cond == 1:
                controlnet_cond_fuser += condition + alpha
            else:
                # Scale must be reshaped for broadcasting: [1, 1, 1, 1] for 4D tensors
                scale_4d = scale.reshape(1, 1, 1, 1)
                controlnet_cond_fuser += condition + alpha * scale_4d

        fused_sample = sample + controlnet_cond_fuser

        return (fused_sample,)


def make_beta_cond_fusion_chunk(controlnet, num_control_type: int):
    """
    Factory function to create ControlNetBetaCondFusionChunk with dynamic forward signature.

    This generates a forward method with the exact number of controlnet_cond_* arguments
    based on num_control_type, similar to how controlnetunion.make_controlnet works.
    """
    # Generate argument names for controlnet_cond inputs
    cond_args = ", ".join([f"controlnet_cond_{i}: torch.Tensor"
                           for i in range(num_control_type)])
    cond_list = ", ".join([f"controlnet_cond_{i}"
                           for i in range(num_control_type)])

    # Generate forward function source code
    src = f"""
def forward(
    self,
    sample: torch.Tensor,
    {cond_args},
    conditioning_scale: torch.Tensor,
):
    controlnet_cond = [{cond_list}]
    return ControlNetBetaCondFusionChunkBase.forward_impl(
        self,
        sample,
        controlnet_cond,
        conditioning_scale,
    )
"""

    # Execute to create the forward function
    namespace = {"torch": torch, "ControlNetBetaCondFusionChunkBase": ControlNetBetaCondFusionChunkBase}
    exec(src, namespace)
    forward_fn = namespace["forward"]

    # Create wrapper class with dynamic forward
    class ControlNetBetaCondFusionChunk(ControlNetBetaCondFusionChunkBase):
        pass

    ControlNetBetaCondFusionChunk.forward = forward_fn

    return ControlNetBetaCondFusionChunk(controlnet, num_control_type)


# =============================================================================
# Chunk 2: GammaDown01 - Down blocks 0 and 1
# =============================================================================

class ControlNetGammaDown01Chunk(nn.Module):
    """
    Chunk 2: GammaDown01 - Down blocks 0 and 1 with their ControlNet projections

    Contains:
        - down_blocks[0]: CrossAttnDownBlock2D, 320->320 (2 resnets + downsampler)
        - down_blocks[1]: CrossAttnDownBlock2D, 320->640 (2 resnets + downsampler)
        - controlnet_down_blocks[0-6]: 1x1 Conv2d projections

    Inputs:
        - fused_sample: [B, 320, H, W] from BetaCondFusion
        - emb: [B, 1280, 1, 1] time embedding (4D for Conv2d compatibility)
        - encoder_hidden_states: [B, dim, 1, seq] text embeddings (4D conv format)

    Outputs:
        - additional_residual_0 to _6: ControlNet residuals for UNet
        - hidden: [B, 640, H/4, W/4] for next chunk
    """

    def __init__(self, controlnet):
        super().__init__()
        self.down_blocks_0 = controlnet.down_blocks[0]
        self.down_blocks_1 = controlnet.down_blocks[1]

        # ControlNet projection layers for these blocks
        # Index 0 is for initial sample (from chunk 0)
        # Indices 1-3 are for down_blocks[0] (2 resnets + 1 downsampler)
        # Indices 4-6 are for down_blocks[1] (2 resnets + 1 downsampler)
        self.controlnet_down_block_0 = controlnet.controlnet_down_blocks[0]
        self.controlnet_down_blocks_1_3 = nn.ModuleList([
            controlnet.controlnet_down_blocks[i] for i in range(1, 4)
        ])
        self.controlnet_down_blocks_4_6 = nn.ModuleList([
            controlnet.controlnet_down_blocks[i] for i in range(4, 7)
        ])

    def forward(
        self,
        fused_sample: torch.Tensor,
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # Store initial sample residual
        down_block_res_samples = [fused_sample]
        sample = fused_sample

        # Down block 0
        if hasattr(self.down_blocks_0, "attentions") and self.down_blocks_0.attentions:
            sample, res_samples = self.down_blocks_0(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = self.down_blocks_0(hidden_states=sample, temb=emb)
        down_block_res_samples.extend(res_samples)

        # Down block 1
        if hasattr(self.down_blocks_1, "attentions") and self.down_blocks_1.attentions:
            sample, res_samples = self.down_blocks_1(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = self.down_blocks_1(hidden_states=sample, temb=emb)
        down_block_res_samples.extend(res_samples)

        # Apply ControlNet projections
        residual_0 = self.controlnet_down_block_0(down_block_res_samples[0])

        residuals_1_3 = []
        for i, proj in enumerate(self.controlnet_down_blocks_1_3):
            residuals_1_3.append(proj(down_block_res_samples[1 + i]))

        residuals_4_6 = []
        for i, proj in enumerate(self.controlnet_down_blocks_4_6):
            residuals_4_6.append(proj(down_block_res_samples[4 + i]))

        return (
            residual_0,
            *residuals_1_3,
            *residuals_4_6,
            sample,  # hidden for next chunk
        )


# =============================================================================
# Chunk 3: DeltaDown2A - First resnet+attention of down_blocks[2]
# =============================================================================

class ControlNetDeltaDown2AChunk(nn.Module):
    """
    Chunk 3: DeltaDown2A - First resnet + attention of down_blocks[2]

    This is the first half of down_blocks[2], split at the resnet boundary
    to reduce chunk size from ~1.5GB to ~750MB.

    Contains:
        - down_blocks[2].resnets[0]: First ResnetBlock2D (640->1280 channels)
        - down_blocks[2].attentions[0]: First SpatialTransformer (attention)
        - controlnet_down_blocks[7]: 1x1 Conv2d projection

    Inputs:
        - hidden: [B, 640, H/4, W/4] from GammaDown01
        - emb: [B, 1280, 1, 1] time embedding (4D for Conv2d compatibility)
        - encoder_hidden_states: [B, dim, 1, seq] text embeddings (4D conv format)

    Outputs:
        - additional_residual_7: ControlNet residual for UNet
        - hidden: [B, 1280, H/4, W/4] for next chunk (DeltaDown2B)
    """

    def __init__(self, controlnet):
        super().__init__()
        # First resnet + attention from down_blocks[2]
        self.resnet_0 = controlnet.down_blocks[2].resnets[0]
        self.attn_0 = controlnet.down_blocks[2].attentions[0]

        # First projection for down_blocks[2]
        self.controlnet_down_block_7 = controlnet.controlnet_down_blocks[7]

    def forward(
        self,
        hidden: torch.Tensor,
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # First resnet (640 -> 1280 channels)
        hidden = self.resnet_0(hidden, emb)

        # First attention (SpatialTransformer uses 'context' parameter, not 'encoder_hidden_states')
        hidden = self.attn_0(hidden, context=encoder_hidden_states)

        # Apply ControlNet projection
        residual_7 = self.controlnet_down_block_7(hidden)

        return (residual_7, hidden)


# =============================================================================
# Chunk 4: DeltaDown2B - Second resnet+attention of down_blocks[2]
# =============================================================================

class ControlNetDeltaDown2BChunk(nn.Module):
    """
    Chunk 4: DeltaDown2B - Second resnet + attention of down_blocks[2]

    This is the second half of down_blocks[2], split at the resnet boundary
    to reduce chunk size from ~1.5GB to ~750MB.

    Contains:
        - down_blocks[2].resnets[1]: Second ResnetBlock2D (1280->1280 channels)
        - down_blocks[2].attentions[1]: Second SpatialTransformer (attention)
        - controlnet_down_blocks[8]: 1x1 Conv2d projection

    Inputs:
        - hidden: [B, 1280, H/4, W/4] from DeltaDown2A
        - emb: [B, 1280, 1, 1] time embedding (4D for Conv2d compatibility)
        - encoder_hidden_states: [B, dim, 1, seq] text embeddings (4D conv format)

    Outputs:
        - additional_residual_8: ControlNet residual for UNet
        - hidden: [B, 1280, H/4, W/4] for next chunk (EpsilonMid)
    """

    def __init__(self, controlnet):
        super().__init__()
        # Second resnet + attention from down_blocks[2]
        self.resnet_1 = controlnet.down_blocks[2].resnets[1]
        self.attn_1 = controlnet.down_blocks[2].attentions[1]

        # Second projection for down_blocks[2]
        self.controlnet_down_block_8 = controlnet.controlnet_down_blocks[8]

    def forward(
        self,
        hidden: torch.Tensor,
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # Second resnet (1280 -> 1280 channels)
        hidden = self.resnet_1(hidden, emb)

        # Second attention (SpatialTransformer uses 'context' parameter, not 'encoder_hidden_states')
        hidden = self.attn_1(hidden, context=encoder_hidden_states)

        # Apply ControlNet projection
        residual_8 = self.controlnet_down_block_8(hidden)

        return (residual_8, hidden)


# =============================================================================
# Chunk 5: EpsilonMid - Mid block
# =============================================================================

class ControlNetEpsilonMidChunk(nn.Module):
    """
    Chunk 5: EpsilonMid - Mid block with its ControlNet projection

    Contains:
        - mid_block: UNetMidBlock2DCrossAttn at 1280 channels
        - controlnet_mid_block: 1x1 Conv2d projection

    Inputs:
        - hidden: [B, 1280, H/4, W/4] from DeltaDown2B
        - emb: [B, 1280, 1, 1] time embedding (4D for Conv2d compatibility)
        - encoder_hidden_states: [B, dim, 1, seq] text embeddings (4D conv format)

    Outputs:
        - additional_residual_9: Final ControlNet residual for UNet mid block
    """

    def __init__(self, controlnet):
        super().__init__()
        self.mid_block = controlnet.mid_block
        self.controlnet_mid_block = controlnet.controlnet_mid_block

    def forward(
        self,
        hidden: torch.Tensor,
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        # Mid block processing
        sample = self.mid_block(
            hidden,
            emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Apply ControlNet projection
        mid_block_res_sample = self.controlnet_mid_block(sample)

        return (mid_block_res_sample,)


# =============================================================================
# Chunk names and classes for conversion
# =============================================================================

# Chunk names for ControlNet Union conversion (6 chunks for 3-down-block architecture)
CONTROLNET_UNION_CHUNK_NAMES = [
    "ControlNetAlphaTimeEmbed",   # Chunk 0: time + control embedding (sin/cos)
    "ControlNetBetaCondFusion",   # Chunk 1: conditioning fusion
    "ControlNetGammaDown01",      # Chunk 2: down_blocks[0,1] with projections
    "ControlNetDeltaDown2A",      # Chunk 3: down_blocks[2] first resnet+attn
    "ControlNetDeltaDown2B",      # Chunk 4: down_blocks[2] second resnet+attn
    "ControlNetEpsilonMid",       # Chunk 5: mid_block with projection
]

# Note: AlphaTimeEmbed and BetaCondFusion are not in this dict because:
# - AlphaTimeEmbed uses direct instantiation
# - BetaCondFusion requires the factory function make_beta_cond_fusion_chunk()
CONTROLNET_UNION_CHUNK_CLASSES = {
    "ControlNetAlphaTimeEmbed": ControlNetAlphaTimeEmbedChunk,
    "ControlNetGammaDown01": ControlNetGammaDown01Chunk,
    "ControlNetDeltaDown2A": ControlNetDeltaDown2AChunk,
    "ControlNetDeltaDown2B": ControlNetDeltaDown2BChunk,
    "ControlNetEpsilonMid": ControlNetEpsilonMidChunk,
}


def get_controlnet_union_chunk_output_names(chunk_name, controlnet=None):
    """
    Get output tensor names for each ControlNet Union architectural chunk.

    Args:
        chunk_name: Name of the chunk
        controlnet: Optional reference controlnet model (not used in 6-chunk architecture)

    6-chunk architecture (10 residuals total for 3-down-block ControlNet Union):
        - AlphaTimeEmbed: emb_out
        - BetaCondFusion: fused_sample_out
        - GammaDown01: additional_residual_0 to _6, hidden_out (7 residuals)
        - DeltaDown2A: additional_residual_7, hidden_out (1 residual)
        - DeltaDown2B: additional_residual_8, hidden_out (1 residual)
        - EpsilonMid: additional_residual_9 (1 residual)

    Total: 7 + 1 + 1 + 1 = 10 residuals (indices 0-9)
    """
    if chunk_name == "ControlNetAlphaTimeEmbed":
        return ["emb_out"]

    if chunk_name == "ControlNetBetaCondFusion":
        return ["fused_sample_out"]

    if chunk_name == "ControlNetGammaDown01":
        # 7 residuals: conv_in (1) + down_blocks[0] (3) + down_blocks[1] (3)
        return [f"additional_residual_{i}" for i in range(7)] + ["hidden_out"]

    if chunk_name == "ControlNetDeltaDown2A":
        # 1 residual: first resnet+attention of down_blocks[2]
        return ["additional_residual_7", "hidden_out"]

    if chunk_name == "ControlNetDeltaDown2B":
        # 1 residual: second resnet+attention of down_blocks[2]
        return ["additional_residual_8", "hidden_out"]

    if chunk_name == "ControlNetEpsilonMid":
        # 1 residual: mid_block
        return ["additional_residual_9"]

    return ["output"]
