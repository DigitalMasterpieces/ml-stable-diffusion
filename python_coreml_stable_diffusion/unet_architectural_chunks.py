#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.
#
# Architectural chunking for SDXL UNet
# Splits UNet into semantically meaningful chunks instead of equal-weight bisection

import torch
import torch.nn as nn
from collections import OrderedDict


class SDXLAlphaEncoderAChunk(nn.Module):
    """
    Chunk 0A: AlphaEncoderA - Time embedding computation only
    Contains: time_proj + time_embedding + add_time_proj + add_embedding

    This chunk is isolated to allow running on CPU+GPU if Neural Engine
    has issues with sin/cos operators (e.g., on M1 iPad Pro).

    Inputs:
        - timestep: [B] timestep values
        - time_ids: [B, 6] SDXL time conditioning
        - text_embeds: [B, 1280] pooled text embeddings

    Outputs:
        - emb: [B, 1280, 1, 1] time embedding
    """

    def __init__(self, unet):
        super().__init__()
        # Time embedding components
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding
        self.add_time_proj = unet.add_time_proj
        self.add_embedding = unet.add_embedding

    def forward(
        self,
        timestep,
        time_ids,
        text_embeds,
    ):
        # 1. Time embedding computation
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        # 2. Add embedding (SDXL specific)
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        aug_emb = self.add_embedding(add_embeds)
        emb = emb + aug_emb

        return (emb,)


class SDXLAlphaEncoderBChunk(nn.Module):
    """
    Chunk 0B: AlphaEncoderB - IP-Adapter processing + conv_in + down_blocks[0,1]
    Contains: encoder_hid_proj + conv_in + down_blocks[0] + down_blocks[1]

    Inputs:
        - sample: [B, 4, H, W] latent input
        - emb: [B, 1280, 1, 1] time embedding (from AlphaEncoderA)
        - encoder_hidden_states: [B, 2048, 1, 77] text embeddings
        - image_embeds: [B, seq, dim] optional IP-Adapter embeddings
        - ip_adapter_scale: [1] optional IP-Adapter scale
        - additional_residual_0 to _6: [optional] ControlNet residuals for conv_in + down_blocks[0,1]

    Outputs:
        - ip_hidden_states: [B, 2048, 1, 16] processed IP-Adapter embeddings (zeros if not used)
        - skip_conv_in: [B, 320, H, W] conv_in output (with ControlNet residual if provided)
        - skip_down0_0: [B, 320, H, W] skip from down_blocks[0] layer 0
        - skip_down0_1: [B, 320, H, W] skip from down_blocks[0] layer 1
        - skip_down0_2: [B, 320, H/2, W/2] skip from down_blocks[0] downsampler
        - skip_down1_0: [B, 640, H/2, W/2] skip from down_blocks[1] layer 0
        - skip_down1_1: [B, 640, H/2, W/2] skip from down_blocks[1] layer 1
        - skip_down1_2: [B, 640, H/4, W/4] skip from down_blocks[1] downsampler
        - hidden: [B, 640, H/4, W/4] main path output
    """

    def __init__(self, unet):
        super().__init__()
        # IP-Adapter support
        self.support_image_prompt = unet.support_image_prompt
        self.encoder_hid_proj = unet.encoder_hid_proj

        # ControlNet support
        self.support_controlnet = unet.support_controlnet

        # Input convolution
        self.conv_in = unet.conv_in

        # Down blocks 0 and 1
        self.down_blocks_0 = unet.down_blocks[0]  # DownBlock2D: 320 -> 320
        self.down_blocks_1 = unet.down_blocks[1]  # CrossAttnDownBlock2D: 320 -> 640

        # Check if blocks have attention (for defensive coding)
        self.down_blocks_0_has_attn = hasattr(self.down_blocks_0, 'attentions') and self.down_blocks_0.attentions is not None
        self.down_blocks_1_has_attn = hasattr(self.down_blocks_1, 'attentions') and self.down_blocks_1.attentions is not None

        # Store config for reference
        self.config = unet.config

    def forward(
        self,
        sample,
        emb,
        encoder_hidden_states,
        image_embeds=None,
        ip_adapter_scale=None,
        # ControlNet residuals for conv_in + down_blocks[0] + down_blocks[1]
        # Indices: 0=conv_in, 1-3=down_blocks[0], 4-6=down_blocks[1]
        additional_residual_0=None,
        additional_residual_1=None,
        additional_residual_2=None,
        additional_residual_3=None,
        additional_residual_4=None,
        additional_residual_5=None,
        additional_residual_6=None,
    ):
        # 1. IP-Adapter processing (if enabled)
        # Process image embeddings and output separately for downstream chunks
        if self.support_image_prompt and image_embeds is not None:
            image_embeds_list = self.encoder_hid_proj(image_embeds)
            ip_hidden_states = image_embeds_list[0].transpose(1, 2).transpose(1, 3)
            # Create tuple for internal use in down blocks
            encoder_hidden_states_for_blocks = (encoder_hidden_states, ip_hidden_states)
        else:
            # Create dummy ip_hidden_states for consistent output shape
            # Shape: [B, cross_attention_dim, 1, num_ip_tokens]
            ip_hidden_states = torch.zeros(
                encoder_hidden_states.shape[0],  # batch
                encoder_hidden_states.shape[1],  # cross_attention_dim
                1,
                16,  # num_ip_tokens for IP-Adapter Plus
                device=encoder_hidden_states.device,
                dtype=encoder_hidden_states.dtype
            )
            encoder_hidden_states_for_blocks = encoder_hidden_states

        # 2. Input convolution
        sample = self.conv_in(sample)

        # Store conv_in output as first skip (for ControlNet compatibility)
        # Apply ControlNet residual if provided
        skip_conv_in = sample
        if self.support_controlnet and additional_residual_0 is not None:
            skip_conv_in = skip_conv_in + additional_residual_0

        # 3. Down block 0: 320 channels, produces 3 skips
        # DownBlock2D returns (hidden_states, output_states_tuple)
        # Note: SDXL down_blocks[0] is DownBlock2D (no attention), blocks 1,2 have attention
        if self.down_blocks_0_has_attn:
            # Check if ip_adapter_scale is supported (custom UNet vs standard diffusers)
            import inspect
            sig = inspect.signature(self.down_blocks_0.forward)
            if 'ip_adapter_scale' in sig.parameters:
                sample, res_samples_0 = self.down_blocks_0(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_blocks,
                    ip_adapter_scale=ip_adapter_scale,
                )
            else:
                sample, res_samples_0 = self.down_blocks_0(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_blocks,
                )
        else:
            sample, res_samples_0 = self.down_blocks_0(
                hidden_states=sample,
                temb=emb,
            )
        # res_samples_0 contains: (after_layer0, after_layer1, after_downsampler)

        # Apply ControlNet residuals to down_blocks[0] outputs if provided
        if self.support_controlnet:
            controlnet_residuals_0 = [additional_residual_1, additional_residual_2, additional_residual_3]
            res_samples_0 = tuple(
                res + ctrl if ctrl is not None else res
                for res, ctrl in zip(res_samples_0, controlnet_residuals_0)
            )

        # 4. Down block 1: 320->640 channels, produces 3 skips
        if self.down_blocks_1_has_attn:
            # Check if ip_adapter_scale is supported (custom UNet vs standard diffusers)
            import inspect
            sig = inspect.signature(self.down_blocks_1.forward)
            if 'ip_adapter_scale' in sig.parameters:
                sample, res_samples_1 = self.down_blocks_1(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_blocks,
                    ip_adapter_scale=ip_adapter_scale,
                )
            else:
                sample, res_samples_1 = self.down_blocks_1(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_blocks,
                )
        else:
            sample, res_samples_1 = self.down_blocks_1(
                hidden_states=sample,
                temb=emb,
            )
        # res_samples_1 contains: (after_layer0, after_layer1, after_downsampler)

        # Apply ControlNet residuals to down_blocks[1] outputs if provided
        if self.support_controlnet:
            controlnet_residuals_1 = [additional_residual_4, additional_residual_5, additional_residual_6]
            res_samples_1 = tuple(
                res + ctrl if ctrl is not None else res
                for res, ctrl in zip(res_samples_1, controlnet_residuals_1)
            )

        # Return all outputs
        # Note: We output ip_hidden_states separately so downstream chunks can use it
        # emb is already [B, 1280, 1, 1] from custom unet.py's TimestepEmbedding (uses Conv2d)
        # Return format: (ip_hidden_states, skip_conv_in, *res_samples_0, *res_samples_1, hidden)
        return (
            ip_hidden_states,         # IP-Adapter hidden states [B, cross_attn_dim, 1, num_tokens]
            skip_conv_in,             # [B, 320, H, W] - conv_in output (with ControlNet residual)
            *res_samples_0,           # down_blocks[0] residuals (with ControlNet residuals)
            *res_samples_1,           # down_blocks[1] residuals (with ControlNet residuals)
            sample,                   # main path output
        )


class SDXLGammaDownblockChunk(nn.Module):
    """
    Chunk 2: GammaDownblock
    Contains: down_blocks[2] (CrossAttnDownBlock2D: 640 -> 1280)

    Note: SDXL only has 3 down_blocks (indices 0, 1, 2), not 4 like SD 1.x/2.x.
    down_blocks[2] is the final down block before mid_block.

    Inputs:
        - hidden: [B, 640, H/4, W/4]
        - emb: [B, 1280, 1, 1]
        - encoder_hidden_states: text embeddings [B, hidden, 1, seq_len]
        - ip_hidden_states: IP-Adapter image embeddings (optional, for IP-Adapter)
        - ip_adapter_scale: optional
        - additional_residual_7, _8: [optional] ControlNet residuals for down_blocks[2]

    Outputs:
        - skip_down2_0: [B, 1280, H/4, W/4] skip from layer 0 (with ControlNet residual)
        - skip_down2_1: [B, 1280, H/4, W/4] skip from layer 1 (with ControlNet residual)
        - hidden: [B, 1280, H/8, W/8] main path output (after downsampler)
    """

    def __init__(self, unet):
        super().__init__()
        self.down_blocks_2 = unet.down_blocks[2]  # CrossAttnDownBlock2D: 640 -> 1280
        self.has_attention = hasattr(self.down_blocks_2, 'attentions') and self.down_blocks_2.attentions is not None
        self.support_image_prompt = unet.support_image_prompt
        self.support_controlnet = unet.support_controlnet

    def forward(
        self,
        hidden,
        emb,
        encoder_hidden_states,
        ip_hidden_states=None,
        ip_adapter_scale=None,
        # ControlNet residuals for down_blocks[2] (indices 7-8 in the full sequence)
        additional_residual_7=None,
        additional_residual_8=None,
    ):
        # emb is [B, 1280, 1, 1] - custom unet.py ResnetBlock2D expects 4D temb for Conv2d
        # Combine encoder_hidden_states and ip_hidden_states into tuple if IP-Adapter is enabled
        if self.support_image_prompt and ip_hidden_states is not None:
            encoder_hidden_states_for_block = (encoder_hidden_states, ip_hidden_states)
        else:
            encoder_hidden_states_for_block = encoder_hidden_states

        if self.has_attention:
            # Check if ip_adapter_scale is supported (custom UNet vs standard diffusers)
            import inspect
            sig = inspect.signature(self.down_blocks_2.forward)
            if 'ip_adapter_scale' in sig.parameters:
                sample, res_samples = self.down_blocks_2(
                    hidden_states=hidden,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                    ip_adapter_scale=ip_adapter_scale,
                )
            else:
                sample, res_samples = self.down_blocks_2(
                    hidden_states=hidden,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                )
        else:
            sample, res_samples = self.down_blocks_2(
                hidden_states=hidden,
                temb=emb,
            )

        # Apply ControlNet residuals to down_blocks[2] outputs if provided
        # down_blocks[2] has 2 resnets and NO downsampler in SDXL
        if self.support_controlnet:
            controlnet_residuals = [additional_residual_7, additional_residual_8]
            res_samples = tuple(
                res + ctrl if ctrl is not None else res
                for res, ctrl in zip(res_samples, controlnet_residuals)
            )

        # Return all residual samples + main path
        # Number of residuals depends on layers_per_block and whether there's a downsampler
        return (*res_samples, sample)


class SDXLSigmaCoreChunk(nn.Module):
    """
    Chunk 3: SigmaCore
    Contains: mid_block (UNetMidBlock2DCrossAttn)

    Note: This is now Chunk 3 (was Chunk 4) since SDXLDeltaTransformer was removed.
    SDXL only has 3 down_blocks, so we go directly from GammaDownblock to SigmaCore.

    Inputs:
        - hidden: [B, 1280, H/8, W/8]
        - emb: [B, 1280, 1, 1]
        - encoder_hidden_states: text embeddings [B, hidden, 1, seq_len]
        - ip_hidden_states: IP-Adapter image embeddings (optional, for IP-Adapter)
        - ip_adapter_scale: optional
        - additional_residual_9: [optional] ControlNet residual for mid_block

    Outputs:
        - hidden: [B, 1280, H/8, W/8] processed mid block output (with ControlNet residual)
    """

    def __init__(self, unet):
        super().__init__()
        self.mid_block = unet.mid_block
        self.has_attention = hasattr(self.mid_block, 'attentions') and self.mid_block.attentions is not None
        self.support_image_prompt = unet.support_image_prompt
        self.support_controlnet = unet.support_controlnet

    def forward(
        self,
        hidden,
        emb,
        encoder_hidden_states,
        ip_hidden_states=None,
        ip_adapter_scale=None,
        # ControlNet residual for mid_block (index 9 in the full sequence)
        additional_residual_9=None,
    ):
        # emb is [B, 1280, 1, 1] - custom unet.py ResnetBlock2D expects 4D temb for Conv2d
        # Combine encoder_hidden_states and ip_hidden_states into tuple if IP-Adapter is enabled
        if self.support_image_prompt and ip_hidden_states is not None:
            encoder_hidden_states_for_block = (encoder_hidden_states, ip_hidden_states)
        else:
            encoder_hidden_states_for_block = encoder_hidden_states

        if self.has_attention:
            # Check if ip_adapter_scale is supported (custom UNet vs standard diffusers)
            import inspect
            sig = inspect.signature(self.mid_block.forward)
            if 'ip_adapter_scale' in sig.parameters:
                sample = self.mid_block(
                    hidden_states=hidden,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                    ip_adapter_scale=ip_adapter_scale,
                )
            else:
                sample = self.mid_block(
                    hidden_states=hidden,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                )
        else:
            sample = self.mid_block(
                hidden_states=hidden,
                temb=emb,
            )

        # Apply ControlNet residual to mid_block output if provided
        if self.support_controlnet and additional_residual_9 is not None:
            sample = sample + additional_residual_9

        return (sample,)


class SDXLThetaUpblockAChunk(nn.Module):
    """
    Chunk 4A: ThetaUpblockA - First layer of up_blocks[0]
    Contains: ResNet0 + Attention0 (first ResNet+Attention pair)

    This is the first half of the original ThetaUpblock, split for memory optimization.
    Processes the first skip connection from down_blocks[2] (the last residual).

    Skip connection ordering in up_blocks:
    - up_blocks consume skips in REVERSE order from the corresponding down_block
    - up_blocks[0] corresponds to down_blocks[2] which has 2 resnets (no downsampler)
    - resnet_0 gets skip_down2_1 (1280 ch), resnet_1 gets skip_down2_0 (1280 ch)
    - resnet_2 gets skip_down1_2 (640 ch, from down_blocks[1] downsampler)

    Inputs:
        - hidden: [B, 1280, H/4, W/4] from SigmaCore (mid_block)
        - emb: [B, 1280, 1, 1]
        - encoder_hidden_states: text embeddings [B, hidden, 1, seq_len]
        - ip_hidden_states: IP-Adapter image embeddings (optional)
        - skip_0: [B, 1280, H/4, W/4] from down_blocks[2] resnet 1 (skip_down2_1)
        - ip_adapter_scale: optional

    Outputs:
        - hidden_out: [B, 1280, H/4, W/4] after first ResNet+Attention
    """

    def __init__(self, unet):
        super().__init__()
        up_block = unet.up_blocks[0]
        self.resnet_0 = up_block.resnets[0]
        self.attn_0 = up_block.attentions[0]
        self.support_image_prompt = unet.support_image_prompt

    def forward(
        self,
        hidden,
        emb,
        encoder_hidden_states,
        ip_hidden_states,
        skip_0,
        ip_adapter_scale=None,
    ):
        # Combine encoder_hidden_states and ip_hidden_states into tuple if IP-Adapter is enabled
        if self.support_image_prompt and ip_hidden_states is not None:
            encoder_hidden_states_for_attn = (encoder_hidden_states, ip_hidden_states)
        else:
            encoder_hidden_states_for_attn = encoder_hidden_states

        # Concat hidden with skip_0 (from down_blocks[2] resnet 1, i.e., skip_down2_1)
        # hidden: [B, 1280, H/4, W/4], skip_0: [B, 1280, H/4, W/4] -> [B, 2560, H/4, W/4]
        hidden = torch.cat([hidden, skip_0], dim=1)

        # ResNet 0
        hidden = self.resnet_0(hidden, emb)

        # Attention 0
        hidden = self.attn_0(hidden, context=encoder_hidden_states_for_attn, ip_adapter_scale=ip_adapter_scale)

        return (hidden,)


class SDXLThetaUpblockBChunk(nn.Module):
    """
    Chunk 4B: ThetaUpblockB - Remaining layers of up_blocks[0]
    Contains: ResNet1 + Attention1 + ResNet2 + Attention2 + Upsampler

    This is the second half of the original ThetaUpblock, split for memory optimization.
    Processes the remaining two skip connections and performs upsampling.

    Skip connection ordering (continuing from ThetaUpblockA):
    - resnet_1 gets skip_down2_0 (1280 ch, from down_blocks[2] resnet 0)
    - resnet_2 gets skip_down1_2 (640 ch, from down_blocks[1] downsampler)

    Inputs:
        - hidden: [B, 1280, H/4, W/4] from ThetaUpblockA
        - emb: [B, 1280, 1, 1]
        - encoder_hidden_states: text embeddings [B, hidden, 1, seq_len]
        - ip_hidden_states: IP-Adapter image embeddings (optional)
        - skip_0: [B, 1280, H/4, W/4] from down_blocks[2] resnet 0 (skip_down2_0)
        - skip_1: [B, 640, H/4, W/4] from down_blocks[1] downsampler (skip_down1_2)
        - ip_adapter_scale: optional

    Outputs:
        - hidden_out: [B, 1280, H/2, W/2] upsampled output
    """

    def __init__(self, unet):
        super().__init__()
        up_block = unet.up_blocks[0]
        self.resnet_1 = up_block.resnets[1]
        self.attn_1 = up_block.attentions[1]
        self.resnet_2 = up_block.resnets[2]
        self.attn_2 = up_block.attentions[2]
        self.upsampler = up_block.upsamplers[0]
        self.support_image_prompt = unet.support_image_prompt

    def forward(
        self,
        hidden,
        emb,
        encoder_hidden_states,
        ip_hidden_states,
        skip_0,
        skip_1,
        ip_adapter_scale=None,
    ):
        # Combine encoder_hidden_states and ip_hidden_states into tuple if IP-Adapter is enabled
        if self.support_image_prompt and ip_hidden_states is not None:
            encoder_hidden_states_for_attn = (encoder_hidden_states, ip_hidden_states)
        else:
            encoder_hidden_states_for_attn = encoder_hidden_states

        # Layer 1: concat with skip_0 (skip_down2_0), then ResNet + Attention
        # hidden: [B, 1280, H/4, W/4], skip_0: [B, 1280, H/4, W/4] -> [B, 2560, H/4, W/4]
        hidden = torch.cat([hidden, skip_0], dim=1)
        hidden = self.resnet_1(hidden, emb)
        hidden = self.attn_1(hidden, context=encoder_hidden_states_for_attn, ip_adapter_scale=ip_adapter_scale)

        # Layer 2: concat with skip_1 (skip_down1_2 from down_blocks[1] downsampler), then ResNet + Attention
        # hidden: [B, 1280, H/4, W/4], skip_1: [B, 640, H/4, W/4] -> [B, 1920, H/4, W/4]
        hidden = torch.cat([hidden, skip_1], dim=1)
        hidden = self.resnet_2(hidden, emb)
        hidden = self.attn_2(hidden, context=encoder_hidden_states_for_attn, ip_adapter_scale=ip_adapter_scale)

        # Upsample: [B, 1280, H/4, W/4] -> [B, 1280, H/2, W/2]
        hidden = self.upsampler(hidden)

        return (hidden,)


class SDXLLambdaUpblockChunk(nn.Module):
    """
    Chunk 5: LambdaUpblock
    Contains: up_blocks[1] (CrossAttnUpBlock2D: 1280 -> 640)

    Note: This is now Chunk 5 (was Chunk 6) since SDXLDeltaTransformer was removed.
    SDXL up_blocks[1] is a CrossAttnUpBlock2D with attention (3 resnets + upsampler).

    Inputs:
        - hidden: [B, 1280, H/2, W/2] from ThetaUpblock (upsampled)
        - emb: [B, 1280, 1, 1]
        - encoder_hidden_states: text embeddings [B, hidden, 1, seq_len]
        - ip_hidden_states: IP-Adapter image embeddings (optional, for IP-Adapter)
        - skip_0: [B, 640, H/2, W/2] from down_blocks[1] resnet 1
        - skip_1: [B, 640, H/2, W/2] from down_blocks[1] resnet 0
        - skip_2: [B, 320, H/2, W/2] from down_blocks[0] downsampler
        - ip_adapter_scale: optional

    Outputs:
        - hidden: [B, 640, H, W] upsampled output
    """

    def __init__(self, unet):
        super().__init__()
        self.up_blocks_1 = unet.up_blocks[1]  # CrossAttnUpBlock2D: 1280 -> 1280
        self.has_attention = hasattr(self.up_blocks_1, 'attentions') and self.up_blocks_1.attentions is not None
        self.support_image_prompt = unet.support_image_prompt

    def forward(
        self,
        hidden,
        emb,
        encoder_hidden_states,
        ip_hidden_states,
        skip_0,
        skip_1,
        skip_2,
        ip_adapter_scale=None,
    ):
        # emb is [B, 1280, 1, 1] - custom unet.py ResnetBlock2D expects 4D temb for Conv2d
        # Combine encoder_hidden_states and ip_hidden_states into tuple if IP-Adapter is enabled
        if self.support_image_prompt and ip_hidden_states is not None:
            encoder_hidden_states_for_block = (encoder_hidden_states, ip_hidden_states)
        else:
            encoder_hidden_states_for_block = encoder_hidden_states

        res_hidden_states_tuple = (skip_2, skip_1, skip_0)

        if self.has_attention:
            # Check if ip_adapter_scale is supported (custom UNet vs standard diffusers)
            import inspect
            sig = inspect.signature(self.up_blocks_1.forward)
            if 'ip_adapter_scale' in sig.parameters:
                sample = self.up_blocks_1(
                    hidden_states=hidden,
                    res_hidden_states_tuple=res_hidden_states_tuple,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                    ip_adapter_scale=ip_adapter_scale,
                )
            else:
                sample = self.up_blocks_1(
                    hidden_states=hidden,
                    res_hidden_states_tuple=res_hidden_states_tuple,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states_for_block,
                )
        else:
            sample = self.up_blocks_1(
                hidden_states=hidden,
                res_hidden_states_tuple=res_hidden_states_tuple,
                temb=emb,
            )

        return (sample,)


class SDXLKappaUpblockChunk(nn.Module):
    """
    Chunk 6: KappaUpblock
    Contains: up_blocks[2] (UpBlock2D: 640 -> 320, no attention)

    Note: This is now Chunk 6 (was Chunk 7) since SDXLDeltaTransformer was removed.
    SDXL up_blocks[2] is an UpBlock2D without attention (3 resnets, no upsampler).

    Inputs:
        - hidden: [B, 640, H, W] from LambdaUpblock (upsampled)
        - emb: [B, 1280, 1, 1]
        - skip_0: [B, 320, H, W] from down_blocks[0] resnet 1
        - skip_1: [B, 320, H, W] from down_blocks[0] resnet 0
        - skip_2: [B, 320, H, W] from conv_in

    Outputs:
        - hidden: [B, 320, H, W] output (no upsampler on final block)
    """

    def __init__(self, unet):
        super().__init__()
        self.up_blocks_2 = unet.up_blocks[2]  # UpBlock2D: 640 -> 320

    def forward(
        self,
        hidden,
        emb,
        skip_0,
        skip_1,
        skip_2,
    ):
        # emb is [B, 1280, 1, 1] - custom unet.py ResnetBlock2D expects 4D temb for Conv2d
        # UpBlock2D doesn't take encoder_hidden_states
        res_hidden_states_tuple = (skip_2, skip_1, skip_0)

        sample = self.up_blocks_2(
            hidden_states=hidden,
            res_hidden_states_tuple=res_hidden_states_tuple,
            temb=emb,
        )

        return (sample,)


class SDXLOmegaDecoderChunk(nn.Module):
    """
    Chunk 7: OmegaDecoder
    Contains: conv_norm_out + conv_act + conv_out

    Note: This is now Chunk 7 (was Chunk 8) since SDXLDeltaTransformer was removed.
    SDXL only has 3 up_blocks (indices 0, 1, 2), so there is no up_blocks[3].
    This chunk only contains the final output convolutions.

    Inputs:
        - hidden: [B, 320, H, W] from KappaUpblock
        - skip_conv_in: [B, 320, H, W] from conv_in (optional residual)

    Outputs:
        - noise_pred: [B, 4, H, W] final noise prediction
    """

    def __init__(self, unet):
        super().__init__()
        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out

    def forward(
        self,
        hidden,
        skip_conv_in,
    ):
        # Add residual from conv_in if the architecture uses it
        # Note: Standard SDXL doesn't add skip_conv_in here, but we keep the input
        # for compatibility with architectures that might use it
        sample = hidden

        # Final output layers
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample,)


# ============================================================================
# Chunk configuration and metadata
# ============================================================================

ARCHITECTURAL_CHUNK_NAMES = [
    "SDXLAlphaEncoderA",    # Chunk 0: time embeddings only
    "SDXLAlphaEncoderB",    # Chunk 1: IP-Adapter + conv_in + down_blocks[0,1]
    "SDXLGammaDownblock",   # Chunk 2: down_blocks[2]
    "SDXLSigmaCore",        # Chunk 3: mid_block
    "SDXLThetaUpblockA",    # Chunk 4: up_blocks[0] layer 0 (ResNet0 + Attention0)
    "SDXLThetaUpblockB",    # Chunk 5: up_blocks[0] layers 1-2 + upsampler
    "SDXLLambdaUpblock",    # Chunk 6: up_blocks[1]
    "SDXLKappaUpblock",     # Chunk 7: up_blocks[2]
    "SDXLOmegaDecoder",     # Chunk 8: conv_norm_out + conv_act + conv_out
]

ARCHITECTURAL_CHUNK_CLASSES = {
    "SDXLAlphaEncoderA": SDXLAlphaEncoderAChunk,
    "SDXLAlphaEncoderB": SDXLAlphaEncoderBChunk,
    "SDXLGammaDownblock": SDXLGammaDownblockChunk,
    "SDXLSigmaCore": SDXLSigmaCoreChunk,
    "SDXLThetaUpblockA": SDXLThetaUpblockAChunk,
    "SDXLThetaUpblockB": SDXLThetaUpblockBChunk,
    "SDXLLambdaUpblock": SDXLLambdaUpblockChunk,
    "SDXLKappaUpblock": SDXLKappaUpblockChunk,
    "SDXLOmegaDecoder": SDXLOmegaDecoderChunk,
}


def _get_block_num_residuals(block):
    """Get the number of residual outputs from a down block."""
    num_resnets = len(block.resnets)
    has_downsampler = hasattr(block, 'downsamplers') and block.downsamplers is not None
    return num_resnets + (1 if has_downsampler else 0)


def get_architectural_chunk_output_names(chunk_name, unet=None):
    """
    Get output tensor names for each architectural chunk.

    Args:
        chunk_name: Name of the chunk
        unet: Optional UNet model to determine dynamic output counts
    """
    # Static output names for chunks with fixed outputs
    # Note: Output names must differ from input names to avoid CoreML conflicts
    static_output_names = {
        "SDXLAlphaEncoderA": ["emb_out"],
        "SDXLSigmaCore": ["hidden_out"],
        "SDXLThetaUpblockA": ["hidden_out"],
        "SDXLThetaUpblockB": ["hidden_out"],
        "SDXLLambdaUpblock": ["hidden_out"],
        "SDXLKappaUpblock": ["hidden_out"],
        "SDXLOmegaDecoder": ["noise_pred"],
    }

    if chunk_name in static_output_names:
        return static_output_names[chunk_name]

    # For chunks with variable outputs based on block configuration
    if unet is None:
        # Fallback to default SDXL config:
        # - down_blocks[0]: DownBlock2D, 2 resnets + downsampler = 3 skips
        # - down_blocks[1]: CrossAttnDownBlock2D, 2 resnets + downsampler = 3 skips
        # - down_blocks[2]: CrossAttnDownBlock2D, 2 resnets, NO downsampler = 2 skips
        # Output names must differ from input names to avoid CoreML conflicts
        default_output_names = {
            "SDXLAlphaEncoderB": [
                "ip_hidden_states_out",
                "skip_conv_in", "skip_down0_0", "skip_down0_1", "skip_down0_2",
                "skip_down1_0", "skip_down1_1", "skip_down1_2", "hidden_out"
            ],
            "SDXLGammaDownblock": [
                "skip_down2_0", "skip_down2_1", "hidden_out"  # No downsampler on down_blocks[2]
            ],
        }
        return default_output_names.get(chunk_name, ["output"])

    # Dynamic output names based on actual UNet configuration
    # Note: Output names must differ from input names to avoid CoreML conflicts
    if chunk_name == "SDXLAlphaEncoderB":
        num_res_0 = _get_block_num_residuals(unet.down_blocks[0])
        num_res_1 = _get_block_num_residuals(unet.down_blocks[1])
        names = ["ip_hidden_states_out", "skip_conv_in"]
        names.extend([f"skip_down0_{i}" for i in range(num_res_0)])
        names.extend([f"skip_down1_{i}" for i in range(num_res_1)])
        names.append("hidden_out")
        return names

    elif chunk_name == "SDXLGammaDownblock":
        num_res = _get_block_num_residuals(unet.down_blocks[2])
        names = [f"skip_down2_{i}" for i in range(num_res)]
        names.append("hidden_out")
        return names

    return ["output"]
