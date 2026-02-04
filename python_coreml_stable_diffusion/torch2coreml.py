#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Copyright (C) 2025 Digital Masterpieces GmbH. All Rights Reserved.
#

from python_coreml_stable_diffusion import (
    unet, controlnet, controlnetunion, chunk_mlprogram, unet_architectural_chunks
)

from python_coreml_stable_diffusion.unet import AttnProcessor2_0, IPAdapterAttnProcessor2_0, UNet2DConditionModelXLWithoutIPAdapter

import argparse
from collections import OrderedDict, defaultdict
from typing import Dict
from copy import deepcopy
import coremltools as ct
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    ControlNetModel,
    ControlNetUnionModel
)
from diffusionkit.tests.torch2coreml import (
    convert_mmdit_to_mlpackage,
    convert_vae_to_mlpackage
)
import gc
from huggingface_hub import snapshot_download, hf_hub_download

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
import requests
import shutil
import time
import re
import pathlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

torch.set_grad_enabled(False)

from transformers import CLIPVisionModel, CLIPVisionModelWithProjection

from types import MethodType


def _get_coreml_inputs(sample_inputs, args):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        ) for k, v in sample_inputs.items()
    ]


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


ABSOLUTE_MIN_PSNR = 35


def report_correctness(original_outputs, final_outputs, log_prefix):
    """ Report PSNR values across two compatible tensors
    """
    original_psnr = compute_psnr(original_outputs, original_outputs)
    final_psnr = compute_psnr(original_outputs, final_outputs)

    dB_change = final_psnr - original_psnr
    logger.info(
        f"{log_prefix}: PSNR changed by {dB_change:.1f} dB ({original_psnr:.1f} -> {final_psnr:.1f})"
    )

    if final_psnr < ABSOLUTE_MIN_PSNR:
        raise ValueError(f"{final_psnr:.1f} dB is too low!")
    else:
        logger.info(
            f"{final_psnr:.1f} dB > {ABSOLUTE_MIN_PSNR} dB (minimum allowed) parity check passed"
        )
    return final_psnr

def _get_out_path(args, submodule_name):
    fname = f"Stable_Diffusion_version_{args.model_version}_{submodule_name}.mlpackage"
    fname = fname.replace("/", "_")
    return os.path.join(args.o, fname)


def _convert_to_coreml(submodule_name, torchscript_module, sample_inputs,
                       output_names, args, out_path=None, precision=None, compute_unit=None):

    if out_path is None:
        out_path = _get_out_path(args, submodule_name)

    compute_unit = compute_unit or ct.ComputeUnit[args.compute_unit]

    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        logger.info(f"Loading model from {out_path}")

        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(
            out_path, compute_units=compute_unit)
        logger.info(
            f"Loading {out_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = compute_unit
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        deployment_target = _get_deployment_target(args.min_deployment_target)
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=deployment_target,
            inputs=_get_coreml_inputs(sample_inputs, args),
            outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
            compute_units=compute_unit,
            compute_precision=precision,
            skip_model_load=not args.check_output_correctness,
        )

        del torchscript_module
        gc.collect()

    return coreml_model, out_path


def _get_deployment_target(target_string):
    """
    Convert deployment target string to coremltools target object.
    
    Args:
        target_string (str): Target deployment string (e.g., "macOS13", "iOS18")
        
    Returns:
        coremltools target object
    """
    target_map = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    
    # Handle newer targets that might not be available in older coremltools versions
    try:
        if target_string == "macOS15":
            return ct.target.macOS15
        elif target_string == "iOS18":
            return ct.target.iOS18
    except AttributeError:
        logger.warning(f"Deployment target {target_string} not available in this coremltools version. "
                      f"Using macOS14 as fallback.")
        return ct.target.macOS14
    
    if target_string in target_map:
        return target_map[target_string]
    else:
        logger.warning(f"Unknown deployment target {target_string}. Using macOS13 as fallback.")
        return ct.target.macOS13


def quantize_weights(args):
    """ Quantize weights to args.quantize_nbits using a palette (look-up table)
    """
    # Optional inputs that need to be restored after quantization
    # (quantization loses the isOptional flag)
    controlnet_optional_inputs = ["additional_residual", "image_embeds"]

    for model_name in ["text_encoder", "text_encoder_2", "image_encoder", "unet", "refiner"]:
        logger.info(f"Quantizing {model_name} to {args.quantize_nbits}-bit precision")
        out_path = _get_out_path(args, model_name)
        _quantize_weights(
            out_path,
            model_name,
            args.quantize_nbits
        )

    # control-unet has optional ControlNet residual inputs and image_embeds
    logger.info(f"Quantizing control-unet to {args.quantize_nbits}-bit precision")
    out_path = _get_out_path(args, "control-unet")
    _quantize_weights(
        out_path,
        "control-unet",
        args.quantize_nbits,
        restore_optional_inputs=controlnet_optional_inputs
    )

    if args.convert_controlnet:
        for controlnet_model_version in args.convert_controlnet:
            controlnet_model_name = controlnet_model_version.replace("/", "_")
            logger.info(f"Quantizing {controlnet_model_name} to {args.quantize_nbits}-bit precision")
            fname = f"ControlNet_{controlnet_model_name}.mlpackage"
            out_path = os.path.join(args.o, fname)
            _quantize_weights(
                out_path,
                controlnet_model_name,
                args.quantize_nbits
            )

    # Quantize architectural chunks if they exist
    # Check if architectural chunking mode is enabled
    is_architectural = args.unet_chunks == "architectural"
    if is_architectural:
        from python_coreml_stable_diffusion.unet_architectural_chunks import ARCHITECTURAL_CHUNK_NAMES

        # Determine which chunks have optional ControlNet inputs
        # AlphaEncoder: additional_residual_0 through _6, image_embeds
        # GammaDownblock: additional_residual_7, _8
        # SigmaCore: additional_residual_9
        chunks_with_controlnet = {
            "SDXLAlphaEncoderB": ["additional_residual", "image_embeds"],
            "SDXLGammaDownblock": ["additional_residual"],
            "SDXLSigmaCore": ["additional_residual"],
        }

        for chunk_name in ARCHITECTURAL_CHUNK_NAMES:
            logger.info(f"Quantizing architectural chunk {chunk_name} to {args.quantize_nbits}-bit precision")
            out_path = _get_out_path(args, chunk_name)
            restore_inputs = chunks_with_controlnet.get(chunk_name, None)
            _quantize_weights(
                out_path,
                chunk_name,
                args.quantize_nbits,
                restore_optional_inputs=restore_inputs
            )

def _quantize_weights(out_path, model_name, nbits, restore_optional_inputs=None):
    """Quantize weights to nbits using palette (look-up table).

    Args:
        out_path: Path to the model
        model_name: Name of the model for logging
        nbits: Number of bits for quantization
        restore_optional_inputs: List of input name prefixes to restore as optional after quantization.
                                 Quantization loses the isOptional flag, so we need to restore it.
    """
    if os.path.exists(out_path):
        logger.info(f"Quantizing {model_name}")
        mlmodel = ct.models.MLModel(out_path,
                                    compute_units=ct.ComputeUnit.CPU_ONLY)

        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
        )

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config,
            op_type_configs={
                "gather": None # avoid quantizing the embedding table
            }
        )

        quantized_model = ct.optimize.coreml.palettize_weights(mlmodel, config=config)

        # Restore optional input flags if specified (quantization loses them)
        if restore_optional_inputs:
            spec = quantized_model.get_spec()
            for input_type in spec.description.input:
                for prefix in restore_optional_inputs:
                    if input_type.name.startswith(prefix):
                        input_type.type.isOptional = True
                        logger.info(f"Restored optional flag for input: {input_type.name}")
                        break
            quantized_model = ct.models.MLModel(spec, weights_dir=quantized_model.weights_dir)

        quantized_model.save(out_path)
        logger.info("Done")
    else:
        logger.info(
            f"Skipped quantizing {model_name} (Not found at {out_path})")


def _compile_coreml_model(source_model_path, output_dir, final_name):
    """ Compiles Core ML models using the coremlcompiler utility from Xcode toolchain
    """
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(
            f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(
        os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile {source_model_path} {output_dir}")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def _download_t5_model(args, t5_save_path):
    t5_url = args.text_encoder_t5_url
    match = re.match(r'https://huggingface.co/(.+)/resolve/main/(.+)', t5_url)
    if not match:
        raise ValueError(f"Invalid Hugging Face URL: {t5_url}")
    repo_id, model_subpath = match.groups()

    download_path = snapshot_download(
        repo_id=repo_id,
        revision="main",
        allow_patterns=[f"{model_subpath}/*"]
    )
    logger.info(f"Downloaded T5 model to {download_path}")

    # Move the downloaded model to the top level of the Resources directory
    logger.info(f"Copying T5 model from {download_path} to {t5_save_path}")
    cache_path = os.path.join(download_path, model_subpath)
    shutil.copytree(cache_path, t5_save_path)


def bundle_resources_for_swift_cli(args):
    """
    - Compiles Core ML models from mlpackage into mlmodelc format
    - Download tokenizer resources for the text encoder
    """
    resources_dir = os.path.join(args.o, "Resources")
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir, exist_ok=True)
        logger.info(f"Created {resources_dir} for Swift CLI assets")

    # Compile model using coremlcompiler (Significantly reduces the load time for unet)
    for source_name, target_name in [("text_encoder", "TextEncoder"),
                                     ("text_encoder_2", "TextEncoder2"),
                                     ("image_encoder", "ImageEncoder"),
                                     ("vae_decoder", "VAEDecoder"),
                                     ("vae_encoder", "VAEEncoder"),
                                     ("unet", "Unet"),
                                     ("unet_chunk1", "UnetChunk1"),
                                     ("unet_chunk2", "UnetChunk2"),
                                     ("unet_chunk3", "UnetChunk3"),
                                     ("unet_chunk4", "UnetChunk4"),
                                     ("refiner", "UnetRefiner"),
                                     ("refiner_chunk1", "UnetRefinerChunk1"),
                                     ("refiner_chunk2", "UnetRefinerChunk2"),
                                     ("refiner_chunk3", "UnetRefinerChunk3"),
                                     ("refiner_chunk4", "UnetRefinerChunk4"),
                                     ("mmdit", "MultiModalDiffusionTransformer"),
                                     ("control-unet", "ControlledUnet"),
                                     ("control-unet_chunk1", "ControlledUnetChunk1"),
                                     ("control-unet_chunk2", "ControlledUnetChunk2"),
                                     ("control-unet_chunk3", "ControlledUnetChunk3"),
                                     ("control-unet_chunk4", "ControlledUnetChunk4"),
                                     ("safety_checker", "SafetyChecker"),
                                     # Architectural chunks for SDXL
                                     ("SDXLAlphaEncoderA", "SDXLAlphaEncoderA"),
                                     ("SDXLAlphaEncoderB", "SDXLAlphaEncoderB"),
                                     ("SDXLGammaDownblock", "SDXLGammaDownblock"),
                                     ("SDXLSigmaCore", "SDXLSigmaCore"),
                                     ("SDXLThetaUpblockA", "SDXLThetaUpblockA"),
                                     ("SDXLThetaUpblockB", "SDXLThetaUpblockB"),
                                     ("SDXLLambdaUpblock", "SDXLLambdaUpblock"),
                                     ("SDXLKappaUpblock", "SDXLKappaUpblock"),
                                     ("SDXLOmegaDecoder", "SDXLOmegaDecoder")]:
        source_path = _get_out_path(args, source_name)
        if os.path.exists(source_path):
            target_path = _compile_coreml_model(source_path, resources_dir,
                                                target_name)
            logger.info(f"Compiled {source_path} to {target_path}")
        else:
            logger.warning(
                f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
            )

    if args.convert_controlnet:
        for controlnet_model_version in args.convert_controlnet:
            controlnet_model_name = controlnet_model_version.replace("/", "_")
            fname = f"ControlNet_{controlnet_model_name}.mlpackage"
            source_path = os.path.join(args.o, fname)
            controlnet_dir = os.path.join(resources_dir, "controlnet")
            target_name = "".join([word.title() for word in re.split('_|-', controlnet_model_name)])

            if os.path.exists(source_path):
                target_path = _compile_coreml_model(source_path, controlnet_dir,
                                                    target_name)
                logger.info(f"Compiled {source_path} to {target_path}")
            else:
                logger.warning(
                    f"{source_path} not found, skipping compilation to {target_name}.mlmodelc"
                )

    # Fetch and save vocabulary JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer vocab.json")
    with open(os.path.join(resources_dir, "vocab.json"), "wb") as f:
        f.write(requests.get(args.text_encoder_vocabulary_url).content)
    logger.info("Done")

    # Fetch and save merged pairs JSON file for text tokenizer
    logger.info("Downloading and saving tokenizer merges.txt")
    with open(os.path.join(resources_dir, "merges.txt"), "wb") as f:
        f.write(requests.get(args.text_encoder_merges_url).content)
    logger.info("Done")

    # Fetch and save pre-converted T5 text encoder model
    t5_model_name = "TextEncoderT5.mlmodelc"
    t5_save_path = os.path.join(resources_dir, t5_model_name)
    if args.include_t5:
        if not os.path.exists(t5_save_path):
            logger.info("Downloading pre-converted T5 encoder model TextEncoderT5.mlmodelc")
            _download_t5_model(args, t5_save_path)
            logger.info("Done")
        else:
            logger.info(f"Skipping T5 download as {t5_save_path} already exists")
            
        # Fetch and save T5 text tokenizer JSON files
        logger.info("Downloading and saving T5 tokenizer files tokenizer_config.json and tokenizer.json")
        with open(os.path.join(resources_dir, "tokenizer_config.json"), "wb") as f:
            f.write(requests.get(args.text_encoder_t5_config_url).content)
        with open(os.path.join(resources_dir, "tokenizer.json"), "wb") as f:
            f.write(requests.get(args.text_encoder_t5_data_url).content)
        logger.info("Done")

    return resources_dir


from transformers.models.clip import modeling_clip

# Copied from https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/clip/modeling_clip.py#L677C1-L692C1
# Starting from transformers >= 4.35.0, the _make_causal_mask function is replaced by _create_4d_causal_attention_mask in modeling_clip.
# For backward compatibility with versions < 4.35.0, both functions are patched here.
def patched_make_causal_mask(input_ids_shape, dtype, device, past_key_values_length: int = 0):
    """ Patch to replace torch.finfo(dtype).min with -1e4
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(-1e4, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
modeling_clip._make_causal_mask = patched_make_causal_mask # For transformers >= 4.30.0 and transformers < 4.35.0
modeling_clip._create_4d_causal_attention_mask = patched_make_causal_mask # For transformers >= 4.35.0

def _load_image_encoder(model_spec: str):
    """
    Load CLIP image encoder from a model with optional subfolder.
    model_spec: "model_name[:subfolder]"
    """
    if ":" in model_spec:
        model_name, subfolder = model_spec.split(":", 1)
    else:
        model_name, subfolder = model_spec, None

    model_name = model_name.strip()
    subfolder = subfolder.strip() if subfolder else None

    try:
        image_encoder = CLIPVisionModel.from_pretrained(
            model_name,
            subfolder=subfolder,
            torch_dtype=torch.float16
        )
    except Exception as e:
        logger.info(f"❌ Failed to load {model_name} (subfolder={subfolder}): {e}")
        return None

    logger.info(f"✅ Loaded image encoder '{model_name}' (subfolder={subfolder})")
    return image_encoder

def _load_ip_adapter(model_spec: str, pipe):
    """
    Load IP adapter from a model with optional subfolder.
    model_spec: "model_name[:subfolder]"
    """
    if ":" in model_spec:
        model_name, subfolder, weight_name = model_spec.split(":", 2)
    else:
        model_name, subfolder, weight_name = model_spec, None, None

    model_name = model_name.strip()
    subfolder = subfolder.strip() if subfolder else ""
    weight_name = weight_name.strip() if weight_name else None

    try:
        pipe.load_ip_adapter(model_name, subfolder=subfolder, weight_name=weight_name)
    except Exception as e:
        logger.info(f"❌ Failed to load {model_name} (subfolder={subfolder}) (weight_name={weight_name}): {e}")
        return None

    logger.info(f"✅ Loaded IP adapter '{model_name}' (subfolder={subfolder}) (weight_name={weight_name})")
    return


def _determine_image_encoder_output_mode(image_encoder):
    config = image_encoder.config

    # Default assumption
    mode = "hidden_states"

    # 1️⃣ Class check
    if isinstance(image_encoder, CLIPVisionModelWithProjection):
        has_projection = config.projection_dim is not None
        different_dims = (
            has_projection and config.projection_dim != config.hidden_size
        )

        # If projection exists and it's intended to be used (same as hidden_size)
        # → projection is internal, so use .image_embeds
        if has_projection and not different_dims:
            mode = "image_embeds"
        # Otherwise, external projection → hidden_states
        else:
            mode = "hidden_states"

    elif isinstance(image_encoder, CLIPVisionModel):
        # Pure vision encoder → definitely hidden_states
        mode = "hidden_states"

    return mode

def convert_image_encoder(image_encoder, submodule_name, args, do_classifier_free_guidance=True):
    """ Converts image encoder for Stable Diffusion
    """
    if image_encoder is None:
        logger.info(f"❌ Failed to convert image encoder, no image encoder defined.")
        return

    image_encoder = image_encoder.to(dtype=torch.float32)

    out_path = _get_out_path(args, submodule_name)

    if os.path.exists(out_path):
        logger.info(
            f"`ImageEncoder` already exists at {out_path}, skipping conversion."
        )
        return

    sample_shape = (
        1,     # B
        image_encoder.config.num_channels, # C
        image_encoder.config.image_size,   # H
        image_encoder.config.image_size,   # W
    )

    sample_image_encoder_inputs = {
        "input_image":
        torch.rand(*sample_shape, dtype=torch.float16)
    }
    sample_image_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_image_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_image_encoder_inputs_spec}")

    class ImageEncoder(nn.Module):

        def __init__(self, mode, do_cfg):
            super().__init__()
            self.image_encoder = image_encoder
            self.mode = mode
            self.do_cfg = do_cfg
            # CLIP normalization
            _means = [0.48145466, 0.4578275, 0.40821073]
            _stds = [0.26862954, 0.26130258, 0.27577711]

            self.stds = torch.tensor(_stds).to(torch.float32)[:,None,None]
            self.means = torch.tensor(_means).to(torch.float32)[:,None,None]

            self.transform_model = torch.nn.Sequential(
                transforms.Normalize(mean=_means,
                std=_stds)
            )

        def forward(self, input_image):
            input_image_normalized = self.transform_model(input_image)

            if self.mode == "hidden_states":
                image_embeds = self.image_encoder(input_image_normalized, output_hidden_states=True).hidden_states[-2]
            else:
                image_embeds = self.image_encoder(input_image_normalized, output_hidden_states=False).image_embeds

            if self.mode == "hidden_states":
                if self.do_cfg:
                    return torch.cat([image_embeds, image_embeds], dim=0)
                else:
                    return image_embeds
            else:
                image_embeds = image_embeds[None, :]
                if self.do_cfg:
                    return torch.cat([image_embeds, image_embeds], dim=0)
                else:
                    return image_embeds

    mode = _determine_image_encoder_output_mode(image_encoder)

    hidden_layer = None
    reference_image_encoder = ImageEncoder(mode, do_classifier_free_guidance).eval()

    logger.info(f"JIT tracing..")
    reference_image_encoder = torch.jit.trace(
        reference_image_encoder,
        (sample_image_encoder_inputs["input_image"].to(torch.float32), ),
    )
    logger.info("Done.")

    output_names = ["image_embeds"]
    coreml_image_encoder, out_path = _convert_to_coreml(
        f"imageencoder",
        reference_image_encoder, sample_image_encoder_inputs,
        output_names, args, out_path=out_path, precision=ct.precision.FLOAT16)

    # Set model metadata
    coreml_image_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_image_encoder.license = "MIT License"
    coreml_image_encoder.version = args.model_version
    coreml_image_encoder.short_description = \
        "The CLIP model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision tasks."

    # Set the input descriptions
    coreml_image_encoder.input_description["input_image"] = "The input image"

    # Set the output descriptions
    coreml_image_encoder.output_description["image_embeds"] = "Image Embeds"

    coreml_image_encoder.save(out_path)

    logger.info(f"Saved image_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        with torch.no_grad():
            baseline_out = reference_image_encoder(sample_image_encoder_inputs["input_image"].to(torch.float32))

        baseline_out = baseline_out.numpy()

        coreml_out = coreml_image_encoder.predict(
            {k: v.numpy() for k, v in sample_image_encoder_inputs.items()}
        )
        coreml_out = coreml_out["image_embeds"]
        report_correctness(
            baseline_out, coreml_out,
            "image_encoder baseline PyTorch to reference CoreML")

    del reference_image_encoder, coreml_image_encoder
    gc.collect()


def convert_text_encoder(text_encoder, tokenizer, submodule_name, args):
    """ Converts the text encoder component of Stable Diffusion
    """
    text_encoder = text_encoder.to(dtype=torch.float32)
    out_path = _get_out_path(args, submodule_name)
    if os.path.exists(out_path):
        logger.info(
            f"`{submodule_name}` already exists at {out_path}, skipping conversion."
        )
        return

    # Create sample inputs for tracing, conversion and correctness verification
    text_encoder_sequence_length = tokenizer.model_max_length

    sample_text_encoder_inputs = {
        "input_ids":
        torch.randint(
            text_encoder.config.vocab_size,
            (1, text_encoder_sequence_length),
            # https://github.com/apple/coremltools/issues/1423
            dtype=torch.float32,
        )
    }
    sample_text_encoder_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_text_encoder_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_text_encoder_inputs_spec}")

    class TextEncoder(nn.Module):

        def __init__(self, with_hidden_states_for_layer=None):
            super().__init__()
            self.text_encoder = text_encoder
            self.with_hidden_states_for_layer = with_hidden_states_for_layer

        def forward(self, input_ids):
            if self.with_hidden_states_for_layer is not None:
                output = self.text_encoder(input_ids, output_hidden_states=True)
                hidden_embeds = output.hidden_states[self.with_hidden_states_for_layer]
                if "text_embeds" in output:
                    return (hidden_embeds, output.text_embeds)
                else:
                    return (hidden_embeds, output.pooler_output)
            else:
                return self.text_encoder(input_ids, return_dict=False)

    # SD XL uses the hidden states after the encoder layers from both encoders,
    # and the pooled `text_embeds` output of the second encoder.
    hidden_layer = -2 if args.xl_version else None
    reference_text_encoder = TextEncoder(with_hidden_states_for_layer=hidden_layer).eval()

    logger.info(f"JIT tracing {submodule_name}..")
    reference_text_encoder = torch.jit.trace(
        reference_text_encoder,
        (sample_text_encoder_inputs["input_ids"].to(torch.int32), ),
    )
    logger.info("Done.")

    if args.xl_version:
        output_names = ["hidden_embeds", "pooled_outputs"]
    else:
        output_names = ["last_hidden_state", "pooled_outputs"]
    coreml_text_encoder, out_path = _convert_to_coreml(
        submodule_name, reference_text_encoder, sample_text_encoder_inputs,
        output_names, args)

    # Set model metadata
    coreml_text_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    if args.xl_version:
        coreml_text_encoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    else:
        coreml_text_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_text_encoder.version = args.model_version
    coreml_text_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_text_encoder.input_description[
        "input_ids"] = "The token ids that represent the input text"

    # Set the output descriptions
    if args.xl_version:
        coreml_text_encoder.output_description[
            "hidden_embeds"] = "Hidden states after the encoder layers"
    else:
        coreml_text_encoder.output_description[
            "last_hidden_state"] = "The token embeddings as encoded by the Transformer model"
    coreml_text_encoder.output_description[
        "pooled_outputs"] = "The version of the `last_hidden_state` output after pooling"

    coreml_text_encoder.save(out_path)

    logger.info(f"Saved text_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = text_encoder(
            sample_text_encoder_inputs["input_ids"].to(torch.int32),
            output_hidden_states=args.xl_version,
            return_dict=True,
        )
        if args.xl_version:
            # TODO: maybe check pooled_outputs too
            baseline_out = baseline_out.hidden_states[hidden_layer].numpy()
        else:
            baseline_out = baseline_out.last_hidden_state.numpy()

        coreml_out = coreml_text_encoder.predict(
            {k: v.numpy() for k, v in sample_text_encoder_inputs.items()}
        )
        coreml_out = coreml_out["hidden_embeds" if args.xl_version else "last_hidden_state"]
        report_correctness(
            baseline_out, coreml_out,
            "text_encoder baseline PyTorch to reference CoreML")

    del reference_text_encoder, coreml_text_encoder
    gc.collect()


def modify_coremltools_torch_frontend_badbmm():
    """
    Modifies coremltools torch frontend for baddbmm to be robust to the `beta` argument being of non-float dtype:
    e.g. https://github.com/huggingface/diffusers/blob/v0.8.1/src/diffusers/models/attention.py#L315
    """
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    if "baddbmm" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["baddbmm"]

    @register_torch_op
    def baddbmm(context, node):
        """
        baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
        output = beta * input + alpha * batch1 * batch2
        Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
        If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
        and out will be a (b×n×p) tensor.
        """
        assert len(node.outputs) == 1
        inputs = _get_inputs(context, node, expected=5)
        bias, batch1, batch2, beta, alpha = inputs

        if beta.val != 1.0:
            # Apply scaling factor beta to the bias.
            if beta.val.dtype == np.int32:
                beta = mb.cast(x=beta, dtype="fp32")
                logger.warning(
                    f"Casted the `beta`(value={beta.val}) argument of `baddbmm` op "
                    "from int32 to float32 dtype for conversion!")
            bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")

            context.add(bias)

        if alpha.val != 1.0:
            # Apply scaling factor alpha to the input.
            batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
            context.add(batch1)

        bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")
        context.add(bmm_node)

        baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
        context.add(baddbmm_node)


def convert_vae_decoder(pipe, args):
    """ Converts the VAE Decoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_decoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_decoder() before convert_unet()")

    z_shape = (
        1,  # B
        pipe.vae.config.latent_channels,  # C
        args.latent_h or pipe.unet.config.sample_size,  # H
        args.latent_w or pipe.unet.config.sample_size,  # W
    )

    if args.custom_vae_version is None and args.xl_version:
        inputs_dtype = torch.float32
        compute_precision = ct.precision.FLOAT32
        # FIXME: Hardcoding to CPU_AND_GPU since ANE doesn't support FLOAT32
        compute_unit = ct.ComputeUnit.CPU_AND_GPU
    else:
        inputs_dtype = torch.float16
        compute_precision = None
        compute_unit = None

    sample_vae_decoder_inputs = {
        "z": torch.rand(*z_shape, dtype=inputs_dtype)
    }

    class VAEDecoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.decode() method
        """

        def __init__(self):
            super().__init__()
            self.post_quant_conv = pipe.vae.post_quant_conv.to(dtype=torch.float32)
            self.decoder = pipe.vae.decoder.to(dtype=torch.float32)

        def forward(self, z):
            return self.decoder(self.post_quant_conv(z))

    baseline_decoder = VAEDecoder().eval()

    # No optimization needed for the VAE Decoder as it is a pure ConvNet
    traced_vae_decoder = torch.jit.trace(
        baseline_decoder, (sample_vae_decoder_inputs["z"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
    coreml_vae_decoder, out_path = _convert_to_coreml(
        "vae_decoder", traced_vae_decoder, sample_vae_decoder_inputs,
        ["image"], args, precision=compute_precision, compute_unit=compute_unit)

    # Set model metadata
    coreml_vae_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    if args.xl_version:
        coreml_vae_decoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    else:
        coreml_vae_decoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_decoder.version = args.model_version
    coreml_vae_decoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_decoder.input_description["z"] = \
        "The denoised latent embeddings from the unet model after the last step of reverse diffusion"

    # Set the output descriptions
    coreml_vae_decoder.output_description[
        "image"] = "Generated image normalized to range [-1, 1]"

    coreml_vae_decoder.save(out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_decoder(
            z=sample_vae_decoder_inputs["z"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vae_decoder.predict(
                {k: v.numpy()
                 for k, v in sample_vae_decoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vae_decoder baseline PyTorch to baseline CoreML")

    del traced_vae_decoder, pipe.vae.decoder, coreml_vae_decoder
    gc.collect()

def convert_vae_decoder_sd3(args):
    """ Converts the VAE component of Stable Diffusion 3
    """
    out_path = _get_out_path(args, "vae_decoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_decoder` already exists at {out_path}, skipping conversion."
        )
        return

    # Convert the VAE Decoder model via DiffusionKit
    converted_vae_path = convert_vae_to_mlpackage(
        model_version=args.model_version,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        output_dir=args.o,
    )

    # Load converted model
    coreml_vae_decoder = ct.models.MLModel(converted_vae_path)

    # Set model metadata
    coreml_vae_decoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_vae_decoder.license = "Stability AI Community License (https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md)"
    coreml_vae_decoder.version = args.model_version
    coreml_vae_decodershort_description = \
        "Stable Diffusion 3 generates images conditioned on text or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/pdf/2403.03206 for details."

    # Set the input descriptions
    coreml_vae_decoder.input_description["z"] = \
        "The denoised latent embeddings from the unet model after the last step of reverse diffusion"

    # Set the output descriptions
    coreml_vae_decoder.output_description[
        "image"] = "Generated image normalized to range [-1, 1]"

    # Set package version metadata
    from python_coreml_stable_diffusion._version import __version__
    coreml_vae_decoder.user_defined_metadata["com.github.apple.ml-stable-diffusion.version"] = __version__
    from diffusionkit.version import __version__
    coreml_vae_decoder.user_defined_metadata["com.github.argmax.diffusionkit.version"] = __version__

    # Save the updated model
    coreml_vae_decoder.save(out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Delete the original file
    if os.path.exists(converted_vae_path):
        shutil.rmtree(converted_vae_path)

    del coreml_vae_decoder
    gc.collect()


def convert_vae_encoder(pipe, args):
    """ Converts the VAE Encoder component of Stable Diffusion
    """
    out_path = _get_out_path(args, "vae_encoder")
    if os.path.exists(out_path):
        logger.info(
            f"`vae_encoder` already exists at {out_path}, skipping conversion."
        )
        return

    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_vae_encoder() before convert_unet()")
    
    height = (args.latent_h or pipe.unet.config.sample_size) * 8
    width = (args.latent_w or pipe.unet.config.sample_size) * 8
    
    x_shape = (
        1,  # B
        3,  # C (RGB range from -1 to 1)
        height,  # H
        width,  # w
    )

    if args.custom_vae_version is None and args.xl_version:
        inputs_dtype = torch.float32
        compute_precision = ct.precision.FLOAT32
        # FIXME: Hardcoding to CPU_AND_GPU since ANE doesn't support FLOAT32
        compute_unit = ct.ComputeUnit.CPU_AND_GPU
    else:
        inputs_dtype = torch.float16
        compute_precision = None
        compute_unit = None

    sample_vae_encoder_inputs = {
        "x": torch.rand(*x_shape, dtype=inputs_dtype)
    }

    class VAEEncoder(nn.Module):
        """ Wrapper nn.Module wrapper for pipe.encode() method
        """

        def __init__(self):
            super().__init__()
            self.quant_conv = pipe.vae.quant_conv.to(dtype=torch.float32)
            self.encoder = pipe.vae.encoder.to(dtype=torch.float32)

        def forward(self, x):
            return self.quant_conv(self.encoder(x))

    baseline_encoder = VAEEncoder().eval()

    # No optimization needed for the VAE Encoder as it is a pure ConvNet
    traced_vae_encoder = torch.jit.trace(
        baseline_encoder, (sample_vae_encoder_inputs["x"].to(torch.float32), ))

    modify_coremltools_torch_frontend_badbmm()
    coreml_vae_encoder, out_path = _convert_to_coreml(
        "vae_encoder", traced_vae_encoder, sample_vae_encoder_inputs,
        ["latent"], args, precision=compute_precision, compute_unit=compute_unit)

    # Set model metadata
    coreml_vae_encoder.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    if args.xl_version:
        coreml_vae_encoder.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    else:
        coreml_vae_encoder.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_vae_encoder.version = args.model_version
    coreml_vae_encoder.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_vae_encoder.input_description["x"] = \
        "The input image to base the initial latents on normalized to range [-1, 1]"

    # Set the output descriptions
    coreml_vae_encoder.output_description["latent"] = "The latent embeddings from the unet model from the input image."

    coreml_vae_encoder.save(out_path)

    logger.info(f"Saved vae_encoder into {out_path}")

    # Parity check PyTorch vs CoreML
    if args.check_output_correctness:
        baseline_out = baseline_encoder(
            x=sample_vae_encoder_inputs["x"].to(torch.float32)).numpy()
        coreml_out = list(
            coreml_vae_encoder.predict(
                {k: v.numpy()
                 for k, v in sample_vae_encoder_inputs.items()}).values())[0]
        report_correctness(baseline_out, coreml_out,
                           "vae_encoder baseline PyTorch to baseline CoreML")

    del traced_vae_encoder, pipe.vae.encoder, coreml_vae_encoder
    gc.collect()

def _attn_processors(unet_cls) -> Dict[str, nn.Module]:
    r"""
    Returns:
        `dict` of attention processors: A dictionary containing all attention processors used in the model with
        indexed by its weight name.
    """
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, nn.Module]):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = module.get_processor()

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in unet_cls.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors

def _set_ip_adapter_additional_scales(unet_cls, ip_scales):
    for attn_name, attn_processor in _attn_processors(unet_cls).items():
        if isinstance(
            attn_processor, (IPAdapterAttnProcessor2_0)
        ):
            for i, scale in enumerate(ip_scales):
                for k, s in scale.items():
                    if attn_name.startswith(k):
                        logger.info(f"Setting processor additional scale of {attn_name} to value {s}")
                        attn_processor.ip_adapter_scale_additional = torch.tensor([s])

def convert_unet(pipe, args, model_name=None):
    """ Converts the UNet component of Stable Diffusion
    """
    if args.unet_support_controlnet:
        unet_name = "control-unet"
    else:
        unet_name = model_name or "unet"

    out_path = _get_out_path(args, unet_name)

    # Check if Unet was previously exported and then chunked
    if args.unet_chunks is not None:
        unet_chunks_exist = all(
            os.path.exists(
                out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
            for idx in range(args.unet_chunks))

    if args.unet_chunks is not None and unet_chunks_exist:
        logger.info("`unet` chunks already exist, skipping conversion.")
        del pipe.unet
        gc.collect()
        return

    # If original Unet does not exist, export it from PyTorch+diffusers
    elif not os.path.exists(out_path):
        # Prepare sample input shapes and values
        batch_size = 2  # for classifier-free guidance
        if args.unet_batch_one:
            batch_size = 1  # for not using classifier-free guidance
        sample_shape = (
            batch_size,                    # B
            pipe.unet.config.in_channels,  # C
            args.latent_h or pipe.unet.config.sample_size,  # H
            args.latent_w or pipe.unet.config.sample_size,  # W
        )

        if not hasattr(pipe, "text_encoder"):
            raise RuntimeError(
                "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
                "Please use convert_unet() before convert_text_encoder()")
        
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            text_token_sequence_length = pipe.text_encoder.config.max_position_embeddings
            hidden_size = pipe.text_encoder.config.hidden_size,
        elif hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            text_token_sequence_length = pipe.text_encoder_2.config.max_position_embeddings
            hidden_size = pipe.text_encoder_2.config.hidden_size,

        encoder_hidden_states_shape = (
            batch_size,
            args.text_encoder_hidden_size or pipe.unet.config.cross_attention_dim or hidden_size,
            1,
            args.text_token_sequence_length or text_token_sequence_length,
        )

        # Create the scheduled timesteps for downstream use
        DEFAULT_NUM_INFERENCE_STEPS = 50
        pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)

        sample_unet_inputs = OrderedDict([
            ("sample", torch.rand(*sample_shape)),
            ("timestep",
             torch.tensor([pipe.scheduler.timesteps[0].item()] *
                          (batch_size)).to(torch.float32)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape))
        ])

        # Prepare inputs
        baseline_sample_unet_inputs = deepcopy(sample_unet_inputs)
        baseline_sample_unet_inputs[
            "encoder_hidden_states"] = baseline_sample_unet_inputs[
                "encoder_hidden_states"].squeeze(2).transpose(1, 2)

        # Initialize reference unet
        if args.xl_version:
            unet_cls = unet.UNet2DConditionModelXL

            # Sample time_ids
            height = (args.latent_h or pipe.unet.config.sample_size) * 8
            width = (args.latent_w or pipe.unet.config.sample_size) * 8

            original_size = (height, width) # output_resolution
            crops_coords_top_left = (0, 0) # topleft_crop_cond
            target_size = (height, width) # resolution_cond
            if hasattr(pipe.config, "requires_aesthetics_score") and pipe.config.requires_aesthetics_score:
                # Part of SDXL's micro-conditioning as explained in section 2.2 of
                # [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                # simulate an aesthetic score of the generated image by influencing the positive and negative text conditions.
                aesthetic_score = 6.0 # default aesthetic_score
                negative_aesthetic_score = 2.5 # default negative_aesthetic_score
                add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
                add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
            else:
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

            if args.unet_batch_one:
                time_ids = [
                    add_time_ids
                ]
            else: # for classifier-free guidance
                time_ids = [
                    add_neg_time_ids,
                    add_time_ids
                ]

            # Pooled text embedding from text_encoder_2
            text_embeds_shape = (
                batch_size,
                pipe.text_encoder_2.config.hidden_size
            )

            additional_xl_inputs = OrderedDict([
                ("time_ids", torch.tensor(time_ids).to(torch.float32)),
                ("text_embeds", torch.rand(*text_embeds_shape)),
            ])

            sample_unet_inputs.update(additional_xl_inputs)
            baseline_sample_unet_inputs['added_cond_kwargs'] = additional_xl_inputs

            if pipe.image_encoder is not None:
                image_enc_cfg = pipe.image_encoder.config
                mode = _determine_image_encoder_output_mode(pipe.image_encoder)

                if mode == "hidden_states":
                    H, W = image_enc_cfg.image_size, image_enc_cfg.image_size
                    P = image_enc_cfg.patch_size
                    D = image_enc_cfg.hidden_size
                    num_patches = (H // P) * (W // P)
                    image_embeds_shape = (batch_size, num_patches + 1, D)
                else:
                    image_embeds_shape = (batch_size, 1, image_enc_cfg.projection_dim)

                additional_xl_inputs["image_embeds"] = torch.rand(*image_embeds_shape)
                additional_xl_inputs["ip_adapter_scale"] = torch.tensor([1.0])
                sample_unet_inputs.update(additional_xl_inputs)
                baseline_sample_unet_inputs['added_cond_kwargs'] = additional_xl_inputs
            else:
                sample_unet_inputs.update(additional_xl_inputs)
        else:
            unet_cls = unet.UNet2DConditionModel

        reference_unet = unet_cls(support_controlnet=args.unet_support_controlnet, support_image_prompt=(pipe.image_encoder is not None), encoder_hid_proj_reference=pipe.unet.encoder_hid_proj, **pipe.unet.config).eval()

        # Hook in additional scales for the IP-Adapter
        if args.load_ip_adapter:
            try:
                ip_scales = json.loads(args.ip_scales)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON passed to --ip_scales")

            # Normalize: always a list of dicts
            if isinstance(ip_scales, dict):
                ip_scales = [ip_scales]

            _set_ip_adapter_additional_scales(reference_unet, ip_scales)

        load_state_dict_summary = reference_unet.load_state_dict(
            pipe.unet.state_dict())

        if args.unet_support_controlnet:
            from .unet import calculate_conv2d_output_shape
            additional_residuals_shapes = []

            # conv_in
            out_h, out_w = calculate_conv2d_output_shape(
                (args.latent_h or pipe.unet.config.sample_size),
                (args.latent_w or pipe.unet.config.sample_size),
                reference_unet.conv_in,
            )
            additional_residuals_shapes.append(
                (batch_size, reference_unet.conv_in.out_channels, out_h, out_w))
            
            # down_blocks
            for down_block in reference_unet.down_blocks:
                additional_residuals_shapes += [
                    (batch_size, resnet.out_channels, out_h, out_w) for resnet in down_block.resnets
                ]
                if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
                    for downsampler in down_block.downsamplers:
                        out_h, out_w = calculate_conv2d_output_shape(out_h, out_w, downsampler.conv)
                    additional_residuals_shapes.append(
                        (batch_size, down_block.downsamplers[-1].conv.out_channels, out_h, out_w))
            
            # mid_block
            additional_residuals_shapes.append(
                (batch_size, reference_unet.mid_block.resnets[-1].out_channels, out_h, out_w)
            )

            baseline_sample_unet_inputs["down_block_additional_residuals"] = ()
            for i, shape in enumerate(additional_residuals_shapes):
                sample_residual_input = torch.rand(*shape)
                sample_unet_inputs[f"additional_residual_{i}"] = sample_residual_input
                if i == len(additional_residuals_shapes) - 1:
                    baseline_sample_unet_inputs["mid_block_additional_residual"] = sample_residual_input
                else:
                    baseline_sample_unet_inputs["down_block_additional_residuals"] += (sample_residual_input, )

        sample_unet_inputs_spec = {
            k: (v.shape, v.dtype)
            for k, v in sample_unet_inputs.items()
        }
        logger.info(f"Sample UNet inputs spec: {sample_unet_inputs_spec}")

        # JIT trace
        logger.info("JIT tracing..")
        if pipe.image_encoder is not None:
            reference_unet = torch.jit.trace(reference_unet, list(sample_unet_inputs.values()))
        else:
            reference_unet = UNet2DConditionModelXLWithoutIPAdapter(reference_unet)
            reference_unet = torch.jit.trace(reference_unet, list(sample_unet_inputs.values()))
        logger.info("Done.")

        if args.check_output_correctness:
            baseline_out = pipe.unet.to(torch.float32)(**baseline_sample_unet_inputs,
                                     return_dict=False)[0].numpy()
            reference_out = reference_unet(*sample_unet_inputs.values())[0].numpy()
            report_correctness(baseline_out, reference_out,
                               "unet baseline to reference PyTorch")

        del pipe.unet
        gc.collect()

        coreml_sample_unet_inputs = {
            k: v.numpy().astype(np.float16)
            for k, v in sample_unet_inputs.items()
        }

        coreml_unet, out_path = _convert_to_coreml(unet_name, reference_unet,
                                                   coreml_sample_unet_inputs,
                                                   ["noise_pred"], args)
        del reference_unet
        gc.collect()

        # Make additional residuals from ControlNet optional (if defined)
        if args.unet_support_controlnet:
            spec = coreml_unet.get_spec()

            # Loop over all inputs in the model description
            for input_type in spec.description.input:
                if input_type.name.startswith("additional_residual"):
                    input_type.type.isOptional = True

            # Update model
            coreml_unet = ct.models.MLModel(spec, weights_dir=coreml_unet.weights_dir)

        # Make image prompt input optional (if defined)
        if args.xl_version and pipe.image_encoder is not None:
            spec = coreml_unet.get_spec()

            # Loop over all inputs in the model description
            for input_type in spec.description.input:
                if input_type.name == "image_embeds":
                    input_type.type.isOptional = True

            # Update model
            coreml_unet = ct.models.MLModel(spec, weights_dir=coreml_unet.weights_dir)

        # Set model metadata
        coreml_unet.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
        if args.xl_version:
            coreml_unet.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
        else:
            coreml_unet.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
        coreml_unet.version = args.model_version if model_name != "refiner" or not hasattr(args, "refiner_version") else args.refiner_version
        coreml_unet.short_description = \
            "Stable Diffusion generates images conditioned on text or other images as input through the diffusion process. " \
            "Please refer to https://arxiv.org/abs/2112.10752 for details."

        # Set the input descriptions
        coreml_unet.input_description["sample"] = \
            "The low resolution latent feature maps being denoised through reverse diffusion"
        coreml_unet.input_description["timestep"] = \
            "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
        coreml_unet.input_description["encoder_hidden_states"] = \
            "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
            "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
            "Shorter text does not reduce computation."
        if args.xl_version:
            coreml_unet.input_description["time_ids"] = \
                "Additional embeddings that if specified are added to the embeddings that are passed along to the UNet blocks."
            coreml_unet.input_description["text_embeds"] = \
                "Additional embeddings from text_encoder_2 that if specified are added to the embeddings that are passed along to the UNet blocks."

            if pipe.image_encoder is not None:
                coreml_unet.input_description["image_embeds"] = \
                "Additional embeddings from image_encoder that if specified are added to the embeddings that are passed along to the UNet blocks."
                coreml_unet.input_description["ip_adapter_scale"] = \
                "Weight of the additional embeddings from image_encoder when used in attention processors."

        # Set the output descriptions
        coreml_unet.output_description["noise_pred"] = \
            "Same shape and dtype as the `sample` input. " \
            "The predicted noise to facilitate the reverse diffusion (denoising) process"

        # Set package version metadata
        from python_coreml_stable_diffusion._version import __version__
        coreml_unet.user_defined_metadata["com.github.apple.ml-stable-diffusion.version"] = __version__

        coreml_unet.save(out_path)
        logger.info(f"Saved unet into {out_path}")

        # Parity check PyTorch vs CoreML
        if args.check_output_correctness:
            coreml_out = list(
                coreml_unet.predict(coreml_sample_unet_inputs).values())[0]
            report_correctness(baseline_out, coreml_out,
                               "unet baseline PyTorch to reference CoreML")

        del coreml_unet
        gc.collect()
    else:
        del pipe.unet
        gc.collect()
        logger.info(
            f"`unet` already exists at {out_path}, skipping conversion.")

def fix_optional_inputs(mlpackage_path, optional_input_names):
    """
    Re-mark specific inputs as optional in a CoreML mlpackage.
    Inputs are marked optional if their name starts with any string in optional_input_names.
    """
    model = ct.models.MLModel(mlpackage_path)
    spec = model.get_spec()

    changed = False
    for input_type in spec.description.input:
        if input_type.name.startswith(tuple(optional_input_names)):
            input_type.type.isOptional = True
            changed = True

    if changed:
        ct.models.MLModel(spec, weights_dir=model.weights_dir).save(mlpackage_path)
        logger.info(f"Patched optional inputs in {mlpackage_path}")
    else:
        logger.info(f"No optional inputs to patch in {mlpackage_path}")


def convert_unet_architectural_chunks(pipe, args):
    """
    Convert UNet into architectural chunks.

    This creates semantically meaningful chunks that split the UNet along
    architectural boundaries rather than by weight size.
    """
    from python_coreml_stable_diffusion.unet_architectural_chunks import (
        ARCHITECTURAL_CHUNK_NAMES,
        ARCHITECTURAL_CHUNK_CLASSES,
        get_architectural_chunk_output_names,
    )

    # Check if all chunks already exist
    all_exist = all(
        os.path.exists(_get_out_path(args, chunk_name))
        for chunk_name in ARCHITECTURAL_CHUNK_NAMES
    )
    if all_exist:
        logger.info("All architectural UNet chunks already exist, skipping conversion.")
        return

    # Build reference UNet
    logger.info("Building reference UNet for architectural chunking...")

    batch_size = 1 if args.unet_batch_one else 2
    latent_h = args.latent_h or pipe.unet.config.sample_size
    latent_w = args.latent_w or pipe.unet.config.sample_size

    reference_unet = unet.UNet2DConditionModelXL(
        support_controlnet=args.unet_support_controlnet,
        support_image_prompt=(pipe.image_encoder is not None),
        encoder_hid_proj_reference=pipe.unet.encoder_hid_proj,
        **pipe.unet.config
    ).eval()

    # Hook in additional scales for the IP-Adapter
    if args.load_ip_adapter:
        try:
            ip_scales = json.loads(args.ip_scales)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON passed to --ip_scales")

        if isinstance(ip_scales, dict):
            ip_scales = [ip_scales]

        _set_ip_adapter_additional_scales(reference_unet, ip_scales)

    load_state_dict_summary = reference_unet.load_state_dict(pipe.unet.state_dict())
    logger.info(f"Loaded UNet state dict: {load_state_dict_summary}")

    # Common sample inputs
    height = latent_h * 8
    width = latent_w * 8

    # Text encoder hidden size
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        text_hidden_size = args.text_encoder_hidden_size or pipe.unet.config.cross_attention_dim or pipe.text_encoder_2.config.hidden_size
        text_token_length = args.text_token_sequence_length or pipe.text_encoder_2.config.max_position_embeddings
        pooled_text_dim = pipe.text_encoder_2.config.hidden_size
    else:
        text_hidden_size = args.text_encoder_hidden_size or pipe.unet.config.cross_attention_dim or 2048
        text_token_length = args.text_token_sequence_length or 77
        pooled_text_dim = 1280

    # Generate sample inputs for each chunk and convert
    for chunk_name in ARCHITECTURAL_CHUNK_NAMES:
        out_path = _get_out_path(args, chunk_name)

        if os.path.exists(out_path):
            logger.info(f"Skipping {chunk_name}, already exists at {out_path}")
            continue

        logger.info(f"Converting {chunk_name}...")

        # Create chunk wrapper
        chunk_class = ARCHITECTURAL_CHUNK_CLASSES[chunk_name]
        chunk_module = chunk_class(reference_unet).eval()

        # Generate sample inputs based on chunk type
        sample_inputs = _get_architectural_chunk_sample_inputs(
            chunk_name, batch_size, latent_h, latent_w,
            text_hidden_size, text_token_length, pooled_text_dim,
            pipe.image_encoder, args
        )

        sample_inputs_spec = {k: (v.shape, v.dtype) for k, v in sample_inputs.items()}
        logger.info(f"Sample inputs for {chunk_name}: {sample_inputs_spec}")

        # JIT trace
        logger.info(f"JIT tracing {chunk_name}...")
        traced_chunk = torch.jit.trace(chunk_module, list(sample_inputs.values()))
        logger.info("Done.")

        # Convert to CoreML
        coreml_sample_inputs = {
            k: v.numpy().astype(np.float16) if v.dtype == torch.float32 else v.numpy()
            for k, v in sample_inputs.items()
        }

        output_names = get_architectural_chunk_output_names(chunk_name, reference_unet)

        coreml_chunk, _ = _convert_to_coreml(
            chunk_name,
            traced_chunk,
            coreml_sample_inputs,
            output_names,
            args,
            out_path=out_path,
        )

        # Make ControlNet additional_residual inputs optional (if defined)
        if args.unet_support_controlnet:
            spec = coreml_chunk.get_spec()
            for input_type in spec.description.input:
                if input_type.name.startswith("additional_residual"):
                    input_type.type.isOptional = True
            coreml_chunk = ct.models.MLModel(spec, weights_dir=coreml_chunk.weights_dir)

        # Set metadata
        coreml_chunk.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
        coreml_chunk.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
        coreml_chunk.version = args.model_version
        coreml_chunk.short_description = f"SDXL UNet {chunk_name} - Architectural chunk for on-device inference"

        # Set package version metadata
        from python_coreml_stable_diffusion._version import __version__
        coreml_chunk.user_defined_metadata["com.github.apple.ml-stable-diffusion.version"] = __version__
        coreml_chunk.user_defined_metadata["chunk_type"] = "architectural"
        coreml_chunk.user_defined_metadata["chunk_name"] = chunk_name

        coreml_chunk.save(out_path)
        logger.info(f"Saved {chunk_name} to {out_path}")

        del traced_chunk, coreml_chunk, chunk_module
        gc.collect()

    logger.info("Architectural UNet chunking complete.")


def _get_architectural_chunk_sample_inputs(
    chunk_name, batch_size, latent_h, latent_w,
    text_hidden_size, text_token_length, pooled_text_dim,
    image_encoder, args
):
    """Generate sample inputs for each architectural chunk."""

    # Common tensor shapes
    # For SDXL, encoder_hidden_states is [B, hidden, 1, seq_len] due to conv format
    encoder_hidden_states_shape = (batch_size, text_hidden_size, 1, text_token_length)
    # emb is the time embedding, output from AlphaEncoder as [B, 1280, 1, 1] for conv compatibility
    emb_shape = (batch_size, 1280, 1, 1)
    # ip_hidden_states shape: [B, cross_attention_dim, 1, num_ip_tokens]
    # For IP-Adapter Plus, num_ip_tokens is typically 16
    ip_hidden_states_shape = (batch_size, text_hidden_size, 1, 16) if image_encoder is not None else None

    # Spatial dimensions at different resolutions
    h, w = latent_h, latent_w  # Full resolution (e.g., 96)
    h2, w2 = latent_h // 2, latent_w // 2  # Half (e.g., 48)
    h4, w4 = latent_h // 4, latent_w // 4  # Quarter (e.g., 24)
    h8, w8 = latent_h // 8, latent_w // 8  # Eighth (e.g., 12)

    # ControlNet support flag
    support_controlnet = getattr(args, 'unet_support_controlnet', False)

    if chunk_name == "SDXLAlphaEncoderA":
        # Time embedding computation only - minimal inputs
        inputs = OrderedDict([
            ("timestep", torch.tensor([1000.0] * batch_size).float()),
            ("time_ids", torch.rand(batch_size, 6)),
            ("text_embeds", torch.rand(batch_size, pooled_text_dim)),
        ])
        return inputs

    elif chunk_name == "SDXLAlphaEncoderB":
        # IP-Adapter + conv_in + down_blocks[0,1]
        inputs = OrderedDict([
            ("sample", torch.rand(batch_size, 4, h, w)),
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ])
        # Add IP-Adapter inputs if enabled
        if image_encoder is not None:
            image_enc_cfg = image_encoder.config
            mode = _determine_image_encoder_output_mode(image_encoder)
            if mode == "hidden_states":
                H, W = image_enc_cfg.image_size, image_enc_cfg.image_size
                P = image_enc_cfg.patch_size
                D = image_enc_cfg.hidden_size
                num_patches = (H // P) * (W // P)
                image_embeds_shape = (batch_size, num_patches + 1, D)
            else:
                image_embeds_shape = (batch_size, 1, image_enc_cfg.projection_dim)
            inputs["image_embeds"] = torch.rand(*image_embeds_shape)
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        # Add ControlNet residual inputs if enabled
        # AlphaEncoderB handles: conv_in (0) + down_blocks[0] (1,2,3) + down_blocks[1] (4,5,6)
        if support_controlnet:
            inputs["additional_residual_0"] = torch.rand(batch_size, 320, h, w)    # conv_in
            inputs["additional_residual_1"] = torch.rand(batch_size, 320, h, w)    # down_blocks[0].resnets[0]
            inputs["additional_residual_2"] = torch.rand(batch_size, 320, h, w)    # down_blocks[0].resnets[1]
            inputs["additional_residual_3"] = torch.rand(batch_size, 320, h2, w2)  # down_blocks[0].downsampler
            inputs["additional_residual_4"] = torch.rand(batch_size, 640, h2, w2)  # down_blocks[1].resnets[0]
            inputs["additional_residual_5"] = torch.rand(batch_size, 640, h2, w2)  # down_blocks[1].resnets[1]
            inputs["additional_residual_6"] = torch.rand(batch_size, 640, h4, w4)  # down_blocks[1].downsampler
        return inputs

    elif chunk_name == "SDXLGammaDownblock":
        inputs = OrderedDict([
            ("hidden", torch.rand(batch_size, 640, h4, w4)),
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ])
        if image_encoder is not None:
            inputs["ip_hidden_states"] = torch.rand(*ip_hidden_states_shape)
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        # Add ControlNet residual inputs if enabled
        # GammaDownblock handles: down_blocks[2] (7,8) - no downsampler
        if support_controlnet:
            inputs["additional_residual_7"] = torch.rand(batch_size, 1280, h4, w4)  # down_blocks[2].resnets[0]
            inputs["additional_residual_8"] = torch.rand(batch_size, 1280, h4, w4)  # down_blocks[2].resnets[1]
        return inputs

    elif chunk_name == "SDXLSigmaCore":
        # Note: down_blocks[2] has NO downsampler in SDXL, so spatial dims stay at h4, w4
        inputs = OrderedDict([
            ("hidden", torch.rand(batch_size, 1280, h4, w4)),  # from GammaDownblock (no downsampler)
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ])
        if image_encoder is not None:
            inputs["ip_hidden_states"] = torch.rand(*ip_hidden_states_shape)
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        # Add ControlNet residual input if enabled
        # SigmaCore handles: mid_block (9) - same spatial dims as hidden
        if support_controlnet:
            inputs["additional_residual_9"] = torch.rand(batch_size, 1280, h4, w4)  # mid_block
        return inputs

    elif chunk_name == "SDXLThetaUpblockA":
        # First layer of up_blocks[0]: ResNet0 + Attention0
        # Takes hidden from SigmaCore and skip_0 from down_blocks[2] resnet 1 (skip_down2_1)
        # Skip connections are consumed in REVERSE order within each up_block
        inputs = OrderedDict([
            ("hidden", torch.rand(batch_size, 1280, h4, w4)),  # from SigmaCore (mid_block)
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
            ("ip_hidden_states", torch.rand(*ip_hidden_states_shape) if image_encoder is not None else torch.zeros(batch_size, text_hidden_size, 1, 16)),
            ("skip_0", torch.rand(batch_size, 1280, h4, w4)),  # from down_blocks[2] resnet 1 (skip_down2_1)
        ])
        if image_encoder is not None:
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        return inputs

    elif chunk_name == "SDXLThetaUpblockB":
        # Remaining layers of up_blocks[0]: ResNet1 + Attention1 + ResNet2 + Attention2 + Upsampler
        # Takes hidden from ThetaUpblockA, skip_0 from down_blocks[2] resnet 0, skip_1 from down_blocks[1] downsampler
        inputs = OrderedDict([
            ("hidden", torch.rand(batch_size, 1280, h4, w4)),  # from ThetaUpblockA
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
            ("ip_hidden_states", torch.rand(*ip_hidden_states_shape) if image_encoder is not None else torch.zeros(batch_size, text_hidden_size, 1, 16)),
            ("skip_0", torch.rand(batch_size, 1280, h4, w4)),  # from down_blocks[2] resnet 0 (skip_down2_0)
            ("skip_1", torch.rand(batch_size, 640, h4, w4)),   # from down_blocks[1] downsampler (skip_down1_2)
        ])
        if image_encoder is not None:
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        return inputs

    elif chunk_name == "SDXLLambdaUpblock":
        # CrossAttnUpBlock2D (up_blocks[1]) - needs encoder_hidden_states
        # Consumes skips: skip[5]=[640, H/2, W/2], skip[4]=[640, H/2, W/2], skip[3]=[320, H/2, W/2]
        inputs = OrderedDict([
            ("hidden", torch.rand(batch_size, 1280, h2, w2)),  # from ThetaUpblockB (upsampled)
            ("emb", torch.rand(*emb_shape)),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
            ("ip_hidden_states", torch.rand(*ip_hidden_states_shape) if image_encoder is not None else torch.zeros(batch_size, text_hidden_size, 1, 16)),
            ("skip_0", torch.rand(batch_size, 640, h2, w2)),   # skip[5] from down_blocks[1] resnet 1
            ("skip_1", torch.rand(batch_size, 640, h2, w2)),   # skip[4] from down_blocks[1] resnet 0
            ("skip_2", torch.rand(batch_size, 320, h2, w2)),   # skip[3] from down_blocks[0] downsampler
        ])
        if image_encoder is not None:
            inputs["ip_adapter_scale"] = torch.tensor([1.0])
        return inputs

    elif chunk_name == "SDXLKappaUpblock":
        # UpBlock2D (up_blocks[2]) - no encoder_hidden_states needed
        # Consumes skips: skip[2]=[320, H, W], skip[1]=[320, H, W], skip[0]=[320, H, W]
        return OrderedDict([
            ("hidden", torch.rand(batch_size, 640, h, w)),     # from LambdaUpblock (upsampled)
            ("emb", torch.rand(*emb_shape)),
            ("skip_0", torch.rand(batch_size, 320, h, w)),     # skip[2] from down_blocks[0] resnet 1
            ("skip_1", torch.rand(batch_size, 320, h, w)),     # skip[1] from down_blocks[0] resnet 0
            ("skip_2", torch.rand(batch_size, 320, h, w)),     # skip[0] from conv_in
        ])

    elif chunk_name == "SDXLOmegaDecoder":
        # Only conv_norm_out + conv_act + conv_out
        return OrderedDict([
            ("hidden", torch.rand(batch_size, 320, h, w)),
            ("skip_conv_in", torch.rand(batch_size, 320, h, w)),
        ])

    else:
        raise ValueError(f"Unknown chunk name: {chunk_name}")


def chunk_unet(args, model_name=None):
    """ Chunks the UNet component of Stable Diffusion
    """
    if args.unet_support_controlnet:
        unet_name = "control-unet"
    else:
        unet_name = model_name or "unet"

    out_path = _get_out_path(args, unet_name)

    # Check if Unet was previously exported and then chunked
    if args.unet_chunks is not None:
        unet_chunks_exist = all(
            os.path.exists(
                out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
            for idx in range(args.unet_chunks))

    if args.unet_chunks is not None and unet_chunks_exist:
        return

    if args.unet_chunks is not None and not unet_chunks_exist:
        logger.info(f"Chunking {unet_name} in {args.unet_chunks} approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        args.merge_chunks_in_pipeline_model = False
        chunk_mlprogram.main(args)

        if args.unet_chunks == 4:
            chunk_dir = os.path.dirname(out_path)
            base = os.path.basename(out_path).replace(".mlpackage", "")

            first_chunks = [
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.startswith(base + "_chunk") and f.endswith(".mlpackage")
            ]

            for c in first_chunks:
                logger.info(f"Chunking sub-chunk: {c}")
                args.mlpackage_path = c
                chunk_mlprogram.main(args)

            # Delete first-pass chunks
            for f in [f"{base}_chunk1.mlpackage", f"{base}_chunk2.mlpackage"]:
                path = os.path.join(chunk_dir, f)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(f"Deleted first-pass chunk: {path}")

            # Rename second-level chunks
            rename_map = {
                f"{base}_chunk1_chunk1.mlpackage": f"{base}_chunk1.mlpackage",
                f"{base}_chunk1_chunk2.mlpackage": f"{base}_chunk2.mlpackage",
                f"{base}_chunk2_chunk1.mlpackage": f"{base}_chunk3.mlpackage",
                f"{base}_chunk2_chunk2.mlpackage": f"{base}_chunk4.mlpackage",
            }

            for src_name, dst_name in rename_map.items():
                src_path = os.path.join(chunk_dir, src_name)
                dst_path = os.path.join(chunk_dir, dst_name)
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                os.rename(src_path, dst_path)
                logger.info(f"Renamed {src_path} -> {dst_path}")
                # Fix: After chunking, the optional input flags are lost. So let's make the image prompt input optional again (if defined)
                fix_optional_inputs(
                    dst_path,
                    optional_input_names={"image_embeds", "additional_residual"},
                )

            # Delete original unet
            if os.path.exists(out_path):
                shutil.rmtree(out_path)

            logger.info(f"Finished chunking unet")

def convert_mmdit(args):
    """ Converts the MMDiT component of Stable Diffusion 3
    """
    out_path = _get_out_path(args, "mmdit")
    if os.path.exists(out_path):
        logger.info(
            f"`mmdit` already exists at {out_path}, skipping conversion."
        )
        return

    # Convert the MMDiT model via DiffusionKit
    converted_mmdit_path = convert_mmdit_to_mlpackage(
        model_version=args.model_version,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        output_dir=args.o,
        # FIXME: Hardcoding to CPU_AND_GPU since ANE doesn't support FLOAT32
        compute_precision=ct.precision.FLOAT32,
        compute_unit=ct.ComputeUnit.CPU_AND_GPU,
    )

    # Load converted model
    coreml_mmdit = ct.models.MLModel(converted_mmdit_path)

    # Set model metadata
    coreml_mmdit.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_mmdit.license = "Stability AI Community License (https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md)"
    coreml_mmdit.version = args.model_version
    coreml_mmdit.short_description = \
    "Stable Diffusion 3 generates images conditioned on text or other images as input through the diffusion process. " \
    "Please refer to https://arxiv.org/pdf/2403.03206 for details."

    # Set the input descriptions
    coreml_mmdit.input_description["latent_image_embeddings"] = \
        "The low resolution latent feature maps being denoised through reverse diffusion"
    coreml_mmdit.input_description["token_level_text_embeddings"] = \
        "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
        "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. "
    coreml_mmdit.input_description["pooled_text_embeddings"] = \
        "Additional embeddings that if specified are added to the embeddings that are passed along to the MMDiT model."
    coreml_mmdit.input_description["timestep"] = \
        "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
    
    # Set the output descriptions
    coreml_mmdit.output_description["denoiser_output"] = \
        "Same shape and dtype as the `latent_image_embeddings` input. " \
        "The predicted noise to facilitate the reverse diffusion (denoising) process"

    # Set package version metadata
    from python_coreml_stable_diffusion._version import __version__
    coreml_mmdit.user_defined_metadata["com.github.apple.ml-stable-diffusion.version"] = __version__
    from diffusionkit.version import __version__
    coreml_mmdit.user_defined_metadata["com.github.argmax.diffusionkit.version"] = __version__

    # Save the updated model
    coreml_mmdit.save(out_path)

    logger.info(f"Saved vae_decoder into {out_path}")

    # Delete the original file
    if os.path.exists(converted_mmdit_path):
        shutil.rmtree(converted_mmdit_path)

    del coreml_mmdit
    gc.collect()

def convert_safety_checker(pipe, args):
    """ Converts the Safety Checker component of Stable Diffusion
    """
    if pipe.safety_checker is None:
        logger.warning(
            f"diffusers pipeline for {args.model_version} does not have a `safety_checker` module! " \
            "`--convert-safety-checker` will be ignored."
        )
        return

    out_path = _get_out_path(args, "safety_checker")
    if os.path.exists(out_path):
        logger.info(
            f"`safety_checker` already exists at {out_path}, skipping conversion."
        )
        return

    pipe.safety_checker = pipe.safety_checker.to(torch.float32)
    
    im_h = pipe.vae.config.sample_size
    im_w = pipe.vae.config.sample_size

    if args.latent_h is not None:
        im_h = args.latent_h * 8

    if args.latent_w is not None:
        im_w = args.latent_w * 8

    sample_image = np.random.randn(
        1,     # B
        im_h,  # H
        im_w,  # w
        3      # C
    ).astype(np.float32)

    # Note that pipe.feature_extractor is not an ML model. It simply
    # preprocesses data for the pipe.safety_checker module.
    safety_checker_input = pipe.feature_extractor(
        pipe.numpy_to_pil(sample_image),
        return_tensors="pt",
    ).pixel_values.to(torch.float32)

    sample_safety_checker_inputs = OrderedDict([
        ("clip_input", safety_checker_input),
        ("images", torch.from_numpy(sample_image)),
        ("adjustment", torch.tensor([0]).to(torch.float32)),
    ])

    sample_safety_checker_inputs_spec = {
        k: (v.shape, v.dtype)
        for k, v in sample_safety_checker_inputs.items()
    }
    logger.info(f"Sample inputs spec: {sample_safety_checker_inputs_spec}")

    # Patch safety_checker's forward pass to be vectorized and avoid conditional blocks
    # (similar to pipe.safety_checker.forward_onnx)
    from diffusers.pipelines.stable_diffusion import safety_checker

    def forward_coreml(self, clip_input, images, adjustment):
        """ Forward pass implementation for safety_checker
        """

        def cosine_distance(image_embeds, text_embeds):
            return F.normalize(image_embeds) @ F.normalize(
                text_embeds).transpose(0, 1)

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = special_scores.gt(0).float().sum(dim=1).gt(0).float()
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = concept_scores.gt(0).float().sum(dim=1).gt(0)[:,
                                                                          None,
                                                                          None,
                                                                          None]

        has_nsfw_concepts_inds, _ = torch.broadcast_tensors(
            has_nsfw_concepts, images)
        images[has_nsfw_concepts_inds] = 0.0  # black image

        return images, has_nsfw_concepts.float(), concept_scores

    baseline_safety_checker = deepcopy(pipe.safety_checker.eval())
    setattr(baseline_safety_checker, "forward",
            MethodType(forward_coreml, baseline_safety_checker))

    # In order to parity check the actual signal, we need to override the forward pass to return `concept_scores` which is the
    # output before thresholding
    # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L100
    def forward_extended_return(self, clip_input, images, adjustment):

        def cosine_distance(image_embeds, text_embeds):
            normalized_image_embeds = F.normalize(image_embeds)
            normalized_text_embeds = F.normalize(text_embeds)
            return torch.mm(normalized_image_embeds,
                            normalized_text_embeds.t())

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds,
                                           self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(
            -1, cos_dist.shape[1])

        concept_scores = (cos_dist -
                          self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0

        return images, has_nsfw_concepts, concept_scores

    setattr(pipe.safety_checker, "forward",
            MethodType(forward_extended_return, pipe.safety_checker))

    # Trace the safety_checker model
    logger.info("JIT tracing..")
    traced_safety_checker = torch.jit.trace(
        baseline_safety_checker, list(sample_safety_checker_inputs.values()))
    logger.info("Done.")
    del baseline_safety_checker
    gc.collect()

    # Cast all inputs to float16
    coreml_sample_safety_checker_inputs = {
        k: v.numpy().astype(np.float16)
        for k, v in sample_safety_checker_inputs.items()
    }

    # Convert safety_checker model to Core ML
    coreml_safety_checker, out_path = _convert_to_coreml(
        "safety_checker", traced_safety_checker,
        coreml_sample_safety_checker_inputs,
        ["filtered_images", "has_nsfw_concepts", "concept_scores"], args)

    # Set model metadata
    coreml_safety_checker.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_safety_checker.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
    coreml_safety_checker.version = args.model_version
    coreml_safety_checker.short_description = \
        "Stable Diffusion generates images conditioned on text and/or other images as input through the diffusion process. " \
        "Please refer to https://arxiv.org/abs/2112.10752 for details."

    # Set the input descriptions
    coreml_safety_checker.input_description["clip_input"] = \
        "The normalized image input tensor resized to (224x224) in channels-first (BCHW) format"
    coreml_safety_checker.input_description["images"] = \
        f"Output of the vae_decoder ({pipe.vae.config.sample_size}x{pipe.vae.config.sample_size}) in channels-last (BHWC) format"
    coreml_safety_checker.input_description["adjustment"] = \
        "Bias added to the concept scores to trade off increased recall for reduce precision in the safety checker classifier"

    # Set the output descriptions
    coreml_safety_checker.output_description["filtered_images"] = \
        f"Identical to the input `images`. If safety checker detected any sensitive content, " \
        "the corresponding image is replaced with a blank image (zeros)"
    coreml_safety_checker.output_description["has_nsfw_concepts"] = \
        "Indicates whether the safety checker model found any sensitive content in the given image"
    coreml_safety_checker.output_description["concept_scores"] = \
        "Concept scores are the scores before thresholding at zero yields the `has_nsfw_concepts` output. " \
        "These scores can be used to tune the `adjustment` input"

    coreml_safety_checker.save(out_path)

    if args.check_output_correctness:
        baseline_out = pipe.safety_checker(
            **sample_safety_checker_inputs)[2].numpy()
        coreml_out = coreml_safety_checker.predict(
            coreml_sample_safety_checker_inputs)["concept_scores"]
        report_correctness(
            baseline_out, coreml_out,
            "safety_checker baseline PyTorch to reference CoreML")

    del traced_safety_checker, coreml_safety_checker, pipe.safety_checker
    gc.collect()

def _get_controlnet_base_model(controlnet_model_version):
    from huggingface_hub import model_info
    info = model_info(controlnet_model_version)
    return info.cardData.get("base_model", None)

def _is_union_controlnet(controlnet_model_version):
    # Download only the config
    config_path = hf_hub_download(controlnet_model_version, "config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Union models include num_control_types (or num_control_type)
    n = cfg.get("num_control_types") or cfg.get("num_control_type")

    return n is not None and n > 1, n

def convert_controlnet(pipe, args):
    """ Converts each ControlNet for Stable Diffusion
    """
    if not hasattr(pipe, "unet"):
        raise RuntimeError(
            "convert_unet() deletes pipe.unet to save RAM. "
            "Please use convert_controlnet() before convert_unet()")

    if not hasattr(pipe, "text_encoder"):
            raise RuntimeError(
                "convert_text_encoder() deletes pipe.text_encoder to save RAM. "
                "Please use convert_controlnet() before convert_text_encoder()")

    for i, controlnet_model_version in enumerate(args.convert_controlnet):
        base_model = _get_controlnet_base_model(controlnet_model_version)

        if base_model is None and args.model_version != "runwayml/stable-diffusion-v1-5":
            logger.warning(
                f"The original ControlNet models were trained using Stable Diffusion v1.5. "
                f"It is possible that model {args.model_version} is not compatible with controlnet.")
        # FIXME: This check does not seem to work with certain ControlNets for SDXL 1.0
        #if base_model is not None and base_model != args.model_version:
        #    raise RuntimeError(
        #        f"ControlNet model {controlnet_model_version} was trained using "
        #        f"Stable Diffusion model {base_model}.\n However, you specified "
        #        f"version {args.model_version} in the command line. Please, use "
        #        f"--model-version {base_model} to convert this model.")

        controlnet_model_name = controlnet_model_version.replace("/", "_")
        fname = f"ControlNet_{controlnet_model_name}.mlpackage"
        out_path = os.path.join(args.o, fname)

        if os.path.exists(out_path):
            logger.info(
                f"`controlnet_{controlnet_model_name}` already exists at {out_path}, skipping conversion."
            )
            continue

        is_union, _ = _is_union_controlnet(controlnet_model_version)
        if is_union:
            logger.info(f"{controlnet_model_version}` will be handled as Union controlnet.")

        # Import controlnet model and initialize reference controlnet
        if is_union:
            original_controlnet = ControlNetUnionModel.from_pretrained(
                controlnet_model_version,
                use_auth_token=True
            )
            # Call factory method with number of control types
            num_control_types = original_controlnet.config.num_control_type
            reference_controlnet = controlnetunion.make_controlnet(num_control_types=num_control_types, **original_controlnet.config).eval()
        else:
            original_controlnet = ControlNetModel.from_pretrained(
                controlnet_model_version,
                use_auth_token=True
            )
            reference_controlnet = controlnet.ControlNetModel(**original_controlnet.config).eval()

        load_state_dict_summary = reference_controlnet.load_state_dict(original_controlnet.state_dict())

        if True: # i == 0:
            batch_size = 2  # for classifier-free guidance
            if args.unet_batch_one:
                batch_size = 1  # for not using classifier-free guidance
            sample_shape = (
                batch_size,                    # B
                pipe.unet.config.in_channels,  # C
                (args.latent_h or pipe.unet.config.sample_size),  # H
                (args.latent_w or pipe.unet.config.sample_size),  # W
            )

            encoder_hidden_states_shape = (
                batch_size,
                reference_controlnet.config.cross_attention_dim or args.text_encoder_hidden_size or pipe.text_encoder.config.hidden_size,
                1,
                args.text_token_sequence_length or pipe.text_encoder.config.max_position_embeddings,
            )

            controlnet_cond_shape = (
                batch_size,                                           # B
                3,                                                    # C
                (args.latent_h or pipe.unet.config.sample_size) * 8,  # H
                (args.latent_w or pipe.unet.config.sample_size) * 8,  # w
            )

            # Create the scheduled timesteps for downstream use
            DEFAULT_NUM_INFERENCE_STEPS = 50
            pipe.scheduler.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS)

            # Prepare inputs
            sample_controlnet_inputs = OrderedDict([
                ("sample", torch.rand(*sample_shape)),
                ("timestep",
                torch.tensor([pipe.scheduler.timesteps[0].item()] *
                             (batch_size)).to(torch.float32)),
                ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
            ])

            # Setup inputs
            if is_union:
                num_control_type = reference_controlnet.config.num_control_type
                for i in range(num_control_type):
                    sample_controlnet_inputs['controlnet_cond_' + str(i)] = torch.rand(*controlnet_cond_shape)
                control_mode = list(range(num_control_type))
                control_type = torch.zeros(num_control_type).scatter_(0, torch.tensor(control_mode), 1)
                control_type_repeat_factor = (
                    batch_size
                )
                control_type = (
                    control_type.reshape(1, -1)
                    .to(torch.int32)
                    .repeat(control_type_repeat_factor, 1)
                )
                sample_controlnet_inputs['control_type'] = control_type
                conditioning_scale = torch.ones(
                    (num_control_type,)
                )
                sample_controlnet_inputs['conditioning_scale'] = conditioning_scale
            else:
                num_control_type = 1
                sample_controlnet_inputs['controlnet_cond'] = torch.rand(*controlnet_cond_shape)

            sample_controlnet_inputs_spec = {}
            for k, v in sample_controlnet_inputs.items():
                if isinstance(v, torch.Tensor):
                    sample_controlnet_inputs_spec[k] = (v.shape, v.dtype)
                elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                    sample_controlnet_inputs_spec[k] = ( [t.shape for t in v],
                                                         [t.dtype for t in v] )
                else:
                    sample_controlnet_inputs_spec[k] = type(v)
            logger.info(
                f"Sample ControlNet inputs spec: {sample_controlnet_inputs_spec}")

            baseline_sample_controlnet_inputs = deepcopy(sample_controlnet_inputs)
            baseline_sample_controlnet_inputs[
                "encoder_hidden_states"] = baseline_sample_controlnet_inputs[
                    "encoder_hidden_states"].squeeze(2).transpose(1, 2)

        num_residuals = reference_controlnet.get_num_residuals()
        output_keys = [f"additional_residual_{i}" for i in range(num_residuals)]

        coreml_sample_controlnet_inputs = {
            k: v.numpy().astype(np.float16) if isinstance(v, torch.Tensor)
               else [t.numpy().astype(np.float16) for t in v] if isinstance(v, list)
               else TypeError(f"Unsupported type for key '{k}': {type(v)}")
            for k, v in sample_controlnet_inputs.items()
        }

        # JIT trace
        logger.info("JIT tracing..")
        reference_controlnet = torch.jit.trace(reference_controlnet,
                                         list(sample_controlnet_inputs.values()))
        logger.info("Done.")

        if args.check_output_correctness:
            baseline_out = original_controlnet(**baseline_sample_controlnet_inputs,
                                     return_dict=False)
            reference_out = reference_controlnet(*sample_controlnet_inputs.values())
            report_correctness(
                baseline_out[-1].numpy(),
                reference_out[-1].numpy(),
                f"{controlnet_model_name} baseline to reference PyTorch")

        del original_controlnet
        gc.collect()

        coreml_controlnet, out_path = _convert_to_coreml(f"controlnet_{controlnet_model_name}", reference_controlnet,
                                                   coreml_sample_controlnet_inputs,
                                                   output_keys, args,
                                                   out_path=out_path)

        del reference_controlnet
        gc.collect()

        # Make inputs optional (for Union ControlNets)
        if is_union:
            spec = coreml_controlnet.get_spec()

            # Loop over all inputs in the model description
            for input_type in spec.description.input:
                if "controlnet_cond" in input_type.name:
                    input_type.type.isOptional = True

            # Update model
            coreml_controlnet = ct.models.MLModel(spec, weights_dir=coreml_controlnet.weights_dir)

        coreml_controlnet.author = f"Please refer to the Model Card available at huggingface.co/{controlnet_model_version}"
        coreml_controlnet.license = "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
        coreml_controlnet.version = controlnet_model_version
        coreml_controlnet.short_description = \
            "ControlNet is a neural network structure to control diffusion models by adding extra conditions. " \
            "Please refer to https://arxiv.org/abs/2302.05543 for details."

        # Set the input descriptions
        coreml_controlnet.input_description["sample"] = \
            "The low resolution latent feature maps being denoised through reverse diffusion"
        coreml_controlnet.input_description["timestep"] = \
            "A value emitted by the associated scheduler object to condition the model on a given noise schedule"
        coreml_controlnet.input_description["encoder_hidden_states"] = \
            "Output embeddings from the associated text_encoder model to condition to generated image on text. " \
            "A maximum of 77 tokens (~40 words) are allowed. Longer text is truncated. " \
            "Shorter text does not reduce computation."
        if is_union:
            for i in range(num_control_type):
                coreml_controlnet.input_description["controlnet_cond_" + str(i)] = \
                "An additional input image for ControlNet to condition the generated images."
            coreml_controlnet.input_description["control_type"] = \
            "A tensor with values `0` or `1` depending on whether the control type is used."
            coreml_controlnet.input_description["conditioning_scale"] = \
            "A tensor with values between `0` and `1` used as conditioning scales."

        # Set the output descriptions
        for i in range(num_residuals):
            coreml_controlnet.output_description[f"additional_residual_{i}"] = \
                "One of the outputs of each downsampling block in ControlNet. " \
                "The value added to the corresponding resnet output in UNet."

        coreml_controlnet.save(out_path)
        logger.info(f"Saved controlnet into {out_path}")

        # Parity check PyTorch vs CoreML
        if args.check_output_correctness:
            coreml_out = coreml_controlnet.predict(coreml_sample_controlnet_inputs)
            report_correctness(
                baseline_out[-1].numpy(),
                coreml_out[output_keys[-1]],
                "controlnet baseline PyTorch to reference CoreML"
            )

        del coreml_controlnet
        gc.collect()

def get_pipeline(args):
    model_version = args.model_version

    logger.info(f"Initializing DiffusionPipeline with {model_version}..")
    if args.custom_vae_version:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.custom_vae_version, torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(model_version,
                                            torch_dtype=torch.float16,
                                            variant="fp16",
                                            use_safetensors=True,
                                            vae=vae,
                                            use_auth_token=True)
    elif args.sd3_version:
        # SD3 uses standard SDXL diffusers pipeline besides the vae, denoiser, and T5 text encoder
        sdxl_base_version = "stabilityai/stable-diffusion-xl-base-1.0"
        args.xl_version = True
        logger.info(f"SD3 version specified, initializing DiffusionPipeline with {sdxl_base_version} for non-SD3 components..")
        pipe = DiffusionPipeline.from_pretrained(sdxl_base_version,
                                            torch_dtype=torch.float16,
                                            variant="fp16",
                                            use_safetensors=True,
                                            use_auth_token=True)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_version,
                                            torch_dtype=torch.float16,
                                            variant="fp16",
                                            use_safetensors=True,
                                            use_auth_token=True)

    logger.info(f"Done. Pipeline in effect: {pipe.__class__.__name__}")

    return pipe


def main(args):
    os.makedirs(args.o, exist_ok=True)

    # Instantiate diffusers pipe as reference
    pipe = get_pipeline(args)

    # Load Image Encoder
    if args.load_image_encoder:
        image_encoder = _load_image_encoder(args.load_image_encoder)
        pipe.image_encoder = image_encoder

    # Load IP Adapter
    if args.load_ip_adapter:
        _load_ip_adapter(args.load_ip_adapter, pipe)

    # Register the selected attention implementation globally
    unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[
        args.attention_implementation]
    logger.info(
        f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}"
    )

    # Convert models
    if args.convert_vae_decoder:
        logger.info("Converting vae_decoder")
        if args.sd3_version:
            convert_vae_decoder_sd3(args)
        else:
            convert_vae_decoder(pipe, args)
        logger.info("Converted vae_decoder")

    if args.convert_vae_encoder:
        logger.info("Converting vae_encoder")
        convert_vae_encoder(pipe, args)
        logger.info("Converted vae_encoder")

    if args.convert_controlnet:
        logger.info("Converting controlnet")
        convert_controlnet(pipe, args)
        logger.info("Converted controlnet")
        
    if args.convert_unet:
        # Check if we should use architectural chunking (requires pipe access)
        if args.unet_chunks == "architectural":
            logger.info("Converting unet with architectural chunking")
            convert_unet_architectural_chunks(pipe, args)
            logger.info("Converted unet with architectural chunking")
        else:
            logger.info("Converting unet")
            convert_unet(pipe, args)
            logger.info("Converted unet")

    if args.convert_text_encoder and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        logger.info("Converting text_encoder")
        convert_text_encoder(pipe.text_encoder, pipe.tokenizer, "text_encoder", args)
        del pipe.text_encoder
        logger.info("Converted text_encoder")

    if args.convert_text_encoder and hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        logger.info("Converting text_encoder_2")
        convert_text_encoder(pipe.text_encoder_2, pipe.tokenizer_2, "text_encoder_2", args)
        del pipe.text_encoder_2
        logger.info("Converted text_encoder_2")

    if args.convert_image_encoder:
        logger.info("Converting image_encoder")
        do_classifier_free_guidance = not args.unet_batch_one
        convert_image_encoder(pipe.image_encoder, "image_encoder", args, do_classifier_free_guidance)
        logger.info("Converted image_encoder")

    if args.convert_safety_checker:
        logger.info("Converting safety_checker")
        convert_safety_checker(pipe, args)
        logger.info("Converted safety_checker")

    if args.convert_unet and args.refiner_version is not None:
        logger.info(f"Converting refiner")
        del pipe
        gc.collect()
        original_model_version = args.model_version
        args.model_version = args.refiner_version
        pipe = get_pipeline(args)
        args.model_version = original_model_version
        convert_unet(pipe, args, model_name="refiner")
        del pipe
        gc.collect()
        logger.info(f"Converted refiner")

    if args.convert_mmdit:
        logger.info("Converting mmdit")
        convert_mmdit(args)
        logger.info("Converted mmdit")

    if args.quantize_nbits is not None:
        logger.info(f"Quantizing weights to {args.quantize_nbits}-bit precision")
        quantize_weights(args)
        logger.info(f"Quantized weights to {args.quantize_nbits}-bit precision")

    if args.unet_chunks is not None:
        # Skip if architectural chunking was already done during convert_unet
        if args.unet_chunks == "architectural":
            logger.info("Skipping chunk_unet (architectural chunking was done during conversion)")
        else:
            logger.info("Chunking unet")
            chunk_unet(args)
            logger.info("Chunked unet")

    if args.bundle_resources_for_swift_cli:
        logger.info("Bundling resources for the Swift CLI")
        bundle_resources_for_swift_cli(args)
        logger.info("Bundled resources for the Swift CLI")

def unet_chunks_type(value):
    if value in {"2", "4"}:
        return int(value)
    if value == "architectural":
        return value
    raise argparse.ArgumentTypeError(
        "Must be 2, 4, or 'architectural'"
    )

def parser_spec():
    parser = argparse.ArgumentParser()

    # Select which models to export (All are needed for text-to-image pipeline to function)
    parser.add_argument("--convert-text-encoder", action="store_true")
    parser.add_argument("--convert-vae-decoder", action="store_true")
    parser.add_argument("--convert-vae-encoder", action="store_true")
    parser.add_argument("--convert-unet", action="store_true")
    parser.add_argument("--convert-mmdit", action="store_true")
    parser.add_argument("--convert-safety-checker", action="store_true")
    parser.add_argument(
        "--convert-controlnet",
        nargs="*",
        type=str,
        help=
        "Converts a ControlNet model hosted on HuggingFace to CoreML format. " \
        "To convert multiple models, provide their names separated by spaces.",
    )
    parser.add_argument("--convert-image-encoder", action="store_true")
    parser.add_argument(
        "--load-image-encoder",
        type=str,
        default="",
        help=
        "Specific loading of image encoder: model_name[:subfolder] to load and use." \
        "If not defined, the default image encoder from the pipeline is used."
    )
    parser.add_argument(
        "--load-ip-adapter",
        type=str,
        default="",
        help=
        "Specific loading of an IP adapter: model_name[:subfolder] to load and use."
    )
    parser.add_argument(
        "--ip-scales",
        type=str,
        default="[]",
        help="JSON list/dict of IP-Adapter additional scales."
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help=
        ("The pre-trained model checkpoint and configuration to restore. "
         "For available versions: https://huggingface.co/models?search=stable-diffusion"
         ))
    parser.add_argument(
        "--refiner-version",
        default=None,
        help=
        ("The pre-trained refiner model checkpoint and configuration to restore. "
         "If specified, this argument will convert and bundle the refiner unet only alongside the model unet. "
         "If you would like to convert a refiner model on it's own, use the --model-version argument instead."
         "For available versions: https://huggingface.co/models?sort=trending&search=stable-diffusion+refiner"
         ))
    parser.add_argument(
        "--custom-vae-version",
        type=str,
        default=None,
        help=
        ("Custom VAE checkpoint to override the pipeline's built-in VAE. "
         "If specified, the specified VAE will be converted instead of the one associated to the `--model-version` checkpoint. "
         "No precision override is applied when using a custom VAE."
         ))
    parser.add_argument("--compute-unit",
                        choices=tuple(cu
                                      for cu in ct.ComputeUnit._member_names_),
                        default="ALL")
    parser.add_argument(
        "--latent-h",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of rows) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--latent-w",
        type=int,
        default=None,
        help=
        "The spatial resolution (number of cols) of the latent space. `Defaults to pipe.unet.config.sample_size`",
    )
    parser.add_argument(
        "--text-token-sequence-length",
        type=int,
        default=None,
        help=
        "The token sequence length for the text encoder. `Defaults to pipe.text_encoder.config.max_position_embeddings`",
    )
    parser.add_argument(
        "--text-encoder-hidden-size",
        type=int,
        default=None,
        help=
        "The hidden size for the text encoder. `Defaults to pipe.text_encoder.config.hidden_size`",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=tuple(ai
                      for ai in unet.AttentionImplementations._member_names_),
        default=unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name,
        help=
        "The enumerated implementations trade off between ANE and GPU performance",
    )
    parser.add_argument(
        "-o",
        default=os.getcwd(),
        help="The resulting mlpackages will be saved into this directory")
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        "If specified, compares the outputs of original PyTorch and final CoreML models and reports PSNR in dB. "
        "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
        )
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help=
        "If specified, generates two mlpackages out of the unet model which approximately equal weights sizes. "
        "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
        )
    parser.add_argument(
        "--unet-chunks",
        default=None,
        type=unet_chunks_type,
        help=(
        "If specified, generates mlpackages out of the unet model. "
        "2/4: approximately equal weight chunks (for ANE). "
        "architectural: Architectural chunks with explicit skip connections for SDXL. "
        "This is required for ANE deployment on iOS and iPadOS. Not required for macOS."
        ),
    )
    parser.add_argument(
        "--quantize-nbits",
        default=None,
        choices=(1, 2, 4, 6, 8),
        type=int,
        help="If specified, quantized each model to nbits precision"
    )
    parser.add_argument(
        "--unet-support-controlnet",
        action="store_true",
        help=
        "If specified, enable unet to receive additional inputs from controlnet. "
        "Each input added to corresponding resnet output."
        )
    parser.add_argument(
        "--unet-batch-one",
        action="store_true",
        help=
        "If specified, a batch size of one will be used for the unet, this is needed if you do not want to do "
        "classifier free guidance. Default unet batch size is two, which is needed for classifier free guidance."
        )
    parser.add_argument("--include-t5", action="store_true")

    # Swift CLI Resource Bundling
    parser.add_argument(
        "--bundle-resources-for-swift-cli",
        action="store_true",
        help=
        "If specified, creates a resources directory compatible with the sample Swift CLI. "
        "It compiles all four models and adds them to a StableDiffusionResources directory "
        "along with a `vocab.json` and `merges.txt` for the text tokenizer")
    parser.add_argument(
        "--text-encoder-vocabulary-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
        help="The URL to the vocabulary file use by the text tokenizer")
    parser.add_argument(
        "--text-encoder-merges-url",
        default=
        "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
        help="The URL to the merged pairs used in by the text tokenizer.")
    parser.add_argument(
        "--text-encoder-t5-url",
        default=
        "https://huggingface.co/argmaxinc/coreml-stable-diffusion-3-medium/resolve/main/TextEncoderT5.mlmodelc",
        help="The URL to the pre-converted T5 encoder model.")
    parser.add_argument(
        "--text-encoder-t5-config-url",
        default=
        "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer_config.json",
        help="The URL to the merged pairs used in by the text tokenizer.")
    parser.add_argument(
        "--text-encoder-t5-data-url",
        default=
        "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer.json",
        help="The URL to the merged pairs used in by the text tokenizer.")
    parser.add_argument(
        "--min-deployment-target",
        default="macOS13",
        help=(
            "Minimum deployment target for Core ML models. "
            "Valid options include macOS13, macOS14, macOS15, iOS16, iOS17, iOS18. "
            "For iOS 18 compatibility with advanced quantization features, use iOS18. "
            "Default is macOS13 for backwards compatibility."
        )
    )
    parser.add_argument(
        "--xl-version",
        action="store_true",
        help=("If specified, the pre-trained model will be treated as an instantiation of "
        "`diffusers.pipelines.StableDiffusionXLPipeline` instead of `diffusers.pipelines.StableDiffusionPipeline`"))
    parser.add_argument(
        "--sd3-version",
        action="store_true",
        help=("If specified, the pre-trained model will be treated as an SD3 model."))

    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()

    main(args)
