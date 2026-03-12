"""
LongCat-Image-Edit inference script for Intel ARC GPU (XPU).

Strategy: Sequential loading — each component is loaded DIRECTLY onto the XPU
using device_map so weights are NEVER staged in system RAM.

Load order:
  1. VAE  — encode input image latents  → unload
  2. Text Encoder (FP8) — encode prompt  → unload
  3. Transformer  — denoising loop       → unload
  4. VAE  — decode result latents        → unload

Requires: torch >= 2.5 with XPU support, diffusers, transformers, accelerate
"""

import gc
import sys
import torch
import math
import numpy as np
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from diffusers import LongCatImageEditPipeline
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import (
    calculate_shift, retrieve_timesteps, calculate_dimensions, prepare_pos_ids, split_quotation,
)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_ID              = "meituan-longcat/LongCat-Image-Edit"
DEVICE                = "xpu"
DTYPE                 = torch.bfloat16

# Text Encoder FP8 on-the-fly quantization.
# NOTE: Requires PyTorch nightly or >= 2.5 with XPU FP8 kernel support.
# If you hit "aspect fp8 not supported" errors, set this to False.
QUANTIZE_TEXT_ENCODER = False
DTYPE_TE              = torch.float8_e4m3fn if QUANTIZE_TEXT_ENCODER else DTYPE

INPUT_IMAGE_PATH  = "assets/test.png"
PROMPT            = "将猫变成狗"
NEGATIVE_PROMPT   = ""
GUIDANCE_SCALE    = 4.5
NUM_INFERENCE_STEPS = 50
SEED              = 43
OUTPUT_PATH       = "./longcat_image_edit_xpu.png"
# ─────────────────────────────────────────────────────────────────────────────

TOKENIZER_MAX_LEN = 512

IMAGE_TOKEN = "<|image_pad|>"
PROMPT_PREFIX = (
    "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes "
    "of the input image(s). Then, based on the user's editing instructions, clearly and precisely "
    "determine how to modify the given image(s), ensuring that only the specified parts are altered "
    "and all other aspects remain consistent with the original(s).<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def free_vram(*refs):
    for r in refs:
        del r
    gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()


def load_directly_to_xpu(cls, *args, dtype, **kwargs):
    try:
        return cls.from_pretrained(
            *args,
            torch_dtype=dtype,
            device_map={"": DEVICE},
            **kwargs,
        )
    except Exception as e:
        print(f"  [warn] device_map load failed ({e}), falling back to CPU then XPU")
        model = cls.from_pretrained(*args, torch_dtype=dtype, low_cpu_mem_usage=True, **kwargs)
        return model.to(DEVICE)


def retrieve_latents(encoder_output, generator=None):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


@torch.no_grad()
def encode_prompt_edit(prompt_str, pil_image, tokenizer, text_processor, text_encoder, device):
    """Encode text+image prompt for the Edit pipeline."""
    raw = text_processor.image_processor(images=pil_image, return_tensors="pt")
    pixel_values = raw["pixel_values"].to(device)
    image_grid_thw = raw["image_grid_thw"].to(device)

    # Build token list for the text part
    all_tokens = []
    for sub, matched in split_quotation(prompt_str):
        if matched:
            for ch in sub:
                all_tokens.extend(tokenizer(ch, add_special_tokens=False)["input_ids"])
        else:
            all_tokens.extend(tokenizer(sub, add_special_tokens=False)["input_ids"])
    all_tokens = all_tokens[:TOKENIZER_MAX_LEN]

    padded = tokenizer.pad(
        {"input_ids": [all_tokens]},
        max_length=TOKENIZER_MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Expand image placeholder into actual image tokens
    merge_length = text_processor.image_processor.merge_size ** 2
    text = PROMPT_PREFIX
    while IMAGE_TOKEN in text:
        n_img_tokens = image_grid_thw.prod() // merge_length
        text = text.replace(IMAGE_TOKEN, "<|placeholder|>" * n_img_tokens, 1)
    text = text.replace("<|placeholder|>", IMAGE_TOKEN)

    prefix_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    prefix_len = prefix_ids.index(vision_start_id)
    suffix_len = len(suffix_ids)

    pt = padded.input_ids.dtype
    pm = padded.attention_mask.dtype

    input_ids = torch.cat([
        torch.tensor(prefix_ids, dtype=pt),
        padded.input_ids[0],
        torch.tensor(suffix_ids, dtype=pt),
    ]).unsqueeze(0).to(device)

    attn_mask = torch.cat([
        torch.ones(len(prefix_ids), dtype=pm),
        padded.attention_mask[0],
        torch.ones(len(suffix_ids), dtype=pm),
    ]).unsqueeze(0).to(device)

    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )
    embeds = out.hidden_states[-1].detach()
    embeds = embeds[:, prefix_len:-suffix_len, :]
    return embeds


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LongCat-Image-Edit — XPU Sequential Inference")
    print("=" * 60)

    device = torch.device(DEVICE)

    # ── Phase 0: lightweight objects ──────────────────────────────────────────
    print("\n[0/4] Loading scheduler / tokenizer / processor …")
    scheduler      = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tokenizer      = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")

    pipe = LongCatImageEditPipeline(
        scheduler=scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=tokenizer,
        text_processor=text_processor,
        transformer=None,
    )
    pipe._guidance_scale = GUIDANCE_SCALE
    do_cfg = GUIDANCE_SCALE > 1.0

    # Preprocess input image
    initial_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    w_img, h_img = initial_image.size
    calc_w, calc_h = calculate_dimensions(1024 * 1024, w_img / h_img)

    resized_image = pipe.image_processor.resize(initial_image, calc_h, calc_w)
    prompt_pil    = pipe.image_processor.resize(initial_image, calc_h // 2, calc_w // 2)
    image_tensor  = pipe.image_processor.preprocess(resized_image, calc_h, calc_w)

    # ── Phase 1: VAE encode input image ───────────────────────────────────────
    print(f"\n[1/4] Loading VAE directly to {DEVICE} for input image encoding …")
    vae = load_directly_to_xpu(AutoencoderKL, MODEL_ID, dtype=DTYPE, subfolder="vae")

    pipe.register_modules(vae=vae)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    num_ch    = 16

    # Compute latent dims
    lat_h = 2 * (int(calc_h) // (pipe.vae_scale_factor * 2))
    lat_w = 2 * (int(calc_w) // (pipe.vae_scale_factor * 2))

    img_t = image_tensor.to(device=device, dtype=DTYPE)
    with torch.no_grad():
        image_latents = retrieve_latents(vae.encode(img_t))
        image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
    image_latents_packed = pipe._pack_latents(image_latents, 1, num_ch, lat_h, lat_w)

    print("      VAE image encoding done. Unloading …")
    pipe.register_modules(vae=None)
    free_vram(vae)
    del vae

    # ── Phase 2: Text Encoder ─────────────────────────────────────────────────
    print(f"\n[2/4] Loading Text Encoder directly to {DEVICE} (dtype={DTYPE_TE}) …")
    text_encoder = load_directly_to_xpu(
        Qwen2_5_VLForConditionalGeneration,
        MODEL_ID,
        dtype=DTYPE_TE,
        subfolder="text_encoder",
    )

    print("      Encoding prompt …")
    with torch.no_grad():
        prompt_embeds = encode_prompt_edit(
            PROMPT, prompt_pil, tokenizer, text_processor, text_encoder, device
        ).to(DTYPE)
        if do_cfg:
            neg_embeds = encode_prompt_edit(
                NEGATIVE_PROMPT, prompt_pil, tokenizer, text_processor, text_encoder, device
            ).to(DTYPE)
        else:
            neg_embeds = None

    prompt_len = prompt_embeds.shape[1]

    print("      Unloading Text Encoder …")
    free_vram(text_encoder)
    del text_encoder

    # Build position IDs now that we know prompt_len
    text_ids = prepare_pos_ids(
        modality_id=0, type="text", start=(0, 0), num_token=prompt_len
    ).to(device)
    if do_cfg:
        neg_text_ids = prepare_pos_ids(
            modality_id=0, type="text", start=(0, 0), num_token=neg_embeds.shape[1]
        ).to(device)
    else:
        neg_text_ids = None

    latents_ids = prepare_pos_ids(
        modality_id=1, type="image",
        start=(prompt_len, prompt_len),
        height=lat_h // 2, width=lat_w // 2,
    ).to(device)
    img_lat_ids = prepare_pos_ids(
        modality_id=2, type="image",
        start=(prompt_len, prompt_len),
        height=lat_h // 2, width=lat_w // 2,
    ).to(device, dtype=torch.float32)  # float32 instead of float64 for XPU compatibility

    latent_image_ids = torch.cat([latents_ids, img_lat_ids], dim=0)

    # Noise latents
    noise_shape = (1, num_ch, lat_h, lat_w)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor(noise_shape, generator=generator, device=device, dtype=DTYPE)
    latents = pipe._pack_latents(latents, 1, num_ch, lat_h, lat_w)

    image_seq_len = latents.shape[1]
    sigmas = np.linspace(1.0, 1.0 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, NUM_INFERENCE_STEPS = retrieve_timesteps(
        scheduler, NUM_INFERENCE_STEPS, device, sigmas=sigmas, mu=mu
    )

    # ── Phase 3: Transformer denoising ────────────────────────────────────────
    print(f"\n[3/4] Loading Transformer directly to {DEVICE} …")
    transformer = load_directly_to_xpu(
        LongCatImageTransformer2DModel,
        MODEL_ID,
        dtype=DTYPE,
        subfolder="transformer",
    )
    print("      Transformer loaded. Denoising …")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            sys.stdout.write(f"\r      Step {i+1}/{NUM_INFERENCE_STEPS}")
            sys.stdout.flush()

            latent_input = torch.cat([latents, image_latents_packed], dim=1)
            ts = t.expand(latent_input.shape[0]).to(DTYPE)

            with transformer.cache_context("cond"):
                noise_text = transformer(
                    hidden_states=latent_input,
                    timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                noise_text = noise_text[:, :image_seq_len]

            if do_cfg:
                with transformer.cache_context("uncond"):
                    noise_uncond = transformer(
                        hidden_states=latent_input,
                        timestep=ts / 1000,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    noise_uncond = noise_uncond[:, :image_seq_len]
                noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
            else:
                noise_pred = noise_text

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print("\n      Denoising done. Unloading Transformer …")
    free_vram(transformer)
    del transformer

    # ── Phase 4: VAE decode ────────────────────────────────────────────────────
    print(f"\n[4/4] Loading VAE directly to {DEVICE} for decoding …")
    vae = load_directly_to_xpu(AutoencoderKL, MODEL_ID, dtype=DTYPE, subfolder="vae")
    pipe.register_modules(vae=vae)

    with torch.no_grad():
        latents_out = pipe._unpack_latents(latents, calc_h, calc_w, pipe.vae_scale_factor)
        latents_out = (latents_out / vae.config.scaling_factor) + vae.config.shift_factor
        latents_out = latents_out.to(dtype=vae.dtype)
        img_tensor = vae.decode(latents_out, return_dict=False)[0]

    image = pipe.image_processor.postprocess(img_tensor, output_type="pil")[0]
    image.save(OUTPUT_PATH)
    print(f"      Saved → {OUTPUT_PATH}")

    pipe.register_modules(vae=None)
    free_vram(vae)
    del vae

    print("\n[Done] All models unloaded.")
