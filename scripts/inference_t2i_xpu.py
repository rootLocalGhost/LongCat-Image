"""
LongCat-Image Text-to-Image inference script for Intel ARC GPU (XPU).

Strategy: Sequential loading — each component is loaded DIRECTLY onto the XPU
using device_map so weights are never fully staged in system RAM.
Order: Text Encoder -> [unload] -> Transformer -> [unload] -> VAE -> [unload]

FP8 on-the-fly quantization is supported for the Text Encoder.
Set `QUANTIZE_TEXT_ENCODER = True` to enable it.
Requires: torch >= 2.5 with XPU support, diffusers, transformers, accelerate
"""

import gc
import sys
import torch
import numpy as np

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from diffusers import LongCatImagePipeline
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.longcat_image.pipeline_longcat_image import (
    calculate_shift, retrieve_timesteps,
    split_quotation, prepare_pos_ids,
    SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH,
    get_prompt_language,
)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_ID              = "meituan-longcat/LongCat-Image"
DEVICE                = "xpu"
DTYPE                 = torch.bfloat16

# Text Encoder FP8 on-the-fly quantization.
# NOTE: Requires PyTorch nightly or >= 2.5 with XPU FP8 kernel support.
# If you hit "aspect fp8 not supported" errors, set this to False.
QUANTIZE_TEXT_ENCODER = False
DTYPE_TE              = torch.float8_e4m3fn if QUANTIZE_TEXT_ENCODER else DTYPE

PROMPT = (
    "一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，"
    "表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上。"
)
NEGATIVE_PROMPT       = ""
HEIGHT                = 768
WIDTH                 = 1344
GUIDANCE_SCALE        = 4.0
NUM_INFERENCE_STEPS   = 50
SEED                  = 43
ENABLE_PROMPT_REWRITE = True
ENABLE_CFG_RENORM     = True
CFG_RENORM_MIN        = 0.0
OUTPUT_PATH           = "./longcat_image_t2i_xpu.png"
# ─────────────────────────────────────────────────────────────────────────────


def free_vram(*refs):
    """Delete objects and empty XPU cache."""
    for r in refs:
        del r
    gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()


def load_directly_to_xpu(cls, *args, dtype, **kwargs):
    """
    Load a model directly onto XPU using device_map so weights are NEVER
    fully buffered in system RAM.  Falls back gracefully if accelerate is
    not available.
    """
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


# ─── Helper functions copied from the pipeline (to avoid needing pipe.device)─
TOKENIZER_MAX_LEN = 512
PROMPT_PREFIX = (
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive text "
    "prompt based on an image content, suitable for input to a text-to-image model."
    "<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def encode_prompt_standalone(prompt_list, tokenizer, text_encoder, device):
    batch_all_tokens = []
    for each_prompt in prompt_list:
        all_tokens = []
        for sub, matched in split_quotation(each_prompt):
            if matched:
                for ch in sub:
                    all_tokens.extend(tokenizer(ch, add_special_tokens=False)["input_ids"])
            else:
                all_tokens.extend(tokenizer(sub, add_special_tokens=False)["input_ids"])
        all_tokens = all_tokens[:TOKENIZER_MAX_LEN]
        batch_all_tokens.append(all_tokens)

    padded = tokenizer.pad(
        {"input_ids": batch_all_tokens},
        max_length=TOKENIZER_MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    prefix_len, suffix_len = len(prefix_ids), len(suffix_ids)

    def _t(ids, ref):
        return torch.tensor(ids, dtype=ref.dtype)

    bs = padded.input_ids.size(0)
    input_ids = torch.cat([
        _t(prefix_ids, padded.input_ids).unsqueeze(0).expand(bs, -1),
        padded.input_ids,
        _t(suffix_ids, padded.input_ids).unsqueeze(0).expand(bs, -1),
    ], dim=-1).to(device)

    attn_mask = torch.cat([
        torch.ones(bs, prefix_len, dtype=padded.attention_mask.dtype),
        padded.attention_mask,
        torch.ones(bs, suffix_len, dtype=padded.attention_mask.dtype),
    ], dim=-1).to(device)

    # Cast input to bfloat16 if model is FP8 (forward pass still needs fp8-compatible inputs)
    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
    )
    embeds = out.hidden_states[-1].detach()
    embeds = embeds[:, prefix_len:-suffix_len, :]
    return embeds


@torch.no_grad()
def rewire_prompt_standalone(prompt, tokenizer, text_processor, text_encoder, device):
    lang = get_prompt_language(prompt)
    if lang == "zh":
        question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{prompt}\n改写后的prompt为："
    else:
        question = SYSTEM_PROMPT_EN + f"\nUser Input: {prompt}\nRewritten prompt:"

    message = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = text_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = text_processor(text=[text], padding=True, return_tensors="pt").to(device)
    gen_ids = text_encoder.generate(**inputs, max_new_tokens=TOKENIZER_MAX_LEN)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return text_processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LongCat-Image T2I — XPU Sequential Inference")
    print("=" * 60)

    device = torch.device(DEVICE)

    # ── Phase 0: lightweight objects (tiny, fine on CPU) ──────────────────────
    print("\n[0/3] Loading scheduler / tokenizer / processor …")
    scheduler      = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tokenizer      = Qwen2Tokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_processor = Qwen2VLProcessor.from_pretrained(MODEL_ID, subfolder="text_processor")

    # Build a minimal pipeline shell (no heavy models yet) so we can reuse
    # its helpers (prepare_latents, _pack/_unpack_latents, image_processor …)
    pipe = LongCatImagePipeline(
        scheduler=scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=tokenizer,
        text_processor=text_processor,
        transformer=None,
    )
    pipe._guidance_scale = GUIDANCE_SCALE
    do_cfg = GUIDANCE_SCALE > 1.0

    # ── Phase 1: Text Encoder ──────────────────────────────────────────────────
    print(f"\n[1/3] Loading Text Encoder directly to {DEVICE} (dtype={DTYPE_TE}) …")
    text_encoder = load_directly_to_xpu(
        Qwen2_5_VLForConditionalGeneration,
        MODEL_ID,
        dtype=DTYPE_TE,
        subfolder="text_encoder",
    )
    print("      Text Encoder loaded.")

    prompt = PROMPT
    if ENABLE_PROMPT_REWRITE:
        print("      Rewriting prompt …")
        prompt = rewire_prompt_standalone(prompt, tokenizer, text_processor, text_encoder, device)
        print("      Prompt rewritten.")

    print("      Encoding prompt embeddings …")
    # Cast embeddings to DTYPE even if encoder ran in FP8
    prompt_embeds = encode_prompt_standalone([prompt], tokenizer, text_encoder, device).to(DTYPE)
    text_ids = prepare_pos_ids(
        modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1]
    ).to(device)

    if do_cfg:
        neg_embeds = encode_prompt_standalone([NEGATIVE_PROMPT], tokenizer, text_encoder, device).to(DTYPE)
        neg_text_ids = prepare_pos_ids(
            modality_id=0, type="text", start=(0, 0), num_token=neg_embeds.shape[1]
        ).to(device)
    else:
        neg_embeds, neg_text_ids = None, None

    print("      Unloading Text Encoder …")
    free_vram(text_encoder)
    del text_encoder

    # ── Prepare latents & timesteps (CPU-only, no big model needed) ────────────
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    num_ch = 16
    latents, latent_img_ids = pipe.prepare_latents(
        1, num_ch, HEIGHT, WIDTH, DTYPE, device, generator, None
    )

    sigmas = np.linspace(1.0, 1.0 / NUM_INFERENCE_STEPS, NUM_INFERENCE_STEPS)
    mu = calculate_shift(
        latents.shape[1],
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, NUM_INFERENCE_STEPS = retrieve_timesteps(
        scheduler, NUM_INFERENCE_STEPS, device, sigmas=sigmas, mu=mu
    )

    # ── Phase 2: Transformer denoising  ───────────────────────────────────────
    print(f"\n[2/3] Loading Transformer directly to {DEVICE} (dtype={DTYPE}) …")
    transformer = load_directly_to_xpu(
        LongCatImageTransformer2DModel,
        MODEL_ID,
        dtype=DTYPE,
        subfolder="transformer",
    )
    print("      Transformer loaded. Starting denoising …")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            sys.stdout.write(f"\r      Step {i+1}/{NUM_INFERENCE_STEPS}")
            sys.stdout.flush()

            ts = t.expand(latents.shape[0]).to(DTYPE)
            with transformer.cache_context("cond"):
                noise_text = transformer(
                    hidden_states=latents,
                    timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_img_ids,
                    return_dict=False,
                )[0]

            if do_cfg:
                with transformer.cache_context("uncond"):
                    noise_uncond = transformer(
                        hidden_states=latents,
                        timestep=ts / 1000,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=latent_img_ids,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
                if ENABLE_CFG_RENORM:
                    cn = torch.norm(noise_text, dim=-1, keepdim=True)
                    nn_ = torch.norm(noise_pred, dim=-1, keepdim=True)
                    scale = (cn / (nn_ + 1e-8)).clamp(min=CFG_RENORM_MIN, max=1.0)
                    noise_pred = noise_pred * scale
            else:
                noise_pred = noise_text

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print("\n      Denoising complete. Unloading Transformer …")
    free_vram(transformer)
    del transformer

    # ── Phase 3: VAE decode ────────────────────────────────────────────────────
    print(f"\n[3/3] Loading VAE directly to {DEVICE} …")
    vae = load_directly_to_xpu(AutoencoderKL, MODEL_ID, dtype=DTYPE, subfolder="vae")
    print("      VAE loaded. Decoding latents …")

    with torch.no_grad():
        latents_out = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
        latents_out = (latents_out / vae.config.scaling_factor) + vae.config.shift_factor
        latents_out = latents_out.to(dtype=vae.dtype)
        image_tensor = vae.decode(latents_out, return_dict=False)[0]

    image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
    image.save(OUTPUT_PATH)
    print(f"      Saved → {OUTPUT_PATH}")

    free_vram(vae)
    del vae

    print("\n[Done] All models unloaded.")
