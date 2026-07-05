"""Embed prompts with the frozen CLIP text encoder (matches make_captions.py)."""

import torch

MAX_TOKENS = 32
CLIP_NAME = "openai/clip-vit-base-patch32"


@torch.no_grad()
def embed_prompts(prompts: list[str], device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (ctx [N,L,512] fp16, mask [N,L] bool). Loads CLIP, then frees it."""
    from transformers import CLIPTextModel, CLIPTokenizerFast

    tok = CLIPTokenizerFast.from_pretrained(CLIP_NAME)
    enc = CLIPTextModel.from_pretrained(CLIP_NAME).to(device).eval()
    t = tok(prompts, padding="max_length", truncation=True,
            max_length=MAX_TOKENS, return_tensors="pt").to(device)
    out = enc(input_ids=t.input_ids, attention_mask=t.attention_mask)
    ctx = out.last_hidden_state.half()
    mask = t.attention_mask.bool()
    del enc
    if device != "cpu":
        torch.cuda.empty_cache()
    return ctx, mask
