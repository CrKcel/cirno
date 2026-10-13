#!/usr/bin/env python
"""Quick generation for smoke tests (no KV/state cache: recomputes the prefix every step, so keep outputs short).

    python scripts/sample.py --model runs/pretrain/final --raw --prompt "The capital of France is"
    python scripts/sample.py --model runs/sft/final --prompt "这张图里有什么？" --image cat.jpg
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from data import load_image, special_ids  # noqa: E402
from hvlm.modeling_hvlm import HVLMForConditionalGeneration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--raw", action="store_true", help="plain text continuation (base model), no chat format")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = HVLMForConditionalGeneration.from_pretrained(a.model).float().to(a.device).eval()
    sp = special_ids(tok)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    pix = None
    if a.raw:
        ids = enc(a.prompt)
    else:
        vc = model.config.vision_config
        body = []
        if a.image:
            pix = load_image(a.image, vc.image_size)[None].to(a.device)
            body = [sp["vision_start"]] + [sp["image_pad"]] * vc.num_image_tokens + [sp["vision_end"]]
        ids = ([sp["im_start"]] + enc("user\n") + body + enc(a.prompt) + [sp["im_end"]] + enc("\n")
               + [sp["im_start"]] + enc("assistant\n"))
    stop = tok.convert_tokens_to_ids("<|endoftext|>" if a.raw else "<|im_end|>")
    with torch.autocast(device_type=torch.device(a.device).type, dtype=torch.bfloat16):
        out = model.generate_simple(torch.tensor([ids], device=a.device), pixel_values=pix,
                                    max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                                    top_k=a.top_k, eos_token_id=stop)
    print(tok.decode(out[0, len(ids):], skip_special_tokens=True))


if __name__ == "__main__":
    main()
