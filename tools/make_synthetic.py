#!/usr/bin/env python
"""Generate a tiny synthetic corpus so the WHOLE pipeline (tokenizer -> pretrain -> align -> sft -> sample) can be
smoke-tested in a few minutes, and so you can see that the vision path really learns something.

    python tools/make_synthetic.py --out data/synth
Creates:
    data/synth/text.jsonl        {"text": ...}  simple English / Chinese sentences
    data/synth/caps.jsonl        image captions  ("a red circle on a blue background")
    data/synth/sft.jsonl         VQA-style chat about the same images + text-only chat
    data/synth/images/*.png      224x224 shapes
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (50, 80, 220), "yellow": (240, 210, 40),
          "purple": (150, 60, 170), "orange": (240, 140, 30)}
ZH = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色", "purple": "紫色", "orange": "橙色"}
SHAPES = ["circle", "square", "triangle"]
ZH_SHAPE = {"circle": "圆形", "square": "正方形", "triangle": "三角形"}


def draw(shape, color, bg, size=224, rng=random):
    im = Image.new("RGB", (size, size), COLORS[bg])
    d = ImageDraw.Draw(im)
    r = rng.randint(40, 75)
    cx, cy = rng.randint(r, size - r), rng.randint(r, size - r)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color])
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=COLORS[color])
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n_images", type=int, default=3000)
    ap.add_argument("--n_text", type=int, default=20000)
    a = ap.parse_args()
    rng = random.Random(0)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    subjects = ["The cat", "A small robot", "My teacher", "The old river", "A young farmer", "The red car"]
    verbs = ["likes", "sees", "builds", "remembers", "follows", "paints"]
    objects = ["a bright idea", "the quiet town", "three green apples", "a long road", "the morning light"]
    with open(os.path.join(a.out, "text.jsonl"), "w", encoding="utf-8") as f:
        for _ in range(a.n_text):
            if rng.random() < 0.5:
                t = " ".join(f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}." for _ in range(rng.randint(2, 6)))
            else:
                t = "".join(f"{rng.choice(['我', '他', '小明', '老师'])}{rng.choice(['喜欢', '看见', '记得', '正在画'])}{rng.choice(['一本书', '三个苹果', '很长的路', '早晨的阳光'])}。" for _ in range(rng.randint(2, 6)))
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(os.path.join(a.out, "caps.jsonl"), "w", encoding="utf-8") as fc, \
            open(os.path.join(a.out, "sft.jsonl"), "w", encoding="utf-8") as fs:
        for i in range(a.n_images):
            shape, color, bg = rng.choice(SHAPES), rng.choice(list(COLORS)), rng.choice(list(COLORS))
            while bg == color:
                bg = rng.choice(list(COLORS))
            rel = f"images/{i:05d}.png"
            draw(shape, color, bg, rng=rng).save(os.path.join(a.out, rel))
            fc.write(json.dumps({"image": rel, "conversations": [
                {"from": "human", "value": "<image>\nDescribe the image briefly."},
                {"from": "gpt", "value": f"a {color} {shape} on a {bg} background."}]}) + "\n")
            qa = [("What shape is in the image?", f"A {shape}."), ("What color is the shape?", f"It is {color}."),
                  ("What color is the background?", f"The background is {bg}."),
                  ("图中的图形是什么颜色？", f"是{ZH[color]}的。"), ("图中是什么形状？", f"是{ZH_SHAPE[shape]}。")]
            q, ans = rng.choice(qa)
            fs.write(json.dumps({"image": rel, "conversations": [
                {"from": "human", "value": f"<image>\n{q}"}, {"from": "gpt", "value": ans}]}, ensure_ascii=False) + "\n")
            if i % 5 == 0:
                fs.write(json.dumps({"messages": [{"role": "user", "content": "Say hello."},
                                                  {"role": "self", "content": "Hi! What's up?"}]}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
