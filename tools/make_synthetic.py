#!/usr/bin/env python
"""Generate a tiny synthetic corpus so the WHOLE pipeline (tokenizer -> pretrain -> align -> sft -> eval ->
sample) can be smoke-tested in a few minutes, and so you can see that the vision path really learns
something.

    python tools/make_synthetic.py --out data/synth
Creates:
    data/synth/text.jsonl        {"text": ...}  simple English / Chinese sentences
    data/synth/caps.jsonl        image captions  ("a red circle on a blue background")
    data/synth/sft.jsonl         VQA-style chat about the same images + text-only chat
    data/synth/eval_*.jsonl      one probe per eval kind (see hvlm/eval_core.py): knowledge probes with
                                 known colour/shape answers and one unanswerable question, a persona card,
                                 an IM-form turn and a role-play rubric.  `scripts/smoke_test.sh` runs the
                                 evaluation stage on these; a random model is expected to score at chance.
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
            open(os.path.join(a.out, "sft.jsonl"), "w", encoding="utf-8") as fs, \
            open(os.path.join(a.out, "eval_knowledge.jsonl"), "w", encoding="utf-8") as ek, \
            open(os.path.join(a.out, "eval_persona.jsonl"), "w", encoding="utf-8") as ep, \
            open(os.path.join(a.out, "eval_imform.jsonl"), "w", encoding="utf-8") as ei, \
            open(os.path.join(a.out, "eval_roleplay.jsonl"), "w", encoding="utf-8") as er:
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
            # Eval probes: the answerable ones have a known keyword, so a trained model's accuracy is
            # meaningful while a random one stays at chance.
            ek.write(json.dumps({"kind": "knowledge", "id": f"syn{i:05d}c", "category": "fact",
                                 "prompt": "图中的图形是什么颜色？", "any_of": [color, ZH[color]],
                                 "expect_unknown": False, "image": rel}, ensure_ascii=False) + "\n")
            ek.write(json.dumps({"kind": "knowledge", "id": f"syn{i:05d}s", "category": "fact",
                                 "prompt": "图中是什么形状？", "any_of": [shape, ZH_SHAPE[shape]],
                                 "expect_unknown": False, "image": rel}, ensure_ascii=False) + "\n")
        ek.write(json.dumps({"kind": "knowledge", "id": "synunk0", "category": "unknown",
                             "prompt": "我昨天穿的什么颜色的袜子？", "expect_unknown": True}, ensure_ascii=False) + "\n")
        ep.write(json.dumps({"kind": "persona", "id": "synp0",
                             "persona": "你是小明，男，18岁，高中生。说话简短。",
                             "probes": [{"field": "name", "question": "你叫什么？", "any_of": ["小明"]},
                                        {"field": "age", "question": "你多大了？", "any_of": ["18", "十八"]}]},
                            ensure_ascii=False) + "\n")
        ei.write(json.dumps({"kind": "imform", "id": "synim0", "tags": ["闲聊"],
                             "messages": [{"role": "user", "content": "在吗"}]}, ensure_ascii=False) + "\n")
        er.write(json.dumps({"kind": "roleplay", "id": "synrp0", "persona": "你是一只猫娘，句尾带「喵」。",
                             "history": [{"role": "user", "content": "在吗"}],
                             "rubric": [{"name": "口癖", "any_of": ["喵"]},
                                        {"name": "不出戏", "not_any_of": ["作为一个AI", "我是AI"]}]},
                            ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
