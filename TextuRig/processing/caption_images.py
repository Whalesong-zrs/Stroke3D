#!/usr/bin/env python3
"""Caption rendered asset images with Gemini using GOOGLE_API_KEY."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from tqdm import tqdm


PROMPT = """You are an expert 3D asset annotator. Write one concise descriptive
phrase of fewer than ten words. State the object identity, style, pose, and only
the most important visual feature. Do not add commentary and avoid the word
'model'. Examples: 'teddy bear wearing a t-shirt, bow tie, and hat.';
'pink and purple cube with spikes and a horn.'"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--requests-per-minute", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GOOGLE_API_KEY in the environment")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    interval = 60.0 / args.requests_per_minute

    for image_path in tqdm(sorted(args.images_dir.glob(args.pattern)), desc="captions"):
        output_path = args.output_dir / f"{image_path.stem}.txt"
        if output_path.exists():
            continue
        error: Exception | None = None
        for attempt in range(args.retries):
            try:
                with Image.open(image_path) as image:
                    response = client.generate_content([PROMPT, image])
                caption = response.text.strip().replace("\n", " ")
                output_path.write_text(caption + "\n", encoding="utf-8")
                error = None
                break
            except Exception as caught:
                error = caught
                time.sleep(max(interval, 2**attempt))
        if error is not None:
            print(f"ERROR {image_path.name}: {error}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
