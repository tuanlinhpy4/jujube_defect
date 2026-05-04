#!/usr/bin/env python3
"""
predict.py - MvANet inference script
====================================
Run inference on single images or a directory of images.
Supports the paper's 5-image voting scheme for individual jujube classification.

Usage:
    # Single image
    python predict.py --checkpoint runs/mvanet_3/run_01/best.pth --input image.jpg

    # Directory of images
    python predict.py --checkpoint runs/mvanet_3/run_01/best.pth --input ./test_images/

    # 5-image voting (for individual jujube with 5 rotation images)
    python predict.py --checkpoint best.pth --input ./jujube_001/ --voting
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from collections import Counter
from typing import List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model
from data import DEFAULT_CLASSES


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description="MvANet Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model (.pth)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture (default: from checkpoint)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image or directory")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Input resolution (default: from checkpoint config)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--voting", action="store_true",
                        help="Use 5-image majority voting (paper scheme)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Show top-k predictions per image")
    parser.add_argument("--output", type=str, default=None,
                        help="Save predictions to JSON file")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device, arch_override=None):
    """Load model plus preprocessing metadata from a trusted checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    arch = arch_override or cfg.get("model", {}).get("arch", "MvANet")
    model_name = cfg.get("model", {}).get("name", "mvanet_3")
    num_classes = cfg.get("model", {}).get("num_classes", 4)
    in_channels = cfg.get("model", {}).get("in_channels", 3)
    input_size = cfg.get("data", {}).get("input_size", 32)
    class_names = cfg.get("data", {}).get("classes", DEFAULT_CLASSES)
    mean = cfg.get("data", {}).get("normalize_mean")
    std = cfg.get("data", {}).get("normalize_std")

    model = build_model(
        arch=arch,
        name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, input_size, class_names, mean, std


def get_transform(input_size: int, mean=None, std=None):
    """Build the same deterministic preprocessing used for eval/test."""
    steps = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def predict_single(model, image_path: str, transform, device: torch.device,
                    class_names: List[str], top_k: int = 1):
    """Predict class for a single image."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item() * 100,
            "class_idx": idx.item(),
        })

    return results


def predict_batch(model, image_paths: List[str], transform,
                  device: torch.device, class_names: List[str]):
    """Predict classes for a batch of images."""
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    results = []
    for i, path in enumerate(image_paths):
        results.append({
            "file": os.path.basename(path),
            "class": class_names[preds[i].item()],
            "confidence": probs[i, preds[i]].item() * 100,
        })

    return results


def predict_with_voting(model, image_paths: List[str], transform,
                        device: torch.device, class_names: List[str]):
    """5-image majority voting for individual jujube classification.

    Paper: one jujube -> 5 rotation images -> majority vote.
    """
    results = predict_batch(model, image_paths, transform, device, class_names)

    # Majority vote
    votes = [r["class"] for r in results]
    vote_counts = Counter(votes)
    final_class = vote_counts.most_common(1)[0][0]

    # Average confidence for winning class
    winning_confs = [r["confidence"] for r in results if r["class"] == final_class]
    avg_conf = sum(winning_confs) / len(winning_confs)

    return {
        "final_class": final_class,
        "confidence": avg_conf,
        "votes": dict(vote_counts),
        "per_image": results,
    }


def get_image_paths(input_path: str) -> List[str]:
    """Get all image paths from file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        paths = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(f))
        return paths
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, input_size, class_names, mean, std = load_model(
        args.checkpoint, device, args.arch
    )

    if args.input_size:
        input_size = args.input_size

    transform = get_transform(input_size, mean=mean, std=std)
    print(f"Input size: {input_size}x{input_size} | Classes: {class_names}")

    # Get images
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s)\n")

    if not image_paths:
        print("No images found!")
        return

    all_results = []

    if args.voting and len(image_paths) < 5:
        raise ValueError("--voting requires at least 5 images")

    if args.voting:
        # Process in groups of 5 (paper: 5 images per jujube)
        print("Using 5-image majority voting scheme\n")
        leftover = len(image_paths) % 5
        if leftover:
            print(f"Warning: ignoring {leftover} leftover image(s) after full groups of 5.")
        for i in range(0, len(image_paths), 5):
            group = image_paths[i:i+5]
            if len(group) < 5:
                break

            result = predict_with_voting(model, group, transform, device,
                                         class_names)
            print(f"  Jujube {i//5 + 1}: {result['final_class']} "
                  f"({result['confidence']:.1f}%) "
                  f"[votes: {result['votes']}]")
            all_results.append(result)
    else:
        # Per-image prediction
        start = time.perf_counter()
        for path in image_paths:
            preds = predict_single(model, path, transform, device,
                                   class_names, top_k=args.top_k)
            filename = os.path.basename(path)
            top = preds[0]
            print(f"  {filename}: {top['class']} ({top['confidence']:.1f}%)", end="")
            if args.top_k > 1:
                others = ", ".join(
                    f"{p['class']}({p['confidence']:.1f}%)" for p in preds[1:]
                )
                print(f"  [{others}]", end="")
            print()

            all_results.append({
                "file": filename,
                "predictions": preds,
            })

        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n  Total: {elapsed:.1f}ms for {len(image_paths)} images "
              f"({elapsed/len(image_paths):.2f}ms/image)")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
