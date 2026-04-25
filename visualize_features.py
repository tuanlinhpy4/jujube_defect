#!/usr/bin/env python3
"""
visualize_features.py — Feature Map & Attention Visualization
==============================================================
Visualize intermediate feature maps, SE attention weights, and
per-receptor outputs (matching paper Fig. 8).

Usage:
    # Visualize feature maps for a single image
    python visualize_features.py --checkpoint best.pth --input image.jpg

    # Visualize for multiple images (one per class)
    python visualize_features.py --checkpoint best.pth --input ./samples/

    # Save to custom directory
    python visualize_features.py --checkpoint best.pth --input image.jpg --output-dir ./viz/
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models import build_model
from data import DEFAULT_CLASSES
from utils.visualize import (
    plot_feature_maps,
    plot_feature_maps_detailed,
    plot_attention_weights,
)


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MvANet features")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default=None,
                        help="Override architecture (default: from checkpoint)")
    parser.add_argument("--input", type=str, required=True,
                        help="Image file or directory")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_image_paths(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        return sorted(str(f) for f in p.iterdir()
                       if f.suffix.lower() in SUPPORTED_EXTS)
    else:
        raise FileNotFoundError(f"Not found: {input_path}")


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model_name = cfg.get("model", {}).get("name", "mvanet_3")
    num_classes = cfg.get("model", {}).get("num_classes", 4)
    input_size = args.input_size or cfg.get("data", {}).get("input_size", 32)
    class_names = cfg.get("data", {}).get("classes", DEFAULT_CLASSES)
    arch = args.arch or cfg.get("model", {}).get("arch", "MvANet")

    model = build_model(arch=arch, name=model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Get images
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in image_paths:
        filename = Path(img_path).stem
        print(f"\nProcessing: {img_path}")

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # Forward with intermediate features
        with torch.no_grad():
            features = model.forward_with_features(tensor)

        # Get predicted class
        pred_idx = features["logits"].argmax(1).item()
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        print(f"  Predicted: {pred_class}")

        # 1. Feature map overview
        plot_feature_maps(
            features, class_name=f"{filename} ({pred_class})",
            save_path=os.path.join(args.output_dir,
                                    f"{filename}_feature_maps.png")
        )

        # 2. Detailed per-receptor feature maps (paper Fig. 8 style)
        plot_feature_maps_detailed(
            features, class_name=f"{filename} ({pred_class})",
            save_path=os.path.join(args.output_dir,
                                    f"{filename}_detailed_features.png")
        )

        # 3. SE Attention weights
        attn_a = features["attn_a_weights"][0].cpu().numpy()
        attn_b = features["attn_b_weights"][0].cpu().numpy()
        plot_attention_weights(
            attn_a, attn_b,
            save_path=os.path.join(args.output_dir,
                                    f"{filename}_attention_weights.png")
        )

    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
