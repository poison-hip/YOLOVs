#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np


def _label_path_from_image_path(image_path: str) -> str:
    image_dir = os.path.dirname(image_path)
    label_dir = "labels".join(image_dir.rsplit("images", 1))
    if label_dir == image_dir:
        raise ValueError(f"Image path must contain 'images' folder: {image_path}")
    label_file = os.path.join(label_dir, os.path.basename(image_path))
    return os.path.splitext(label_file)[0] + ".txt"


def load_wh_pixels(train_list: str, img_w: int, img_h: int) -> np.ndarray:
    wh = []
    with open(train_list, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    for image_path in image_paths:
        label_path = _label_path_from_image_path(image_path)
        if not os.path.exists(label_path):
            continue
        try:
            labels = np.loadtxt(label_path, ndmin=2)
        except Exception:
            continue
        if labels.size == 0:
            continue
        # yolo format: cls cx cy w h, where w/h are normalized.
        widths = labels[:, 3] * img_w
        heights = labels[:, 4] * img_h
        valid = (widths > 0) & (heights > 0)
        for w, h in zip(widths[valid], heights[valid]):
            wh.append([float(w), float(h)])
    if not wh:
        raise RuntimeError("No valid labels found for anchor clustering.")
    return np.array(wh, dtype=np.float32)


def iou_wh(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    min_w = np.minimum(clusters[:, 0], box[0])
    min_h = np.minimum(clusters[:, 1], box[1])
    inter = min_w * min_h
    union = box[0] * box[1] + clusters[:, 0] * clusters[:, 1] - inter
    return inter / np.clip(union, 1e-9, None)


def kmeans_anchors(wh: np.ndarray, k: int, max_iter: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = wh[rng.choice(len(wh), size=k, replace=False)]
    prev_assign = None
    for _ in range(max_iter):
        distances = np.stack([1.0 - iou_wh(box, centroids) for box in wh], axis=0)
        assign = np.argmin(distances, axis=1)
        if prev_assign is not None and np.all(assign == prev_assign):
            break
        prev_assign = assign
        for i in range(k):
            pts = wh[assign == i]
            if len(pts):
                centroids[i] = np.median(pts, axis=0)
    return centroids


def avg_best_iou(wh: np.ndarray, anchors: np.ndarray) -> float:
    best = [np.max(iou_wh(box, anchors)) for box in wh]
    return float(np.mean(best))


def split_scales(anchors: np.ndarray, num_heads: int = 3) -> List[np.ndarray]:
    if len(anchors) % num_heads != 0:
        raise ValueError("Number of anchors must be divisible by num_heads.")
    order = np.argsort(anchors[:, 0] * anchors[:, 1])
    anchors = anchors[order]
    per_head = len(anchors) // num_heads
    return [anchors[i * per_head: (i + 1) * per_head] for i in range(num_heads)]


def to_cfg_anchor_line(anchors: np.ndarray) -> str:
    vals = []
    for w, h in anchors:
        vals.extend([str(int(round(w))), str(int(round(h)))])
    return "anchors = " + ", ".join(vals)


def main():
    parser = argparse.ArgumentParser(description="Compute YOLO anchors with IoU k-means from YOLO txt labels.")
    parser.add_argument("--train_list", required=True, help="Path to train.txt (image paths)")
    parser.add_argument("--img_width", type=int, required=True, help="Model input width")
    parser.add_argument("--img_height", type=int, required=True, help="Model input height")
    parser.add_argument("--num_anchors", type=int, default=9, help="Total anchors (default 9)")
    parser.add_argument("--num_heads", type=int, default=3, help="Number of detection heads (default 3)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    wh = load_wh_pixels(args.train_list, args.img_width, args.img_height)
    anchors = kmeans_anchors(wh, k=args.num_anchors, seed=args.seed)
    scales = split_scales(anchors, args.num_heads)
    merged = np.concatenate(scales, axis=0)
    score = avg_best_iou(wh, merged)

    print(f"Loaded boxes: {len(wh)}")
    print(f"Avg best IoU: {score:.4f}")
    print("")
    print("Small -> Large anchors by area:")
    for i, group in enumerate(scales):
        group_name = ["P2(stride4)", "P3(stride8)", "P4(stride16)"][i] if args.num_heads == 3 else f"head{i}"
        line = ", ".join([f"({int(round(w))},{int(round(h))})" for w, h in group])
        print(f"{group_name}: {line}")
    print("")
    print("Copy this into cfg [yolo] anchors line:")
    print(to_cfg_anchor_line(merged))
    print("Recommended masks for 3 heads:")
    print("P2: mask = 0,1,2")
    print("P3: mask = 3,4,5")
    print("P4: mask = 6,7,8")


if __name__ == "__main__":
    main()
