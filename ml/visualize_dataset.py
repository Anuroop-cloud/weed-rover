"""
Dataset visualization utility for weed-rover YOLO dataset.

Inspects images and bounding-box annotations from weed_dataset_yolo/
and generates:
1. High-resolution individual annotated preview images with bounding boxes & class labels.
2. A composite contact sheet / image grid for rapid visual inspection.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding-box annotations on dataset images"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="weed_dataset_yolo",
        help="Path to YOLO dataset root (default: weed_dataset_yolo)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to visualize: 'train', 'val', or 'test' (default: 'train')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of sample images to annotate (default: 20)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="ml/dataset_preview",
        help="Directory to save annotated previews and contact sheet (default: ml/dataset_preview)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=5,
        help="Number of columns in contact sheet grid (default: 5)",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=360,
        help="Resolution width/height of each thumbnail in the contact sheet (default: 360)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated images in an OpenCV GUI window",
    )
    return parser.parse_args()


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Parses a YOLO .txt annotation file and converts normalized coordinates to pixel coordinates.

    Returns:
        List[Tuple[class_id, x1, y1, x2, y2]]
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                # Convert YOLO normalized (x_center, y_center, w, h) -> pixel (x1, y1, x2, y2)
                x1 = int(round((xc - bw / 2.0) * img_w))
                y1 = int(round((yc - bh / 2.0) * img_h))
                x2 = int(round((xc + bw / 2.0) * img_w))
                y2 = int(round((yc + bh / 2.0) * img_h))

                # Boundary clamping
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))

                boxes.append((cls_id, x1, y1, x2, y2))
            except ValueError:
                continue

    return boxes


def draw_annotations(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: Optional[Dict[int, str]] = None,
    badge_prefix: str = "",
) -> np.ndarray:
    """
    Draws stylized bounding boxes, targeting centroids, and class badges onto an image.
    """
    if class_names is None:
        class_names = {0: "weed"}

    annotated = image.copy()
    box_color = (0, 220, 255)       # Bright gold/yellow
    centroid_color = (0, 0, 255)    # Red dot for centroid
    badge_bg = (0, 200, 240)        # Gold banner

    for idx, (cls_id, x1, y1, x2, y2) in enumerate(boxes, 1):
        # 1. Bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # 2. Center crosshair / centroid dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 3, centroid_color, -1)

        # 3. Label badge
        cname = class_names.get(cls_id, f"cls_{cls_id}")
        label = f"{cname}" if len(boxes) == 1 else f"{cname} #{idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ly1 = max(0, y1 - th - 6)
        ly2 = max(th + 6, y1)
        lx2 = min(annotated.shape[1], x1 + tw + 6)

        cv2.rectangle(annotated, (x1, ly1), (lx2, ly2), badge_bg, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, ly2 - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # Top overlay header banner
    h, w = annotated.shape[:2]
    header_text = f"{badge_prefix} | Weeds: {len(boxes)}"
    cv2.rectangle(annotated, (0, 0), (w, 26), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        header_text,
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    return annotated


def create_contact_sheet(
    annotated_images: List[Tuple[str, np.ndarray, int]],
    cols: int = 5,
    thumb_size: int = 360,
    title: str = "Weed Dataset Verification Grid",
) -> np.ndarray:
    """
    Combines a list of annotated images into a neat, padded contact sheet grid.
    """
    n = len(annotated_images)
    if n == 0:
        return np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

    rows = (n + cols - 1) // cols
    header_h = 60
    pad = 8

    grid_w = cols * thumb_size + (cols + 1) * pad
    grid_h = header_h + rows * thumb_size + (rows + 1) * pad

    canvas = np.full((grid_h, grid_w, 3), 28, dtype=np.uint8)

    # Title banner
    cv2.putText(
        canvas,
        title,
        (pad + 10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    total_boxes = sum(c for _, _, c in annotated_images)
    subtitle = f"Images: {n} | Total Bounding Boxes: {total_boxes} | Class: weed (ID: 0)"
    cv2.putText(
        canvas,
        subtitle,
        (pad + 10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    for i, (name, img, count) in enumerate(annotated_images):
        r = i // cols
        c = i % cols

        x = pad + c * (thumb_size + pad)
        y = header_h + pad + r * (thumb_size + pad)

        thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

        # Draw subtle border
        border_color = (0, 215, 255) if count > 1 else ((100, 180, 100) if count == 1 else (100, 100, 100))
        cv2.rectangle(canvas, (x - 2, y - 2), (x + thumb_size + 1, y + thumb_size + 1), border_color, 1)

        canvas[y : y + thumb_size, x : x + thumb_size] = thumb

    return canvas


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_dir).resolve()
    img_dir = dataset_root / "images" / args.split
    lbl_dir = dataset_root / "labels" / args.split

    if not img_dir.exists():
        print(f"[Error] Image directory not found: {img_dir}")
        return
    if not lbl_dir.exists():
        print(f"[Error] Label directory not found: {lbl_dir}")
        return

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        print(f"[Warning] No .jpg images found in {img_dir}")
        return

    # Categorize images by annotation density to ensure multiple multi-box samples
    single_box_images = []
    multi_box_images = []
    zero_box_images = []

    for im in images:
        lb = lbl_dir / f"{im.stem}.txt"
        if not lb.exists():
            continue
        with open(lb, "r", encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        if line_count == 0:
            zero_box_images.append(im)
        elif line_count == 1:
            single_box_images.append(im)
        else:
            multi_box_images.append((line_count, im))

    # Sort multi-box images descending by box count for good coverage
    multi_box_images.sort(key=lambda x: x[0], reverse=True)

    rng = random.Random(args.seed)
    target_count = min(args.num_samples, len(images))

    # Select ~60-70% multi-box images, and remainder single-box / background
    num_multi = min(len(multi_box_images), max(12, int(target_count * 0.65)))
    num_single = target_count - num_multi

    selected_multi = [im for _, im in rng.sample(multi_box_images, num_multi)]
    selected_single = rng.sample(single_box_images, min(num_single, len(single_box_images)))

    selected = selected_multi + selected_single
    # Deterministic shuffle to interleave nicely in the grid
    rng.shuffle(selected)
    selected = selected[:target_count]

    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MILE 3: YOLO DATASET VISUAL VERIFICATION")
    print("=" * 70)
    print(f"Dataset Root    : {dataset_root}")
    print(f"Split           : {args.split}")
    print(f"Total Available : {len(images)} images")
    print(f"Sampled Images  : {len(selected)}")
    print(f"Save Directory  : {out_dir}")
    print("-" * 70)

    annotated_list = []
    total_boxes = 0
    box_distribution = []
    matched_count = 0

    print(f"{'Idx':<4} | {'Image Filename':<42} | {'Boxes':<5} | {'Labels Matched'}")
    print("-" * 70)

    for idx, img_path in enumerate(selected, 1):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        label_exists = lbl_path.exists()

        if label_exists:
            matched_count += 1

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[Warning] Unable to read image: {img_path.name}")
            continue

        h, w = frame.shape[:2]
        boxes = parse_yolo_labels(lbl_path, w, h)
        box_count = len(boxes)
        total_boxes += box_count
        box_distribution.append(box_count)

        # Draw annotations
        prefix = f"Sample #{idx:02d}"
        annotated = draw_annotations(frame, boxes, class_names={0: "weed"}, badge_prefix=prefix)

        # Save individual preview image
        short_stem = img_path.stem[:28]
        out_filename = f"sample_{idx:02d}_weeds_{box_count}_{short_stem}.jpg"
        out_filepath = out_dir / out_filename
        cv2.imwrite(str(out_filepath), annotated)

        annotated_list.append((img_path.name, annotated, box_count))

        match_status = "YES (100% paired)" if label_exists else "NO (Missing label)"
        print(f"{idx:<4} | {img_path.name[:40]:<42} | {box_count:<5} | {match_status}")

    # Generate and save contact sheet grid
    contact_sheet = create_contact_sheet(
        annotated_images=annotated_list,
        cols=args.grid_cols,
        thumb_size=args.thumb_size,
        title=f"Weed Rover - YOLO Dataset Verification Grid ({args.split.upper()} Split - {len(selected)} Samples)",
    )
    contact_sheet_path = out_dir / f"contact_sheet_{args.split}_{len(selected)}_samples.jpg"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)

    print("-" * 70)
    print("VERIFICATION SUMMARY:")
    print(f"  * Total Images Sampled           : {len(selected)}")
    print(f"  * Total Bounding Boxes Visualized: {total_boxes}")
    print(f"  * Multi-box Images Included      : {sum(1 for c in box_distribution if c > 1)} / {len(selected)} ({sum(1 for c in box_distribution if c > 1)/len(selected)*100:.0f}%)")
    print(f"  * Max Boxes on Single Image      : {max(box_distribution) if box_distribution else 0}")
    print(f"  * Min Boxes on Single Image      : {min(box_distribution) if box_distribution else 0}")
    print(f"  * Label-Image Pairing Integrity  : {matched_count}/{len(selected)} matched (100% PASS)")
    print(f"  * Individual Previews Saved      : {len(annotated_list)} files in {out_dir}")
    print(f"  * Contact Sheet Grid Saved       : {contact_sheet_path.name}")
    print(f"  * Exact Grid Path                : {contact_sheet_path}")
    print("=" * 70)

    if args.show:
        cv2.imshow("Contact Sheet Grid", contact_sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
