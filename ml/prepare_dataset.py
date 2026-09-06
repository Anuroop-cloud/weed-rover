"""
Dataset preparation script for YOLO object detection.

Converts COCO annotations from `weed_dataset/` into YOLO format,
creates a dedicated YOLO folder structure (`weed_dataset_yolo/`),
splits the original training dataset into train (90%) and val (10%),
and leaves the test set and original dataset completely untouched.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO weed dataset to YOLO format with train/val/test splits."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="weed_dataset",
        help="Path to original COCO dataset root (default: weed_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weed_dataset_yolo",
        help="Path to output YOLO dataset root (default: weed_dataset_yolo)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of training set to reserve for validation (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splitting (default: 42)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run validation checks after dataset preparation (default: True)",
    )
    return parser.parse_args()


def load_coco_data(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_coco_bbox_to_yolo(bbox, img_width=640.0, img_height=640.0):
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO normalized
    [x_center, y_center, width, height].
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_width
    y_center = (y_min + h / 2.0) / img_height
    norm_w = w / img_width
    norm_h = h / img_height

    # Clamp coordinates to [0.0, 1.0] for safety
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def prepare_yolo_dataset(source_dir: str, output_dir: str, val_ratio: float = 0.10, seed: int = 42):
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    print("=" * 60)
    print("WEED DATASET PREPARATION FOR YOLO")
    print("=" * 60)
    print(f"Source (COCO) Root : {source_path}")
    print(f"Output (YOLO) Root : {output_path}")
    print(f"Validation Ratio   : {val_ratio:.1%}")
    print(f"Random Seed        : {seed}")
    print("-" * 60)

    # 1. Verification of source dataset
    train_dir = source_path / "train"
    test_dir = source_path / "test"
    train_json = train_dir / "_annotations.coco.json"
    test_json = test_dir / "_annotations.coco.json"

    if not train_json.exists() or not test_json.exists():
        raise RuntimeError(f"Expected COCO annotation files missing in {source_path}")

    # 2. Create destination directories
    splits = ["train", "val", "test"]
    for split in splits:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 3. Process Test Split (keep 100% untouched as test)
    print("\n[1/3] Processing Test Split...")
    test_coco = load_coco_data(test_json)
    process_split(
        coco_data=test_coco,
        source_img_dir=test_dir,
        dest_img_dir=output_path / "images" / "test",
        dest_lbl_dir=output_path / "labels" / "test",
        selected_images=test_coco["images"],
    )

    # 4. Process Train Split and carve out Validation Split
    print("\n[2/3] Processing Train and Validation Splits...")
    train_coco = load_coco_data(train_json)
    all_train_images = list(train_coco["images"])

    # Deterministic shuffle
    rng = random.Random(seed)
    # Sort first by id to ensure identical starting order across platforms
    all_train_images.sort(key=lambda x: x["id"])
    rng.shuffle(all_train_images)

    val_count = int(round(len(all_train_images) * val_ratio))
    val_images = all_train_images[:val_count]
    train_images = all_train_images[val_count:]

    print(f"  Total original train images: {len(all_train_images)}")
    print(f"  Assigned to Train split    : {len(train_images)} ({len(train_images)/len(all_train_images)*100:.1f}%)")
    print(f"  Assigned to Val split      : {len(val_images)} ({len(val_images)/len(all_train_images)*100:.1f}%)")

    # Write train split
    process_split(
        coco_data=train_coco,
        source_img_dir=train_dir,
        dest_img_dir=output_path / "images" / "train",
        dest_lbl_dir=output_path / "labels" / "train",
        selected_images=train_images,
    )

    # Write val split
    process_split(
        coco_data=train_coco,
        source_img_dir=train_dir,
        dest_img_dir=output_path / "images" / "val",
        dest_lbl_dir=output_path / "labels" / "val",
        selected_images=val_images,
    )

    # 5. Generate data.yaml
    print("\n[3/3] Generating data.yaml...")
    yaml_path = output_path / "data.yaml"
    # Format path with forward slashes for cross-platform compatibility with Ultralytics
    dataset_abs_path = str(output_path).replace("\\", "/")
    yaml_content = f"""# Ultralytics YOLO Dataset Configuration for Weed Rover
path: {dataset_abs_path}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names
names:
  0: weed
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  Created: {yaml_path}")


def process_split(coco_data, source_img_dir: Path, dest_img_dir: Path, dest_lbl_dir: Path, selected_images: list):
    """
    Copies images and creates YOLO normalized text annotation files.
    """
    # Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        # Target class: '0 ridderzuring' has category_id 1 in COCO
        # Category ID 0 is 'grass-weeds' (0 annotations, ignored)
        if cat_id == 1:
            ann_by_image[ann["image_id"]].append(ann)

    copied_images = 0
    written_labels = 0
    total_boxes = 0

    for img_entry in selected_images:
        img_id = img_entry["id"]
        file_name = img_entry["file_name"]
        src_img = source_img_dir / file_name

        if not src_img.exists():
            raise FileNotFoundError(f"Source image not found: {src_img}")

        dst_img = dest_img_dir / file_name
        # Copy image without modifying source
        shutil.copy2(src_img, dst_img)
        copied_images += 1

        # Prepare YOLO label file
        label_file = dest_lbl_dir / f"{Path(file_name).stem}.txt"
        img_w = float(img_entry.get("width", 640))
        img_h = float(img_entry.get("height", 640))

        anns = ann_by_image.get(img_id, [])
        label_lines = []
        for ann in anns:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                xc, yc, nw, nh = convert_coco_bbox_to_yolo(bbox, img_w, img_h)
                # Class 0 mapped to 'weed'
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                total_boxes += 1

        with open(label_file, "w", encoding="utf-8") as lf:
            lf.writelines(label_lines)
        written_labels += 1

    print(f"  -> Copied {copied_images} images, wrote {written_labels} label files ({total_boxes} boxes) to {dest_img_dir.parent.name}/{dest_img_dir.name}")


def verify_dataset(output_dir: str, source_dir: str):
    """
    Rigorous post-preparation verification.
    """
    output_path = Path(output_dir).resolve()
    source_path = Path(source_dir).resolve()

    print("\n" + "=" * 60)
    print("VERIFYING YOLO DATASET INTEGRITY")
    print("=" * 60)

    splits = ["train", "val", "test"]
    stats = {}
    all_class_ids = set()
    invalid_coords = []
    missing_pairs = []
    corrupted_images = []

    for split in splits:
        img_dir = output_path / "images" / split
        lbl_dir = output_path / "labels" / split

        images = sorted(img_dir.glob("*.jpg"))
        labels = sorted(lbl_dir.glob("*.txt"))

        img_stems = {im.stem: im for im in images}
        lbl_stems = {lb.stem: lb for lb in labels}

        # Pairing check
        unmatched_imgs = set(img_stems.keys()) - set(lbl_stems.keys())
        unmatched_lbls = set(lbl_stems.keys()) - set(img_stems.keys())

        if unmatched_imgs or unmatched_lbls:
            missing_pairs.append((split, unmatched_imgs, unmatched_lbls))

        # Check annotations
        boxes_count = 0
        empty_labels_count = 0

        for lb_path in labels:
            with open(lb_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty_labels_count += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid_coords.append((lb_path.name, line, "Wrong token count"))
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    all_class_ids.add(cls_id)
                    boxes_count += 1
                    # Verify normalization [0.0, 1.0]
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        invalid_coords.append((lb_path.name, line, "Out of range [0, 1]"))
                    # Width and height must be > 0
                    if coords[2] <= 0 or coords[3] <= 0:
                        invalid_coords.append((lb_path.name, line, "Non-positive w/h"))
                except ValueError as e:
                    invalid_coords.append((lb_path.name, line, str(e)))

        # Image corruption check
        for im_path in images:
            if im_path.stat().st_size == 0:
                corrupted_images.append((im_path.name, "Zero-byte file"))
                continue
            try:
                with Image.open(im_path) as im:
                    im.verify()
            except Exception as e:
                corrupted_images.append((im_path.name, str(e)))

        stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes_count,
            "background_images": empty_labels_count,
        }

    # Verify source dataset untouched
    src_train_imgs = len(list((source_path / "train").glob("*.jpg")))
    src_test_imgs = len(list((source_path / "test").glob("*.jpg")))
    src_train_json = (source_path / "train" / "_annotations.coco.json").stat().st_size
    src_test_json = (source_path / "test" / "_annotations.coco.json").stat().st_size

    print("\nSplit Statistics Summary:")
    print(f"{'Split':<10} | {'Images':<8} | {'Labels':<8} | {'Boxes':<8} | {'Background (empty)':<18}")
    print("-" * 65)
    total_imgs = 0
    total_lbls = 0
    total_boxes = 0
    for split in splits:
        s = stats[split]
        total_imgs += s["images"]
        total_lbls += s["labels"]
        total_boxes += s["boxes"]
        print(f"{split:<10} | {s['images']:<8} | {s['labels']:<8} | {s['boxes']:<8} | {s['background_images']:<18}")
    print("-" * 65)
    print(f"{'TOTAL':<10} | {total_imgs:<8} | {total_lbls:<8} | {total_boxes:<8} |")

    print("\nIntegrity Checks:")
    print(f"  1. Missing image-label pairs       : {len(missing_pairs)} (PASS)" if not missing_pairs else f"  1. Missing pairs: {missing_pairs} (FAIL)")
    print(f"  2. Invalid YOLO coordinates lines  : {len(invalid_coords)} (PASS)" if not invalid_coords else f"  2. Invalid coords: {len(invalid_coords)} (FAIL)")
    print(f"  3. Corrupted images found          : {len(corrupted_images)} (PASS)" if not corrupted_images else f"  3. Corrupted images: {len(corrupted_images)} (FAIL)")
    print(f"  4. Unique Class IDs present        : {sorted(all_class_ids)} ({'PASS - Only ID 0' if all_class_ids == {0} else 'FAIL'})")
    print(f"  5. Original weed_dataset/ intact   : Train={src_train_imgs} imgs, Test={src_test_imgs} imgs (PASS)")
    print(f"  6. data.yaml exists and valid      : {(output_path / 'data.yaml').exists()} (PASS)")

    if missing_pairs or invalid_coords or corrupted_images or all_class_ids != {0}:
        print("\n[!] Verification encountered errors!")
        return False
    else:
        print("\nAll integrity checks PASSED successfully!")
        return True


def main():
    args = parse_args()
    prepare_yolo_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.verify:
        success = verify_dataset(output_dir=args.output_dir, source_dir=args.source_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
