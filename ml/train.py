"""
Reproducible YOLOv8 baseline training script for weed detection.

Trains a lightweight YOLOv8 model (yolov8n.pt) on the prepared dataset (weed_dataset_yolo/data.yaml),
evaluates on the validation set, and copies the best model weights to models/best.pt.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 baseline model on weed dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model architecture or weights to start from (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16 for CPU/balanced training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device: 'cpu', '0', 'cuda:0' (default: cpu)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="weed_dataset_yolo/data.yaml",
        help="Path to dataset configuration YAML (default: weed_dataset_yolo/data.yaml)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project directory to save outputs (default: runs)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="weed_yolov8n_baseline",
        help="Experiment name under project (default: weed_yolov8n_baseline)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes (default: 0 for reliable Windows CPU execution)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience in epochs without mAP improvement (default: 8)",
    )
    return parser.parse_args()


def verify_dataset_yaml(yaml_path: Path):
    """Verifies that the dataset configuration YAML and split folders exist and are valid."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", "."))
    train_split = dataset_root / data_cfg.get("train", "images/train")
    val_split = dataset_root / data_cfg.get("val", "images/val")
    test_split = dataset_root / data_cfg.get("test", "images/test")

    if not train_split.exists():
        raise FileNotFoundError(f"Train directory not found: {train_split}")
    if not val_split.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_split}")
    if not test_split.exists():
        raise FileNotFoundError(f"Test directory not found: {test_split}")

    nc = data_cfg.get("nc", 0)
    names = data_cfg.get("names", {})
    if isinstance(names, dict):
        class_name = names.get(0, "")
    elif isinstance(names, list) and len(names) > 0:
        class_name = names[0]
    else:
        class_name = ""

    if nc != 1 or class_name != "weed":
        raise ValueError(
            f"Invalid dataset configuration: expected nc=1 and class name 'weed', but got nc={nc}, names={names}"
        )

    print("Pre-training Dataset Verification:")
    print(f"  * Dataset YAML      : {yaml_path} (Valid)")
    print(f"  * Train Images Dir  : {train_split} ({len(list(train_split.glob('*.jpg')))} images)")
    print(f"  * Val Images Dir    : {val_split} ({len(list(val_split.glob('*.jpg')))} images)")
    print(f"  * Test Images Dir   : {test_split} ({len(list(test_split.glob('*.jpg')))} images - NOT used in training)")
    print(f"  * Classes           : {nc} ('{class_name}')")


def find_best_epoch_from_csv(csv_path: Path) -> int:
    """Parses training results.csv to determine which epoch had the best mAP50."""
    if not csv_path.exists():
        return 1
    best_epoch = 1
    max_map50 = -1.0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip keys in case of whitespace in column headers
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                epoch_str = clean_row.get("epoch", "")
                map50_str = clean_row.get("metrics/mAP50(B)", "")
                if epoch_str and map50_str:
                    try:
                        ep = int(epoch_str)
                        m50 = float(map50_str)
                        if m50 > max_map50:
                            max_map50 = m50
                            best_epoch = ep
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[Warning] Could not parse results.csv: {e}")
    return best_epoch


def main():
    args = parse_args()
    yaml_path = Path(args.data).resolve()

    print("=" * 70)
    print("YOLO BASELINE MODEL TRAINING (MILE 4)")
    print("=" * 70)
    print(f"Model Architecture : {args.model}")
    print(f"Epochs             : {args.epochs}")
    print(f"Image Resolution   : {args.imgsz}x{args.imgsz}")
    print(f"Batch Size         : {args.batch}")
    print(f"Device             : {args.device}")
    print(f"Dataset YAML       : {yaml_path}")
    print(f"Output Project     : {args.project}")
    print(f"Experiment Name    : {args.name}")
    print("-" * 70)

    # 1. Pre-training verifications
    verify_dataset_yaml(yaml_path)

    # 2. Initialize Model
    print(f"\n[1/4] Loading model '{args.model}'...")
    model = YOLO(args.model)

    # 3. Launch Training
    print(f"\n[2/4] Starting training on training split (excluding test set)...")
    start_time = time.time()

    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    # 4. Locate best.pt and copy to models/best.pt
    print(f"\n[3/4] Locating best checkpoint and archiving to models/best.pt...")
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(args.project) / args.name
    trained_best_pt = save_dir / "weights" / "best.pt"

    if not trained_best_pt.exists():
        # Fallback search if path differs
        candidates = list(save_dir.glob("**/best.pt"))
        if candidates:
            trained_best_pt = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find best.pt in {save_dir}")

    models_dir = Path("models").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    target_best_pt = models_dir / "best.pt"

    shutil.copy2(trained_best_pt, target_best_pt)
    print(f"  * Checkpoint source : {trained_best_pt}")
    print(f"  * Copied to         : {target_best_pt} ({target_best_pt.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Run explicit validation on validation set
    print(f"\n[4/4] Evaluating best checkpoint on the validation set...")
    val_model = YOLO(str(target_best_pt))
    val_results = val_model.val(
        data=str(yaml_path),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        verbose=False,
    )

    # Extract metrics
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)

    # Determine best epoch
    csv_file = save_dir / "results.csv"
    best_epoch = find_best_epoch_from_csv(csv_file)

    print("\n" + "=" * 70)
    print("TRAINING & VALIDATION SUMMARY REPORT (MILE 4)")
    print("=" * 70)
    print(f"  * Model Architecture    : {args.model}")
    print(f"  * Epochs Trained        : {args.epochs}")
    print(f"  * Best Epoch            : {best_epoch}")
    print(f"  * Total Training Time   : {time_str} ({elapsed_time:.1f}s)")
    print(f"  * Precision (P)         : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  * Recall (R)            : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  * mAP@0.50 (mAP50)      : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"  * mAP@0.50:0.95 (mAP)   : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"  * Best Model Weights    : {target_best_pt}")
    print(f"  * Training Run Artifacts: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
