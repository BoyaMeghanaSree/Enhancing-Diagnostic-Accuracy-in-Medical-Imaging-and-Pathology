"""
Helper utility to create lightweight demo checkpoints and sample images.
Allows end-to-end testing of app.py and predict.py without needing 2GB Kaggle datasets.
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw
import timm
import torch


def create_demo_assets(root_dir):
    created = []
    # 1. Create runs_xray/resnet50
    xray_dir = os.path.join(root_dir, "runs_xray", "resnet50")
    os.makedirs(xray_dir, exist_ok=True)

    xray_meta = {
        "model": "resnet50",
        "timm_name": "resnet50",
        "params_M": 23.5,
        "classes": ["NORMAL", "PNEUMONIA"],
        "img_size": 224,
        "color": False,
        "normal_classes": ["NORMAL"],
        "seed": 42,
        "epochs_run": 1,
        "train_time_s": 5,
        "accuracy": 0.95,
        "macro_f1": 0.94,
        "macro_auc": 0.98
    }
    xray_meta_path = os.path.join(xray_dir, "results.json")
    with open(xray_meta_path, "w", encoding="utf-8") as f:
        json.dump(xray_meta, f, indent=2)
    created.append(xray_meta_path)

    xray_ckpt_path = os.path.join(xray_dir, "best.pt")
    if not os.path.exists(xray_ckpt_path):
        model = timm.create_model("resnet50", pretrained=False, num_classes=2)
        torch.save(model.state_dict(), xray_ckpt_path)
    created.append(xray_ckpt_path)

    # 2. Create runs_mri/resnet50 (Brain Tumor MRI - 4 classes)
    mri_dir = os.path.join(root_dir, "runs_mri", "resnet50")
    os.makedirs(mri_dir, exist_ok=True)

    mri_meta = {
        "model": "resnet50",
        "timm_name": "resnet50",
        "params_M": 23.5,
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "img_size": 224,
        "color": False,
        "normal_classes": ["no_tumor"],
        "seed": 42,
        "epochs_run": 1,
        "train_time_s": 5,
        "accuracy": 0.92,
        "macro_f1": 0.91,
        "macro_auc": 0.97
    }
    mri_meta_path = os.path.join(mri_dir, "results.json")
    with open(mri_meta_path, "w", encoding="utf-8") as f:
        json.dump(mri_meta, f, indent=2)
    created.append(mri_meta_path)

    mri_ckpt_path = os.path.join(mri_dir, "best.pt")
    if not os.path.exists(mri_ckpt_path):
        model = timm.create_model("resnet50", pretrained=False, num_classes=4)
        torch.save(model.state_dict(), mri_ckpt_path)
    created.append(mri_ckpt_path)

    # 2. Create sample scans
    samples_dir = os.path.join(root_dir, "sample_scans")
    os.makedirs(samples_dir, exist_ok=True)

    # Sample Chest X-Ray (synthetic chest cavity)
    xray_img_path = os.path.join(samples_dir, "sample_chest_xray.png")
    img_xray = Image.new("L", (256, 256), color=15)
    draw_xray = ImageDraw.Draw(img_xray)
    # Draw ribs/chest outlines
    draw_xray.ellipse([40, 30, 110, 210], fill=120, outline=180, width=2)
    draw_xray.ellipse([146, 30, 216, 210], fill=120, outline=180, width=2)
    draw_xray.rectangle([115, 20, 141, 230], fill=160)
    img_xray.save(xray_img_path)
    created.append(xray_img_path)

    # Sample Brain MRI (synthetic brain ellipse)
    mri_img_path = os.path.join(samples_dir, "sample_brain_mri.png")
    img_mri = Image.new("L", (256, 256), color=10)
    draw_mri = ImageDraw.Draw(img_mri)
    draw_mri.ellipse([30, 30, 226, 226], fill=140, outline=220, width=3)
    draw_mri.ellipse([70, 70, 186, 186], fill=90, outline=160, width=2)
    draw_mri.ellipse([130, 80, 170, 120], fill=240)  # synthetic lesion
    img_mri.save(mri_img_path)
    created.append(mri_img_path)

    return created


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.abspath(__file__))
    files = create_demo_assets(root)
    print(f"Created {len(files)} demo assets successfully:")
    for f in files:
        print(" -", f)
