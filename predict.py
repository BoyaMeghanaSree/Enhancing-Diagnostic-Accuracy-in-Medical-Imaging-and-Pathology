"""
Command-line prediction on one image.
  python src/predict.py --image scan.jpg --run_dir runs_xray --model resnet50 --normal_classes NORMAL
Saves <image>_gradcam.png next to the image (or --heatmap path).
"""
import argparse, json, os
from PIL import Image
from explain_and_triage import load_run, analyse

def main():
    ap = argparse.ArgumentParser(description="Command-line prediction and Grad-CAM generation for medical scans.")
    ap.add_argument("--image", required=True, help="Path to input scan image")
    ap.add_argument("--run_dir", required=True, help="Path to run directory (e.g. runs_xray)")
    ap.add_argument("--model", default="resnet50", help="Model key (e.g. resnet50)")
    ap.add_argument("--normal_classes", default="NORMAL,no_tumor,notumor,Healthy", help="Comma-separated list of normal class names")
    ap.add_argument("--threshold", type=float, default=0.5, help="High-risk probability threshold")
    ap.add_argument("--heatmap", default=None, help="Output path for heatmap image")
    a = ap.parse_args()

    if not os.path.exists(a.image):
        raise FileNotFoundError(f"Input image not found: {a.image}")

    model, meta = load_run(a.run_dir, a.model)
    normal_classes = [s.strip() for s in a.normal_classes.split(",") if s.strip()]
    with Image.open(a.image) as img:
        img_rgb = img.convert("RGB")
        r = analyse(img_rgb, model, a.model, meta, normal_classes, a.threshold)

    out = a.heatmap or os.path.splitext(a.image)[0] + "_gradcam.png"
    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    r.pop("heat").save(out)
    print(json.dumps(r, indent=2))
    print("Heatmap saved to", out)


if __name__ == "__main__":
    main()

