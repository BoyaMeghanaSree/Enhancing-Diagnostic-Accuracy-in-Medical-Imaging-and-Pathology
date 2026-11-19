"""
Medical image classification trainer + evaluator (timm models: ResNet, DenseNet, EfficientNet, ViT, DeiT, Swin, ConvNeXt, MaxViT).

Train :  python medical_imaging_models.py --data data_xray --model resnet50 --out runs_xray
Re-eval: python medical_imaging_models.py --data data_xray --model resnet50 --out runs_xray --eval_only

Fixes vs. original:
  * macro AUC was silently NaN for 2-class data (sklearn needs P(positive) 1-D for binary) -> fixed
  * model selection on validation macro-F1 (robust to class imbalance)
  * per-class report, ROC curve, sensitivity/specificity for "abnormal vs normal" (what the triage app does)
  * history / config saved; img_size + color flag saved so inference uses identical preprocessing
  * preprocessing imported from common.py; --eval_only mode; clear errors for missing folders
"""
import argparse, json, os, random, time
import numpy as np
import torch, torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import MODEL_ZOO, train_transform, eval_transform

DEFAULT_NORMAL = "NORMAL,no_tumor,notumor,Healthy"


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def create_model_instance(model_key, num_classes, pretrained, img_size=224):
    """Instantiate model with dynamic img_size adaptation for vision transformers and CNNs."""
    timm_name = MODEL_ZOO[model_key]
    try:
        return timm.create_model(timm_name, num_classes=num_classes, pretrained=pretrained, img_size=img_size)
    except TypeError:
        return timm.create_model(timm_name, num_classes=num_classes, pretrained=pretrained)


def get_loaders(data_dir, img_size, batch_size, workers, gray=True):
    for split in ("train", "val", "test"):
        if not os.path.isdir(os.path.join(data_dir, split)):
            raise FileNotFoundError(f"Missing '{split}' folder in '{data_dir}'. Run dataset_adapter.py first.")
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform(img_size, gray))streamlit run app.py

    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), eval_transform(img_size, gray))
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), eval_transform(img_size, gray))
    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise ValueError("train/val/test have different class folders")
    pin = torch.cuda.is_available()
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=workers, pin_memory=pin)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), train_ds.classes, train_ds


def class_weights(train_ds, device):
    counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    ys, ps, probs, loss_sum, n = [], [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        probs.append(torch.softmax(out, 1).cpu().numpy())
        ps.extend(out.argmax(1).cpu().numpy()); ys.extend(y.cpu().numpy())
    num_cls = getattr(model, "num_classes", 0)
    probs_arr = np.concatenate(probs) if probs else np.empty((0, num_cls))
    return np.array(ys), np.array(ps), probs_arr, loss_sum / max(n, 1)


def abnormal_scores(yt, probs, classes, normal_names):
    """(y_abnormal, P(abnormal)) - same definition the triage app uses: 1 - sum P(normal classes)."""
    norm_set = {n.strip().lower() for n in normal_names if n.strip()}
    nidx = [i for i, c in enumerate(classes) if c.lower() in norm_set]
    if not nidx or len(nidx) == len(classes): return None
    return (~np.isin(yt, nidx)).astype(int), 1.0 - probs[:, nidx].sum(1)


def compute_metrics(yt, pt, probs, classes, normal_names):
    p, r, f1, _ = precision_recall_fscore_support(yt, pt, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(yt, probs[:, 1]) if len(classes) == 2 else \
              roc_auc_score(yt, probs, multi_class="ovr", average="macro", labels=list(range(len(classes))))
        auc = float(auc)
        if np.isnan(auc): auc = None
    except (ValueError, IndexError):
        auc = None
    m = dict(accuracy=float(accuracy_score(yt, pt)), macro_precision=float(p), macro_recall=float(r),
             macro_f1=float(f1), macro_auc=auc,
             per_class=classification_report(yt, pt, labels=list(range(len(classes))), target_names=classes,
                                             output_dict=True, zero_division=0))
    ab = abnormal_scores(yt, probs, classes, normal_names)
    if ab is not None:
        y_ab, p_ab = ab
        tn, fp, fn, tp = confusion_matrix(y_ab, (p_ab >= 0.5).astype(int), labels=[0, 1]).ravel()
        try:
            ab_auc = float(roc_auc_score(y_ab, p_ab))
            if np.isnan(ab_auc): ab_auc = None
        except (ValueError, IndexError):
            ab_auc = None
        m["triage_abnormal_vs_normal"] = dict(auc=ab_auc, sensitivity=float(tp / max(tp + fn, 1)),
                                              specificity=float(tn / max(tn + fp, 1)), threshold=0.5)
    return m


def run(model_key, args, device):
    set_seed(args.seed)
    normal_names = [s.strip() for s in args.normal_classes.split(",") if s.strip()]
    out_dir = os.path.join(args.out, model_key); os.makedirs(out_dir, exist_ok=True)
    gray = not args.color
    train_l, val_l, test_l, classes, train_ds = get_loaders(args.data, args.img_size, args.batch, args.workers, gray)
    print("Classes:", classes)

    model = create_model_instance(model_key, num_classes=len(classes),
                                  pretrained=not (args.no_pretrained or args.eval_only),
                                  img_size=args.img_size)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ckpt = os.path.join(out_dir, "best.pt")
    history, train_time = [], 0.0

    if not args.eval_only:
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
        lr = args.lr if args.lr is not None else (1e-4 if model_key in ("resnet50", "convnext") else 5e-5)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
        use_amp = device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_val, bad, t0 = -1.0, 0, time.time()
        for ep in range(args.epochs):
            model.train(); run_loss, n = 0.0, 0
            for x, y in tqdm(train_l, desc=f"{model_key} ep {ep+1}/{args.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                run_loss += loss.item() * x.size(0); n += x.size(0)
            sched.step()
            yv, pv, _, vloss = evaluate(model, val_l, device, criterion)
            vacc = accuracy_score(yv, pv)
            vf1 = float(precision_recall_fscore_support(yv, pv, average="macro", zero_division=0)[2])
            tr_loss = run_loss / max(n, 1)
            history.append(dict(epoch=ep + 1, train_loss=tr_loss, val_loss=vloss, val_acc=vacc, val_macro_f1=vf1))
            print(f"[{model_key}] ep {ep+1}: train_loss {tr_loss:.4f} val_loss {vloss:.4f} val_acc {vacc:.4f} val_f1 {vf1:.4f}")
            if vf1 > best_val:
                best_val, bad = vf1, 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= args.patience:
                    print("Early stopping."); break
        train_time = time.time() - t0
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if history:
            plt.figure(figsize=(6, 4))
            epochs_x = [h["epoch"] for h in history]
            marker_style = "o" if len(history) <= 5 else None
            plt.plot(epochs_x, [h["train_loss"] for h in history], label="train", marker=marker_style)
            plt.plot(epochs_x, [h["val_loss"] for h in history], label="val", marker=marker_style)
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150); plt.close()

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Train first (without --eval_only).")
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    yt, pt, probs, _ = evaluate(model, test_l, device)   # test set used ONCE, after model selection on val
    metrics = compute_metrics(yt, pt, probs, classes, normal_names)
    print(classification_report(yt, pt, target_names=classes, zero_division=0))

    cm = confusion_matrix(yt, pt, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {model_key}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close()

    ab = abnormal_scores(yt, probs, classes, normal_names)
    if ab is not None and len(set(ab[0])) == 2:
        fpr, tpr, _ = roc_curve(*ab)
        plt.figure(figsize=(5, 5)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
        plt.xlabel("1 - specificity"); plt.ylabel("sensitivity"); plt.title("ROC: abnormal vs normal")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_abnormal.png"), dpi=150); plt.close()

    res_path = os.path.join(out_dir, "results.json")
    prev_results = {}
    if args.eval_only and os.path.exists(res_path):
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                prev_results = json.load(f)
        except Exception:
            pass

    epochs_run = prev_results.get("epochs_run", len(history))
    train_time_s = prev_results.get("train_time_s", round(train_time))

    res = dict(model=model_key, timm_name=MODEL_ZOO[model_key], params_M=round(n_params, 1), classes=classes,
               img_size=args.img_size, color=args.color, normal_classes=normal_names, seed=args.seed,
               epochs_run=epochs_run, train_time_s=train_time_s, **metrics)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {ckpt}\n       {res_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ val/ test/")
    ap.add_argument("--model", default="resnet50", choices=list(MODEL_ZOO) + ["all"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--color", action="store_true", help="keep RGB instead of grayscale->3ch")
    ap.add_argument("--normal_classes", default=DEFAULT_NORMAL, help="comma list of class names meaning 'no disease'")
    ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate saved best.pt on test set")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    for k in (list(MODEL_ZOO) if args.model == "all" else [args.model]):
        run(k, args, device)


if __name__ == "__main__":
    main()
