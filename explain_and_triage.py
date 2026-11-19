"""
Grad-CAM explainability + triage logic + inference helpers (used by app.py and predict.py).
Works for CNNs (ResNet, DenseNet, EfficientNet, ConvNeXt, MaxViT) and ViT/DeiT/Swin.

Fixes vs. original: EfficientNet layer is bn2 (post-BN/act features), grad enabled explicitly,
batch-size check, loading + preprocessing + triage moved here so app/CLI share one code path.
"""
import json, os
import numpy as np
import torch, torch.nn.functional as F
import timm
import matplotlib
from PIL import Image
from common import MODEL_ZOO, eval_transform  # MODEL_ZOO re-exported


def get_target_layer(model, key):
    """Returns (layer, channels_last). Last spatial feature map of each architecture."""
    if key == "resnet50":      return model.layer4[-1], False
    if key == "densenet":      return model.features, False
    if key == "efficientnet":  return model.bn2, False
    if key == "convnext":      return model.stages[-1], False
    if key == "maxvit":        return model.stages[-1], False
    if key in ("vit", "deit"): return model.blocks[-1].norm1, False
    if key == "swin":          return model.layers[-1].blocks[-1].norm1, True   # NHWC in timm>=0.9
    raise ValueError(f"Unknown model key: {key}")


def to_spatial(act, channels_last):
    if act.dim() == 4:
        return act.permute(0, 3, 1, 2) if channels_last else act
    if act.dim() == 3:  # (B, tokens, C): drop class token if present, reshape to grid
        B, L, C = act.shape
        n = int(round(L ** 0.5))
        if n * n != L:
            act = act[:, L - (int(round((L - 1) ** 0.5)) ** 2):, :]
            L = act.shape[1]; n = int(round(L ** 0.5))
        return act.reshape(B, n, n, C).permute(0, 3, 1, 2)
    raise ValueError(f"Unexpected activation shape: {tuple(act.shape)}")


def grad_cam(model, x, layer, channels_last, class_idx=None):
    """x: (1,3,H,W). Returns (cam HxW in [0,1] numpy, logits (1,K) tensor)."""
    assert x.shape[0] == 1, "grad_cam supports one image at a time"
    store = {}
    def fwd(module, inp, o):
        act = o[0] if isinstance(o, tuple) else o
        store["a"] = act
        if act.requires_grad:
            act.register_hook(lambda g: store.__setitem__("g", g))
    model.eval()
    h1 = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad()
            out = model(x)
            if class_idx is None:
                class_idx = int(out.argmax(1))
            out[0, class_idx].backward()
    finally:
        h1.remove()
    A = to_spatial(store["a"].detach(), channels_last)
    if "g" in store and store["g"] is not None:
        G = to_spatial(store["g"].detach(), channels_last)
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A).sum(1, keepdim=True))
    else:
        cam = F.relu(A.mean(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), out.detach()


def load_run(run_dir, model_key):
    """Load best.pt + results.json written by medical_imaging_models.py."""
    run = os.path.join(run_dir, model_key)
    meta_path = os.path.join(run, "results.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing results metadata at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = timm.create_model(MODEL_ZOO[model_key], pretrained=False, num_classes=len(meta["classes"]))
    ckpt_path = os.path.join(run, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, meta


def overlay(img, cam, size=224, alpha=0.45):
    base = np.array(img.convert("RGB").resize((size, size))).astype(float) / 255
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return Image.fromarray((((1 - alpha) * base + alpha * heat) * 255).astype(np.uint8))


def analyse(img, model, model_key, meta, normal_classes, threshold=0.5, low_conf_thr=0.60):
    """Full inference: preprocessing (identical to training eval) -> prediction -> Grad-CAM -> triage flag."""
    classes, size = meta["classes"], meta.get("img_size", 224)
    device = next(model.parameters()).device
    x = eval_transform(size, gray=not meta.get("color", False))(img.convert("RGB")).unsqueeze(0).to(device)
    layer, ch_last = get_target_layer(model, model_key)
    cam, logits = grad_cam(model, x, layer, ch_last)
    probs = torch.softmax(logits, 1)[0].cpu().numpy()
    pred = int(probs.argmax())
    nidx = [classes.index(n) for n in normal_classes if n in classes]
    p_abn = 1.0 - float(sum(probs[i] for i in nidx)) if nidx else 1.0 - float(probs[pred])
    p_abn = max(0.0, min(1.0, p_abn))
    return dict(pred=classes[pred], conf=float(probs[pred]), p_abn=p_abn,
                flag="HIGH-RISK" if p_abn >= threshold else "ROUTINE",
                low_conf=int(probs[pred] < low_conf_thr),
                probs={k: float(v) for k, v in zip(classes, probs)}, heat=overlay(img, cam, size))

