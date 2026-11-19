"""
Dataset adapter: turns the unzipped Kaggle folders into  <target>/{train,val,test}/<class>/img
  * Chest X-Ray Pneumonia (paultimothymooney/chest-xray-pneumonia)
  * Brain Tumor MRI, 4 classes (sartajbhuvaji/brain-tumor-classification-mri
    or masoudnickparvar/brain-tumor-mri-dataset - same folder layout)

Changes vs. the original:
  * The Kaggle X-ray 'val' folder has only 16 images -> useless for model selection.
    We pool train+val and make a new validation split (val_ratio).
  * X-ray filenames contain a patient id; we split BY PATIENT so the same child never
    appears in both train and val (prevents data leakage).
  * The official test folders are never touched/mixed into train or val.
  * Finds the split folders automatically (handles chest_xray/chest_xray nesting, Training/training...).
"""
import argparse, glob, os, random, re, shutil
from collections import defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png")


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT))


def patient_key(path):
    """Best-effort patient id from the Kaggle X-ray filenames; otherwise each file is its own group."""
    name = os.path.basename(path)
    m = re.match(r"person(\d+)_", name, re.I)          # PNEUMONIA: person123_bacteria_456.jpeg
    if m: return "p" + m.group(1)
    m = re.search(r"IM-(\d{4})", name)                  # NORMAL: IM-0115-0001.jpeg / NORMAL2-IM-1427-0001.jpeg
    if m: return "n" + m.group(1)
    return name


def group_split(files, val_ratio, seed):
    if not files:
        return [], []
    groups = defaultdict(list)
    for f in files: groups[patient_key(f)].append(f)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    target = max(1, int(len(files) * val_ratio)) if len(files) > 1 and val_ratio > 0 else int(len(files) * val_ratio)
    train, val = [], []
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])
    return train, val


def _find_split_dirs(source, train_names, test_names):
    for root, dirs, _ in os.walk(source):
        low = {d.lower(): d for d in dirs}
        t = next((low[n] for n in train_names if n in low), None)
        s = next((low[n] for n in test_names if n in low), None)
        if t and s: return os.path.join(root, t), os.path.join(root, s)
    raise FileNotFoundError(f"Could not find train/test folders inside {source}")


def _find_cls_dir(parent_dir, cls_name):
    """Case-insensitive matching of class directory."""
    if not parent_dir or not os.path.isdir(parent_dir):
        return None
    for d in os.listdir(parent_dir):
        if d.lower() == cls_name.lower() and os.path.isdir(os.path.join(parent_dir, d)):
            return os.path.join(parent_dir, d)
    return None


def _prepare_target(target_dir, overwrite):
    if os.path.exists(target_dir) and os.listdir(target_dir):
        if not overwrite:
            raise SystemExit(f"'{target_dir}' already exists and is not empty. Use --overwrite to rebuild it.")
        shutil.rmtree(target_dir, ignore_errors=True)


def _copy(files, dst):
    os.makedirs(dst, exist_ok=True)
    for f in files: shutil.copy(f, dst)


def _summary(target_dir):
    for split in ("train", "val", "test"):
        split_dir = os.path.join(target_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            n = len(list_images(os.path.join(split_dir, cls)))
            print(f"  {split:5s} {cls:20s} {n}")


def setup_chest_xray(source_dir, target_dir="data_xray", val_ratio=0.15, seed=42, overwrite=False):
    print(f"Setting up Chest X-Ray: {source_dir} -> {target_dir}")
    train_src, test_src = _find_split_dirs(source_dir, ("train",), ("test",))
    root = os.path.dirname(train_src)
    val_src = next((os.path.join(root, d) for d in os.listdir(root) if d.lower() in ("val", "validation")), None)
    _prepare_target(target_dir, overwrite)
    for cls in ("NORMAL", "PNEUMONIA"):
        train_cls_dir = _find_cls_dir(train_src, cls) or os.path.join(train_src, cls)
        pool = list_images(train_cls_dir)
        if val_src:
            val_cls_dir = _find_cls_dir(val_src, cls)
            if val_cls_dir:
                pool += list_images(val_cls_dir)
        tr, va = group_split(pool, val_ratio, seed)
        _copy(tr, os.path.join(target_dir, "train", cls))
        _copy(va, os.path.join(target_dir, "val", cls))
        test_cls_dir = _find_cls_dir(test_src, cls) or os.path.join(test_src, cls)
        _copy(list_images(test_cls_dir), os.path.join(target_dir, "test", cls))
    _summary(target_dir); print("Chest X-Ray dataset ready.")


def setup_brain_mri_4class(source_dir, target_dir="data_mri", val_ratio=0.15, seed=42, overwrite=False):
    print(f"Setting up 4-class Brain MRI: {source_dir} -> {target_dir}")
    train_src, test_src = _find_split_dirs(source_dir, ("training", "train"), ("testing", "test"))
    classes = sorted(d for d in os.listdir(train_src) if os.path.isdir(os.path.join(train_src, d)))
    _prepare_target(target_dir, overwrite)
    for cls in classes:
        # NOTE: these datasets have no patient ids, so this is a random split (some near-duplicate
        # slices of one patient may land in both train and val; the TEST set stays untouched).
        train_cls_dir = os.path.join(train_src, cls)
        tr, va = group_split(list_images(train_cls_dir), val_ratio, seed)
        _copy(tr, os.path.join(target_dir, "train", cls))
        _copy(va, os.path.join(target_dir, "val", cls))
        test_cls_dir = _find_cls_dir(test_src, cls) or os.path.join(test_src, cls)
        _copy(list_images(test_cls_dir), os.path.join(target_dir, "test", cls))
    _summary(target_dir); print("Brain MRI dataset ready.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xray_src", help="Path to unzipped chest_xray folder")
    ap.add_argument("--mri_src", help="Path to unzipped brain MRI folder")
    ap.add_argument("--xray_out", default="data_xray")
    ap.add_argument("--mri_out", default="data_mri")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.xray_src: setup_chest_xray(a.xray_src, a.xray_out, a.val_ratio, a.seed, a.overwrite)
    if a.mri_src: setup_brain_mri_4class(a.mri_src, a.mri_out, a.val_ratio, a.seed, a.overwrite)
    if not (a.xray_src or a.mri_src): ap.print_help()
