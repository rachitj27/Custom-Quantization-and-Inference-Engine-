"""Prepare a Colab runtime so the TensorRT INT8 benchmark matches the CPU runs.

Run after cloning the fire-8 dataset:

    !git clone -q --depth 1 \\
        https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection.git /content/src
    !python setup_colab.py

Three adjustments make the TensorRT row comparable to the rest of the table:

1. **Remapped labels.** This is the important one. The upstream dataset ships
   3-class labels (0=Fire, 1=default, 2=smoke), but the model was trained on the
   2-class scheme produced by `datasets/fire-8/remap.py`: 0 stays fire, 1 is
   dropped, 2 becomes smoke. Score against the raw upstream labels and mAP comes
   out 0.0000 -- the ground truth simply does not describe the same classes the
   model predicts. Applying the same remap here reproduces the local labels
   exactly (verified byte-for-byte across all 49 test files: 31 fire + 20 smoke).
   It also makes every calibration image usable; Ultralytics otherwise rejects
   roughly half of them as "Label class 2 exceeds dataset class count 2".

2. **Same 49 test images.** The upstream repo tracks 55 test images, but six are
   absent from the working copy every other benchmark was scored on.

3. **Same 100 calibration images.** Ultralytics builds its TensorRT INT8
   calibration set from `data[split or "val"]`, which defaults to the validation
   split. Every other INT8 configuration in this repo calibrates on the first 100
   training images, so those are copied into `calib/` and `val:` points at it.

Re-running is safe: the dataset is re-copied from the pristine clone first, so
the remap is never applied twice (which would drop the smoke class).
"""

import os
import shutil

SRC = "/content/src/datasets/fire-8"
ROOT = "/content/fire-8"
N_CALIB = 100
SPLITS = ("train", "valid", "test")

# Tracked upstream but missing from the working copy the CPU benchmarks used.
MISSING_LOCALLY = [
    "fire2_mp4-50_jpg.rf.07b08a19a25759f464ebccd1158f1d23.jpg",
    "img_115_jpg.rf.e97bf08faaa0cec274caadec8924920d.jpg",
    "img_260_jpg.rf.5b8b0c766e34a110aee8d4806e44c796.jpg",
    "img_262_jpg.rf.084879e778014a5ada2f7921f59953cb.jpg",
    "pexels-alan-w-8365989_mp4-8_jpg.rf.e78f4eaaecff4f8896e4877db29d09e0.jpg",
    "smoke2_mp4-10_jpg.rf.1fa6f0db9bdabbf2c52ff08f1c18d040.jpg",
]


def remap_labels(labels_dir):
    """Upstream 3-class -> the 2-class scheme the model was trained on.

    0 (Fire) stays 0, 1 (default) is dropped, 2 (smoke) becomes 1.
    Mirrors datasets/fire-8/remap.py.
    """
    changed = 0
    for name in os.listdir(labels_dir):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, name)
        with open(path) as f:
            lines = f.readlines()

        out = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls == 0:
                out.append(" ".join(parts))
            elif cls == 2:
                out.append(" ".join(["1"] + parts[1:]))
            # cls == 1 ("default") is dropped

        with open(path, "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
        changed += 1
    return changed


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"{SRC} not found -- clone the dataset repo first.")

    # Always start from the pristine clone so the remap is applied exactly once.
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.copytree(SRC, ROOT)

    total = 0
    for split in SPLITS:
        d = os.path.join(ROOT, split, "labels")
        if os.path.isdir(d):
            total += remap_labels(d)
    print(f"remapped labels in {total} files (0=fire, 1=smoke)")

    removed = 0
    for name in MISSING_LOCALLY:
        for sub, fname in (("images", name),
                           ("labels", os.path.splitext(name)[0] + ".txt")):
            path = os.path.join(ROOT, "test", sub, fname)
            if os.path.exists(path):
                os.remove(path)
                removed += 1

    calib_img = os.path.join(ROOT, "calib", "images")
    calib_lbl = os.path.join(ROOT, "calib", "labels")
    os.makedirs(calib_img, exist_ok=True)
    os.makedirs(calib_lbl, exist_ok=True)

    train_img = os.path.join(ROOT, "train", "images")
    picked = sorted(f for f in os.listdir(train_img) if f.lower().endswith(".jpg"))[:N_CALIB]
    for name in picked:
        shutil.copy(os.path.join(train_img, name), os.path.join(calib_img, name))
        label = os.path.splitext(name)[0] + ".txt"
        src = os.path.join(ROOT, "train", "labels", label)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(calib_lbl, label))

    with open(os.path.join(ROOT, "data.yaml"), "w") as f:
        f.write(
            "# Colab-only config. `val` points at calib/ because Ultralytics\n"
            "# draws TensorRT INT8 calibration images from data[split or 'val'],\n"
            "# and every other INT8 config in this repo calibrates on these same\n"
            "# 100 training images. Accuracy is scored against test/ by\n"
            "# tensorrt_int8_colab.py, which reads the directories directly.\n"
            f"path: {ROOT}\n"
            "train: train/images\n"
            "val: calib/images\n"
            "test: test/images\n"
            "\n"
            "nc: 2\n"
            "names: ['fire', 'smoke']\n"
        )

    # Report the ground-truth totals so a mismatch is obvious immediately.
    counts = {0: 0, 1: 0}
    test_labels = os.path.join(ROOT, "test", "labels")
    test_images = os.path.join(ROOT, "test", "images")
    stems = {os.path.splitext(f)[0] for f in os.listdir(test_images)
             if f.lower().endswith(".jpg")}
    for name in os.listdir(test_labels):
        if os.path.splitext(name)[0] not in stems:
            continue
        for line in open(os.path.join(test_labels, name)):
            if line.strip():
                counts[int(line.split()[0])] = counts.get(int(line.split()[0]), 0) + 1

    print(f"removed {removed} files for the 6 images absent from the local working copy")
    print(f"calibration images: {len(picked)}")
    print(f"test images:        {len(stems)}  (expect 49)")
    print(f"ground truth:       {counts.get(0, 0)} fire + {counts.get(1, 0)} smoke "
          f"= {counts.get(0, 0) + counts.get(1, 0)}  (expect 31 + 20 = 51)")

    if len(stems) != 49 or counts.get(0) != 31 or counts.get(1) != 20:
        print("WARNING: does not match the CPU benchmarks -- mAP will not be comparable")


if __name__ == "__main__":
    main()
