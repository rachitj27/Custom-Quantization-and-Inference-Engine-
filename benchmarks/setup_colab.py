"""Prepare a Colab runtime so the TensorRT INT8 benchmark matches the CPU runs.

Run after cloning the fire-8 dataset:

    !git clone -q --depth 1 \\
        https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection.git /content/src
    !cp -r /content/src/datasets/fire-8 /content/fire-8
    !python setup_colab.py

Two adjustments make the TensorRT row comparable to the rest of the table:

1. **Same 49 test images.** The upstream repo tracks 55 test images, but six of
   them are absent from the working copy every other benchmark was scored on.
   Scoring TensorRT over 55 while the CPU runtimes were scored over 49 would
   make the mAP column meaningless, so those six are removed here.

2. **Same 100 calibration images.** Ultralytics builds its TensorRT INT8
   calibration set from `data[split or "val"]`, and by default that is the
   44-image validation split. Every other INT8 configuration in this repo
   calibrates on the first 100 training images, so this writes them into a
   `calib/` directory and points `val:` at it.
"""

import os
import shutil

ROOT = "/content/fire-8"
N_CALIB = 100

# Tracked upstream but missing from the working copy the CPU benchmarks used.
MISSING_LOCALLY = [
    "fire2_mp4-50_jpg.rf.07b08a19a25759f464ebccd1158f1d23.jpg",
    "img_115_jpg.rf.e97bf08faaa0cec274caadec8924920d.jpg",
    "img_260_jpg.rf.5b8b0c766e34a110aee8d4806e44c796.jpg",
    "img_262_jpg.rf.084879e778014a5ada2f7921f59953cb.jpg",
    "pexels-alan-w-8365989_mp4-8_jpg.rf.e78f4eaaecff4f8896e4877db29d09e0.jpg",
    "smoke2_mp4-10_jpg.rf.1fa6f0db9bdabbf2c52ff08f1c18d040.jpg",
]


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"{ROOT} not found -- clone the dataset first.")

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
    shutil.rmtree(os.path.join(ROOT, "calib"), ignore_errors=True)
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

    n_test = len([f for f in os.listdir(os.path.join(ROOT, "test", "images"))
                  if f.lower().endswith(".jpg")])
    print(f"removed {removed} files for the 6 images absent from the local working copy")
    print(f"calibration images: {len(picked)}")
    print(f"test images:        {n_test}  (must be 49 to match the CPU benchmarks)")
    if n_test != 49:
        print("WARNING: test image count does not match the CPU runs -- mAP will not be comparable")


if __name__ == "__main__":
    main()
