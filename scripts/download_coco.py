"""
Usage:
    python scripts/download_coco.py
    python scripts/download_coco.py --max-images 8000 --split coco2017-train
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

COCO_URLS = {
    "coco2017-val": "http://images.cocodataset.org/zips/val2017.zip",
    "coco2017-train": "http://images.cocodataset.org/zips/train2017.zip",
    "coco2014-val": "http://images.cocodataset.org/zips/val2014.zip",
}


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        print(f"\rDownloading... {pct:5.1f}% ({downloaded/1e6:.0f} MB / {total_size/1e6:.0f} MB)", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(COCO_URLS.keys()), default="coco2017-val",
                         help="Which COCO zip to download.")
    parser.add_argument("--out-dir", default="data/coco", help="Output directory.")
    parser.add_argument("--max-images", type=int, default=5000,
                         help="Cap on number of images to keep (randomly sampled).")
    parser.add_argument("--test-fraction", type=float, default=0.1,
                         help="Fraction of kept images reserved for the held-out test/eval set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "_raw"
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    for d in (raw_dir, train_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / f"{args.split}.zip"
    url = COCO_URLS[args.split]

    if not zip_path.exists():
        print(f"Downloading {url} ...")
        urlretrieve(url, zip_path, reporthook=reporthook)
        print()
    else:
        print(f"{zip_path} already exists, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(members)
        members = members[: args.max_images]
        for m in members:
            zf.extract(m, raw_dir)

    # Flatten (COCO zips extract into a subfolder like val2017/xxxx.jpg)
    all_images = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.jpeg")) + list(raw_dir.rglob("*.png"))
    random.shuffle(all_images)
    n_test = max(1, int(len(all_images) * args.test_fraction))
    test_images = all_images[:n_test]
    train_images = all_images[n_test:]

    for img in train_images:
        shutil.copy(img, train_dir / img.name)
    for img in test_images:
        shutil.copy(img, test_dir / img.name)

    shutil.rmtree(raw_dir)
    zip_path.unlink()  # remove the zip to save disk space; re-download if needed

    print(f"Done. {len(train_images)} train images -> {train_dir}")
    print(f"      {len(test_images)} test images  -> {test_dir}")


if __name__ == "__main__":
    main()
