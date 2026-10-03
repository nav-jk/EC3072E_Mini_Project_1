import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

ADAIN_BASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0"
ADAIN_FILES = {
    "vgg_normalised.pth": f"{ADAIN_BASE}/vgg_normalised.pth",
    "decoder.pth": f"{ADAIN_BASE}/decoder.pth",
}

# File IDs taken from the official StyTr^2 repo README (diyiiyiii/StyTR-2):
# https://github.com/diyiiyiii/StyTR-2
STYTR2_GDRIVE_IDS = {
    "vgg_normalised.pth": "1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M",
    "vit_embedding.pth": "1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B",
    "decoder.pth": "1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9",
    "transformer.pth": "1dnobsaLeE889T_LncCkAA2RkqzwsfHYy",
}


def download_adain(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in ADAIN_FILES.items():
        dest = out_dir / fname
        if dest.exists():
            print(f"[AdaIN] {dest} already exists, skipping.")
            continue
        print(f"[AdaIN] Downloading {fname} ...")
        urlretrieve(url, dest)
        print(f"[AdaIN] Saved -> {dest}")


def download_stytr2(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
        import gdown  # noqa: F401

    import gdown

    for fname, file_id in STYTR2_GDRIVE_IDS.items():
        dest = out_dir / fname
        if dest.exists():
            print(f"[StyTr^2] {dest} already exists, skipping.")
            continue
        print(f"[StyTr^2] Downloading {fname} ...")
        try:
            gdown.download(id=file_id, output=str(dest), quiet=False)
            if not dest.exists() or dest.stat().st_size == 0:
                raise RuntimeError("Downloaded file missing/empty")
        except Exception as e:
            print(f"[StyTr^2] FAILED to download {fname}: {e}")
            print(f"  Manual fallback: open "
                  f"https://drive.google.com/file/d/{file_id}/view in a browser, "
                  f"download it, and place it at {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adain-dir", default="models/adain")
    parser.add_argument("--stytr2-dir", default="models/stytr2")
    parser.add_argument("--skip-adain", action="store_true")
    parser.add_argument("--skip-stytr2", action="store_true")
    args = parser.parse_args()

    if not args.skip_adain:
        download_adain(Path(args.adain_dir))
    if not args.skip_stytr2:
        download_stytr2(Path(args.stytr2_dir))

    print("\nDone. Expected layout:")
    print(f"  {args.adain_dir}/vgg_normalised.pth")
    print(f"  {args.adain_dir}/decoder.pth")
    print(f"  {args.stytr2_dir}/vgg_normalised.pth, vit_embedding.pth, decoder.pth, transformer.pth")


if __name__ == "__main__":
    main()
