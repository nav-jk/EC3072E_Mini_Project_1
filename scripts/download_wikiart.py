"""
Download a WikiArt style-image subset, organized into per-style folders:
    data/wikiart/<style_name>/*.jpg

This version accesses the Hugging Face Parquet export directly and uses
DuckDB to filter the style column before reading the matching image rows.

It preserves the existing command-line interface:

    python scripts/download_wikiart.py
    python scripts/download_wikiart.py --styles expressionism cubism impressionism --per-style 150

Requirements:
    pip install duckdb requests pillow datasets
"""

import argparse
import io
import os
import sys
from pathlib import Path

import requests
from PIL import Image


DATASET_NAME = "huggan/wikiart"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"

PARQUET_API = (
    "https://huggingface.co/api/datasets/"
    "huggan/wikiart/parquet"
)


def get_style_mapping():
    """
    Obtain the WikiArt style ClassLabel mapping without downloading images.

    The dataset's style column is an integer ClassLabel. We use the Hugging
    Face Datasets library only for its schema/feature information.
    """
    from datasets import load_dataset

    print("Reading WikiArt style labels...")

    ds = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )

    features = ds.features["style"]

    return {
        features.int2str(i).lower(): i
        for i in range(features.num_classes)
    }


def match_styles(requested_styles, name_to_id):
    """
    Preserve the original case-insensitive/fuzzy style matching behavior.
    """
    wanted = {}

    for style in requested_styles:
        key = style.lower().replace(" ", "_")

        match = next(
            (
                name
                for name in name_to_id
                if key in name or name in key
            ),
            None,
        )

        if match is None:
            print(
                f"[WARN] Could not find a WikiArt style matching "
                f"'{style}'. Available styles include: "
                f"{sorted(name_to_id.keys())[:10]} ..."
            )
            continue

        wanted[style] = name_to_id[match]

    if not wanted:
        raise SystemExit(
            "No requested styles matched the dataset's style labels. "
            "Adjust --styles."
        )

    return wanted


def get_parquet_urls():
    """
    Get the Parquet files generated for the WikiArt dataset.

    The Hugging Face Hub API returns files grouped by config and split.
    """
    token = os.environ.get("HF_TOKEN")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        PARQUET_API,
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    if DATASET_CONFIG in data:
        split_data = data[DATASET_CONFIG]
    elif "default" in data:
        split_data = data["default"]
    else:
        # Some API responses can be represented differently.
        split_data = data

    urls = split_data.get(DATASET_SPLIT, [])

    if not urls:
        raise RuntimeError(
            f"No Parquet files found for "
            f"{DATASET_CONFIG}/{DATASET_SPLIT}."
        )

    return urls


def get_parquet_files():
    """
    Resolve the Parquet URL(s) and print what will be queried.
    """
    print("Getting WikiArt Parquet files from Hugging Face...")

    urls = get_parquet_urls()

    print(f"Found {len(urls)} Parquet file(s).")

    for i, url in enumerate(urls):
        print(f"  [{i}] {url}")

    return urls


def existing_count(directory):
    """
    Count existing image files so an interrupted download can resume.
    """
    extensions = {
        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG",
        ".png",
        ".PNG",
    }

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix in extensions
    )


def extract_image_bytes(image_value):
    """
    Handle the image representation returned by DuckDB for a Hugging Face
    Image feature.

    Depending on the Parquet/Arrow representation, DuckDB may return the
    image as:
      - a struct/dict containing 'bytes'
      - a struct/dict containing 'path'
      - raw bytes
    """
    if image_value is None:
        return None

    if isinstance(image_value, (bytes, bytearray, memoryview)):
        return bytes(image_value)

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")

        if image_bytes is not None:
            if isinstance(
                image_bytes,
                (bytes, bytearray, memoryview),
            ):
                return bytes(image_bytes)

        # If the Parquet representation contains only a path, try it as
        # a Hugging Face dataset-relative path.
        image_path = image_value.get("path")

        if image_path:
            return download_image_path(image_path)

    # DuckDB may expose STRUCT values with attribute access in some versions.
    try:
        image_bytes = image_value["bytes"]

        if image_bytes is not None:
            return bytes(image_bytes)
    except Exception:
        pass

    return None


def download_image_path(path):
    """
    Fallback for Parquet rows where the Image feature contains a path rather
    than embedded bytes.
    """
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(
            path,
            timeout=120,
        )
        response.raise_for_status()
        return response.content

    # The WikiArt Hugging Face dataset is expected to have embedded image
    # bytes in the converted Parquet. If not, fail clearly rather than
    # silently downloading the whole repository.
    raise RuntimeError(
        f"Image row contains a non-URL path instead of image bytes: {path}"
    )


def save_image(image_bytes, output_path):
    """
    Convert any supported source image into RGB JPEG, matching the original
    script's output behavior.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95,
        )


def download_with_duckdb(wanted, counts, dirs, per_style, parquet_urls):
    """
    Query the remote Parquet file directly using DuckDB.

    Only rows whose style is one of the requested ClassLabel IDs are selected.
    The style column is filtered before the image column is materialized.
    """
    try:
        import duckdb
    except ImportError:
        print(
            "[ERROR] DuckDB is not installed.\n"
            "Install it with:\n"
            "    pip install duckdb"
        )
        return False

    requested_ids = list(wanted.values())

    # Map integer class IDs back to the requested CLI style names.
    id_to_style = {
        style_id: style
        for style, style_id in wanted.items()
    }

    remaining = {
        style
        for style, count in counts.items()
        if count < per_style
    }

    if not remaining:
        return True

    # DuckDB accepts HTTP(S) Parquet URLs directly. The parquet_scan/read_parquet
    # reader supports predicate pushdown, so we don't load the complete table
    # into Python memory.
    con = duckdb.connect()

    try:
        for parquet_url in parquet_urls:
            if not remaining:
                break

            print(
                "\nReading Parquet directly with DuckDB..."
            )
            print(
                "Filtering for requested WikiArt styles before "
                "reading matching images..."
            )

            id_sql = ", ".join(str(int(x)) for x in requested_ids)

            query = f"""
                SELECT style, image
                FROM read_parquet('{parquet_url}')
                WHERE style IN ({id_sql})
            """

            try:
                cursor = con.execute(query)

                while True:
                    batch = cursor.fetchmany(32)

                    if not batch:
                        break

                    for style_id, image_value in batch:
                        if style_id not in id_to_style:
                            continue

                        style = id_to_style[style_id]

                        if style not in remaining:
                            continue

                        image_bytes = extract_image_bytes(
                            image_value
                        )

                        if image_bytes is None:
                            print(
                                f"\n[WARN] Could not extract image "
                                f"bytes for a {style} row."
                            )
                            continue

                        filename = (
                            f"{style}_{counts[style]:04d}.jpg"
                        )

                        output_path = dirs[style] / filename

                        try:
                            save_image(
                                image_bytes,
                                output_path,
                            )
                        except Exception as exc:
                            print(
                                f"\n[WARN] Could not decode/save "
                                f"{style} image: {exc}"
                            )
                            continue

                        counts[style] += 1

                        print(
                            f"  [{style}] "
                            f"{counts[style]}/{per_style}",
                            end="\r",
                            flush=True,
                        )

                        if counts[style] >= per_style:
                            remaining.remove(style)

                            print(
                                f"\n  [DONE] {style}: "
                                f"{counts[style]} images"
                            )

                            if not remaining:
                                break

                    if not remaining:
                        break

            except Exception as exc:
                print(
                    f"\n[WARN] Could not query Parquet file:\n"
                    f"{exc}"
                )

                # Try the next parquet shard, if one exists.
                continue

    finally:
        con.close()

    print()

    return not remaining


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        default="data/wikiart",
    )

    parser.add_argument(
        "--styles",
        nargs="+",
        default=["expressionism", "cubism"],
        help=(
            "Style names to pull (case-insensitive match against the "
            "dataset's 'style' label). The first one is used to TRAIN "
            "the Fast Neural Style Transfer network "
            "(see CFG['style_name_for_training'] in the notebook); "
            "the rest are extra styles for qualitative/quantitative "
            "comparison with AdaIN and StyTr^2, which don't need "
            "per-style training."
        ),
    )

    parser.add_argument(
        "--per-style",
        type=int,
        default=100,
        help="Number of images to save per style.",
    )

    args = parser.parse_args()

    if args.per_style <= 0:
        raise SystemExit("--per-style must be greater than 0.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_to_id = get_style_mapping()
    wanted = match_styles(args.styles, name_to_id)

    counts = {}
    dirs = {}

    for style in wanted:
        directory = (
            out_dir / style.lower().replace(" ", "_")
        )

        directory.mkdir(parents=True, exist_ok=True)
        dirs[style] = directory

        counts[style] = min(
            existing_count(directory),
            args.per_style,
        )

    print(
        f"Collecting {args.per_style} images per style for: "
        f"{list(wanted.keys())}"
    )

    for style, count in counts.items():
        if count:
            print(
                f"  [RESUME] {style}: "
                f"{count}/{args.per_style} already present"
            )

    if all(count >= args.per_style for count in counts.values()):
        print("All requested WikiArt images are already present.")
        return

    parquet_urls = get_parquet_files()

    complete = download_with_duckdb(
        wanted=wanted,
        counts=counts,
        dirs=dirs,
        per_style=args.per_style,
        parquet_urls=parquet_urls,
    )

    print()

    for style, count in counts.items():
        print(
            f"  {style}: saved {count} images -> {dirs[style]}"
        )

    if not complete:
        print(
            "\n[WARN] The requested number of images was not "
            "collected for every style."
        )
        print(
            "[INFO] No /filter or /rows fallback was used."
        )

    if any(count == 0 for count in counts.values()):
        print(
            "[WARN] Some styles got 0 images -- check the style "
            "name spelling against the dataset's style labels."
        )


if __name__ == "__main__":
    main()
