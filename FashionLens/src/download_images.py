import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PRODUCTS_PATH = DATA_DIR / "products.parquet"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 24
TIMEOUT = 15


def clear_downloaded_images(image_dir: Path | None = None) -> int:
    """Remove previously downloaded image files so the folder can be rebuilt cleanly."""
    image_dir = image_dir or IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def download_images(
    products_path: Path = PRODUCTS_PATH,
    image_dir: Path = IMAGE_DIR,
    max_workers: int = MAX_WORKERS,
    timeout: int = TIMEOUT,
    clear_existing: bool = True,
) -> tuple[int, int]:
    """Download local product images for the current catalog subset."""
    if not products_path.exists():
        raise FileNotFoundError(f"Catalog not found at {products_path}")

    df = pd.read_parquet(products_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        removed = clear_downloaded_images(image_dir)
        print(f"Removed {removed} existing files from {image_dir}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    lock = threading.Lock()
    success = 0
    failed = 0

    def download_row(row: dict) -> tuple[str, str, str | None]:
        article_id = str(row["article_id"])
        image_url = str(row.get("image_url", ""))
        save_path = image_dir / f"{article_id}.jpg"

        if save_path.exists():
            return "skipped", article_id, None

        if not image_url or image_url == "nan":
            return "failed", article_id, "missing url"

        try:
            response = session.get(image_url, timeout=timeout)
            response.raise_for_status()
            save_path.write_bytes(response.content)
            return "success", article_id, None
        except Exception as exc:
            return "failed", article_id, str(exc)

    rows = [row._asdict() for row in df.itertuples(index=False)]
    print(f"Downloading {len(rows)} images with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_row, row) for row in rows]
        for future in as_completed(futures):
            status, article_id, err = future.result()
            with lock:
                if status == "success":
                    success += 1
                    if success % 100 == 0:
                        print(f"Downloaded {success} images...")
                elif status == "failed":
                    failed += 1
                    print(f"Failed {article_id}: {err}")

    print(f"Done. Downloaded {success} images, failed {failed} images.")
    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images for the FashionLens subset")
    parser.add_argument("--skip-cleanup", action="store_true", help="Keep existing images in the folder")
    args = parser.parse_args()

    download_images(clear_existing=not args.skip_cleanup)


if __name__ == "__main__":
    main()