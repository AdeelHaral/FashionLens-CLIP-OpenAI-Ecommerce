from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
PRODUCTS_PATH = DATA_DIR / "products.parquet"
QDRANT_DIR = DATA_DIR / "qdrant_db"

EMBED_DIR = DATA_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "fashionlens_products")
IMAGE_BATCH_SIZE = 16
TEXT_BATCH_SIZE = 64


def get_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    try:
        device = torch.device("cuda")
        _ = torch.zeros(1, device=device)
        return "cuda"
    except Exception:
        print("CUDA is unavailable for this PyTorch build; falling back to CPU.")
        return "cpu"


def load_model(device: Optional[str] = None) -> tuple[CLIPModel, CLIPProcessor, str]:
    device = device or get_device()
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    except Exception:
        fallback = "openai/clip-vit-base-patch32"
        print(f"Model {MODEL_NAME} unavailable, falling back to {fallback}")
        model = CLIPModel.from_pretrained(fallback)
        processor = CLIPProcessor.from_pretrained(fallback)

    if device == "cuda":
        model = model.half()
    model = model.to(device).eval()
    return model, processor, device


def load_catalog() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {PRODUCTS_PATH}")

    df = pd.read_parquet(PRODUCTS_PATH)
    for col in [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "section_name",
        "detail_desc",
        "text_to_embed",
        "image_url",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""
    df["article_id"] = df["article_id"].astype(str)
    return df.reset_index(drop=True)


def _build_text_description(row: pd.Series) -> str:
    name = str(row.get("prod_name", "")).strip()
    product_type = str(row.get("product_type_name", "")).strip()
    product_group = str(row.get("product_group_name", "")).strip()
    department = str(row.get("department_name", "")).strip()
    section = str(row.get("section_name", "")).strip()
    color_value = str(row.get("perceived_colour_value_name", "")).strip()
    color_master = str(row.get("perceived_colour_master_name", "")).strip()
    detail_desc = str(row.get("detail_desc", "")).strip()
    text_to_embed = str(row.get("text_to_embed", "")).strip()

    parts = []
    for value in [name, product_type, product_group, department, section, color_value, color_master]:
        if value:
            parts.append(value)

    rich_text = detail_desc or text_to_embed or ""
    if rich_text:
        rich_text = " ".join(rich_text.split())
        if len(rich_text) > 220:
            rich_text = rich_text[:220] + "..."
        parts.append(rich_text)

    description = " | ".join(parts)
    return description or "fashion ecommerce product"


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def _encode_text_batch(texts: list[str], processor: CLIPProcessor, model: CLIPModel, device: str) -> np.ndarray:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.text_model(**inputs)
        pooled = out.pooler_output
        feats = model.text_projection(pooled)
    return _normalize(feats.float().cpu().numpy())


def _encode_image(image: Image.Image, processor: CLIPProcessor, model: CLIPModel, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.vision_model(**inputs)
        pooled = out.pooler_output
        feats = model.visual_projection(pooled)
    return _normalize(feats.float().cpu().numpy())


def _encode_image_path(path: Path, processor: CLIPProcessor, model: CLIPModel, device: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            return _encode_image(img.convert("RGB"), processor, model, device)
    except Exception:
        return None


def _combine_vectors(text_vec: np.ndarray, image_vec: np.ndarray | None, alpha: float = 0.6) -> np.ndarray:
    if image_vec is None:
        return _normalize(text_vec)
    combined = (alpha * text_vec) + ((1 - alpha) * image_vec)
    return _normalize(combined)


def _get_qdrant_client() -> QdrantClient:
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(QDRANT_DIR))
    except RuntimeError as exc:
        if "already accessed" in str(exc).lower():
            print("Removing stale Qdrant local state and retrying...")
            for child in QDRANT_DIR.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    import shutil
                    shutil.rmtree(child, ignore_errors=True)
            return QdrantClient(path=str(QDRANT_DIR))
        raise


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, force_rebuild: bool = False) -> None:
    collections = {item.name for item in client.get_collections().collections}
    if collection_name in collections and not force_rebuild:
        return

    if collection_name in collections and force_rebuild:
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def build_qdrant_index(
    df: pd.DataFrame | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force_rebuild: bool = False,
    model: CLIPModel | None = None,
    processor: CLIPProcessor | None = None,
    device: Optional[str] = None,
) -> dict:
    """Build a CLIP-based semantic index inside a local Qdrant store."""
    df = load_catalog() if df is None else df.copy()
    device = device or get_device()
    if model is None or processor is None:
        model, processor, device = load_model(device)

    client = _get_qdrant_client()
    dummy_text = ["fashion product"]
    dummy_vec = _encode_text_batch(dummy_text, processor, model, device)
    vector_size = int(dummy_vec.shape[-1])
    _ensure_collection(client, collection_name, vector_size, force_rebuild=force_rebuild)

    print(f"Indexing {len(df)} products into Qdrant collection {collection_name}...")
    for start in range(0, len(df), TEXT_BATCH_SIZE):
        batch = df.iloc[start : start + TEXT_BATCH_SIZE].copy()
        texts = [_build_text_description(row) for _, row in batch.iterrows()]
        text_vectors = _encode_text_batch(texts, processor, model, device)

        points = []
        for idx, row in batch.iterrows():
            image_path = IMAGE_DIR / f"{row['article_id']}.jpg"
            image_vector = _encode_image_path(image_path, processor, model, device)
            text_vector = text_vectors[idx - start]
            if text_vector.ndim != 1:
                text_vector = np.asarray(text_vector).reshape(-1)
            if image_vector is not None and image_vector.ndim != 1:
                image_vector = np.asarray(image_vector).reshape(-1)
            combined_vector = _combine_vectors(text_vector, image_vector)
            payload = {
                "article_id": str(row["article_id"]),
                "prod_name": str(row.get("prod_name", "")),
                "product_type_name": str(row.get("product_type_name", "")),
                "product_group_name": str(row.get("product_group_name", "")),
                "department_name": str(row.get("department_name", "")),
                "section_name": str(row.get("section_name", "")),
                "perceived_colour_value_name": str(row.get("perceived_colour_value_name", "")),
                "perceived_colour_master_name": str(row.get("perceived_colour_master_name", "")),
                "detail_desc": str(row.get("detail_desc", "")),
                "text_to_embed": str(row.get("text_to_embed", "")),
                "image_url": str(row.get("image_url", "")),
                "text_description": texts[idx - start],
            }
            vector_payload = combined_vector.reshape(-1).tolist()
            points.append(
                qmodels.PointStruct(
                    id=int(row.get("catalog_index", idx)),
                    vector=vector_payload,
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"  Indexed {min(start + TEXT_BATCH_SIZE, len(df))}/{len(df)} products")

    return {
        "collection": collection_name,
        "catalog_size": len(df),
        "vector_size": vector_size,
        "qdrant_path": str(QDRANT_DIR),
        "device": device,
        "model": MODEL_NAME,
    }


def search_products(
    query: Optional[str] = None,
    image: Optional[Image.Image] = None,
    top_k: int = 12,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model: CLIPModel | None = None,
    processor: CLIPProcessor | None = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Search the indexed catalog using text, image, or hybrid CLIP embeddings."""
    if query is None and image is None:
        return pd.DataFrame(columns=["article_id", "prod_name", "department_name", "image_url", "score"])

    device = device or get_device()
    if model is None or processor is None:
        model, processor, device = load_model(device)

    client = _get_qdrant_client()
    if query is not None:
        query_vec = _encode_text_batch([query], processor, model, device)[0]
    else:
        query_vec = _encode_image(image, processor, model, device)[0]

    if query is not None and image is not None:
        image_vec = _encode_image(image, processor, model, device)[0]
        query_vec = _combine_vectors(query_vec, image_vec)

    response = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    )
    hits = response.points

    rows = []
    for hit in hits:
        payload = hit.payload or {}
        rows.append(
            {
                "article_id": str(payload.get("article_id", "")),
                "prod_name": str(payload.get("prod_name", "")),
                "product_group_name": str(payload.get("product_group_name", "")),
                "department_name": str(payload.get("department_name", "")),
                "section_name": str(payload.get("section_name", "")),
                "perceived_colour_value_name": str(payload.get("perceived_colour_value_name", "")),
                "perceived_colour_master_name": str(payload.get("perceived_colour_master_name", "")),
                "image_url": str(payload.get("image_url", "")),
                "score": float(hit.score),
            }
        )

    return pd.DataFrame(rows)


def recommend_by_text(
    query: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    top_k: int = 12,
) -> pd.DataFrame:
    return search_products(query=query, top_k=top_k, model=model, processor=processor, device=device)


def recommend_by_image(
    image: Image.Image,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    df: pd.DataFrame,
    top_k: int = 12,
) -> pd.DataFrame:
    return search_products(image=image, top_k=top_k, model=model, processor=processor, device=device)


def build_recommendation_index() -> dict:
    print("CWD:", os.getcwd())
    print("BASE_DIR:", BASE_DIR.resolve())
    model, processor, device = load_model()
    return build_qdrant_index(model=model, processor=processor, device=device)


if __name__ == "__main__":
    print(json.dumps(build_recommendation_index(), indent=2))