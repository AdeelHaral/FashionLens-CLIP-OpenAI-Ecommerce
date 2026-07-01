import os
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from recommendation_pipeline import get_device, load_catalog, load_model, search_products

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="FashionLens AI", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_models():
    return load_model(get_device())


@st.cache_data
def get_catalog():
    return load_catalog()


def download_image(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


model, processor, device = load_models()
df = get_catalog()


def product_card(result: pd.Series, index: int) -> None:
    article_id = str(result.get("article_id", ""))
    score = float(result.get("score", 0.0))
    product = df[df["article_id"] == article_id]

    if product.empty:
        name = "Unknown Product"
        department = "Fashion"
        image_url = ""
    else:
        row = product.iloc[0]
        name = str(row.get("prod_name", "Unknown Product"))
        department = str(row.get("department_name", "Fashion"))
        image_url = str(row.get("image_url", ""))

    metadata_line = ""
    if product.empty:
        metadata_line = "Rich metadata unavailable"
    else:
        row = product.iloc[0]
        group = str(row.get("product_group_name", "")).strip()
        section = str(row.get("section_name", "")).strip()
        color = str(row.get("perceived_colour_value_name", "")).strip()
        details = str(row.get("detail_desc", "")).strip()
        parts = [p for p in [group, section, color] if p]
        metadata_line = " · ".join(parts) if parts else "Semantic fashion product"
        if details:
            metadata_line = f"{metadata_line} · {details[:70]}{'...' if len(details) > 70 else ''}" 

    st.markdown(
        f"""
        <div style='border:1px solid #f3e8ff; border-radius:18px; padding:14px; background:#ffffff; box-shadow:0 6px 18px rgba(15,23,42,0.04);'>
            <div style='font-size:0.74rem; color:#7c3aed; margin-bottom:8px;'>#{index + 1} · {department}</div>
            <div style='font-size:1.03rem; font-weight:700; color:#111827; margin-bottom:6px;'>{name}</div>
            <div style='font-size:0.84rem; color:#4b5563; margin-bottom:6px;'>{metadata_line}</div>
            <div style='font-size:0.84rem; color:#4b5563;'>Semantic similarity {score:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image_url and image_url != "nan":
        try:
            st.image(download_image(image_url), use_container_width=True)
        except Exception:
            st.info("Image unavailable")
    else:
        st.info("No image available")

    st.button("Add to bag", key=f"bag_{article_id}_{index}", use_container_width=True)


def main() -> None:
    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #fff7ed, #ffffff); padding: 24px 24px; border-radius: 22px; margin-bottom: 18px; border: 1px solid #fed7aa;'>
            <h1 style='color: #111827; margin: 0;'>🛍️ FashionLens</h1>
            <p style='color: #4b5563; margin: 8px 0 0 0;'>A polished CLIP-based shopping experience for semantic fashion search and discovery.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Discover your next look")
    st.caption("Describe a style, upload a reference image, or combine both for a richer shopping experience.")

    with st.expander("How it works", expanded=False):
        st.write(
            "The app converts text and image queries into CLIP embeddings, then compares them with product vectors stored in a local Qdrant index. "
            "This is the same idea used in modern ecommerce search systems: retrieve items by meaning rather than exact keyword matching."
        )

    demo_queries = [
        "blue floral summer dress",
        "minimal beige blazer",
        "white sneakers streetwear",
        "black tailored trousers",
        "soft knit sweater",
    ]
    cols = st.columns(5)
    for col, query in zip(cols, demo_queries):
        if col.button(query, use_container_width=True):
            st.session_state["demo_query"] = query

    with st.sidebar:
        st.header("Filters")
        department_filter = st.selectbox("Department", options=["All"] + sorted(df["department_name"].dropna().unique().tolist()))
        top_k = st.slider("Results to show", min_value=6, max_value=24, value=12, step=1)
        st.caption("The current catalog is a balanced subset of fashion departments from the H&M dataset.")

    st.subheader("Search")
    search_mode = st.segmented_control("Search mode", options=["Text", "Image", "Hybrid"], default="Text")

    search_query = ""
    uploaded_image = None

    if "demo_query" in st.session_state and st.session_state["demo_query"]:
        initial_query = st.session_state["demo_query"]
    else:
        initial_query = ""

    if search_mode == "Text":
        search_query = st.text_input(
            "Search by style, product name, or category",
            placeholder="e.g. blue floral dress, white sneakers",
            value=initial_query,
        )
    elif search_mode == "Image":
        uploaded_image = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            image_bytes = BytesIO(uploaded_image.getvalue())
            st.image(Image.open(image_bytes), use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Optional text prompt", placeholder="e.g. minimalist black blazer", key="hybrid_text")
        with col2:
            uploaded_image = st.file_uploader("Optional product image", type=["png", "jpg", "jpeg"], key="hybrid_image")
        if uploaded_image is not None:
            image_bytes = BytesIO(uploaded_image.getvalue())
            st.image(Image.open(image_bytes), use_container_width=True)

    if search_query or uploaded_image is not None:
        if uploaded_image is not None:
            image_obj = Image.open(BytesIO(uploaded_image.getvalue())).convert("RGB")
        else:
            image_obj = None

        results = search_products(
            query=search_query if search_query else None,
            image=image_obj,
            top_k=top_k,
            model=model,
            processor=processor,
            device=device,
        )

        if not results.empty:
            st.markdown("---")
            filtered_results = []
            for _, res in results.iterrows():
                article_id = str(res.get("article_id", ""))
                row = df[df["article_id"] == article_id]
                if department_filter != "All" and not row.empty:
                    if str(row.iloc[0].get("department_name", "")) != department_filter:
                        continue
                filtered_results.append(res)

            if filtered_results:
                st.markdown("### Recommended products")
                for i in range(0, len(filtered_results), 3):
                    cols = st.columns(3)
                    for j, res in enumerate(filtered_results[i:i + 3]):
                        with cols[j]:
                            product_card(res, i + j)
            else:
                st.info("No products matched the selected department filter.")
        else:
            st.warning("No matching products were found. Try a different description or upload another image.")
    else:
        st.info("Enter a search prompt or upload an image to discover products.")


if __name__ == "__main__":
    main()
