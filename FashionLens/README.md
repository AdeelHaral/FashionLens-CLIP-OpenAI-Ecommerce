# FashionLens

FashionLens is a compact CLIP-based fashion recommender built on the H&M ecommerce product dataset. It supports:

- text-to-product retrieval
- image-to-product retrieval
- hybrid text + image search
- rich metadata-aware semantic retrieval using product name, product group, color, section, detail description, and text-to-embed fields
- a polished Streamlit demo for ecommerce-style exploration

## Project structure

- src/create_subset.py: builds a balanced 5k-product subset from the H&M dataset
- src/download_images.py: downloads product images for the subset
- src/recommendation_pipeline.py: CLIP text/image encoding and similarity search
- src/app.py: Streamlit app UI

## Quick start

The pipeline is designed to work from the H&M dataset in the Qdrant catalog and to create a focused fashion subset that uses rich metadata for better semantic matching.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create the subset:
   ```bash
   python src/create_subset.py
   ```

3. Download images for the subset:
   ```bash
   python src/download_images.py
   ```

4. Download product images for the curated subset:
   ```bash
   python src/download_images.py
   ```

5. Build CLIP embeddings and the local Qdrant index:
   ```bash
   python src/recommendation_pipeline.py
   ```

6. Launch the app from the project root:
   ```bash
   python3 -m streamlit run src/app.py --server.headless true --server.port 8501
   ```

## Demo queries

Use these queries for your presentation or local testing:

- "blue floral summer dress"
- "minimal beige blazer"
- "white sneakers with a streetwear look"
- "black tailored trousers"
- "soft knit sweater"
- "red satin evening dress"

## Notes

- The subset builder focuses on a handful of product groups and departments to keep the demo compact and visually consistent.
- The embedding pipeline uses rich metadata such as product name, product type, product group, color, section, detail description, and text-to-embed fields to build stronger semantic representations.
- The app supports text, image, and hybrid search with a clean ecommerce-style interface.
