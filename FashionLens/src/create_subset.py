from pathlib import Path

import pandas as pd
from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 5000
DEFAULT_SELECTED_DEPARTMENTS = ["Dress", "Dresses", "Blouse", "Trouser", "Trousers", "Knitwear", "Sweater", "Cardigan"]
RICH_METADATA_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "section_name",
    "detail_desc",
    "text_to_embed",
    "department_name",
    "image_url",
]


def build_catalog_frame(raw_data) -> pd.DataFrame:
    """Create a catalog dataframe from the raw H&M dataset using the full rich metadata set."""
    available_columns = list(getattr(raw_data, "column_names", []))
    selected_columns = [col for col in RICH_METADATA_COLUMNS + ["article_id"] if col in available_columns]

    frame = {}
    for col in selected_columns:
        frame[col] = raw_data[col]

    for col in RICH_METADATA_COLUMNS:
        if col not in frame:
            frame[col] = [""] * len(raw_data)

    frame["article_id"] = [str(value) for value in raw_data["article_id"]]
    return pd.DataFrame(frame)


def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw H&M catalog into a compact ecommerce-friendly table with rich product metadata."""
    df = df.copy()
    for col in RICH_METADATA_COLUMNS + ["article_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    df["article_id"] = df["article_id"].astype(str)
    df["department_name"] = df["department_name"].replace({"": "Unknown"})
    df = df[df["prod_name"].str.strip() != ""].copy()
    df = df[df["image_url"].str.strip() != ""].copy()
    return df.reset_index(drop=True)


def select_products_subset(
    df: pd.DataFrame,
    selected_departments=None,
    selected_product_groups=None,
    target_size: int = TARGET_SIZE,
    max_per_department: int = 1000,
    max_product_groups: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced subset that focuses on a few selected fashion product groups and departments."""
    catalog = normalize_catalog(df)
    if selected_departments is None:
        selected_departments = DEFAULT_SELECTED_DEPARTMENTS

    if selected_product_groups is None:
        product_group_counts = catalog["product_group_name"].value_counts()
        selected_product_groups = product_group_counts.head(max_product_groups).index.tolist()
    selected_product_groups = [group for group in selected_product_groups if str(group).strip()]

    if selected_product_groups:
        catalog = catalog[catalog["product_group_name"].isin(selected_product_groups)].copy()

    available_departments = [dept for dept in selected_departments if dept in catalog["department_name"].values]
    if not available_departments:
        available_departments = ["Unknown"]

    balanced_chunks = []
    for dept in available_departments:
        dept_df = catalog[catalog["department_name"] == dept]
        if dept_df.empty:
            continue
        sample_size = min(max_per_department, len(dept_df))
        balanced_chunks.append(dept_df.sample(n=sample_size, random_state=random_state))

    if balanced_chunks:
        subset = pd.concat(balanced_chunks, ignore_index=True)
    else:
        subset = catalog.sample(n=min(target_size, len(catalog)), random_state=random_state)

    subset = subset.drop_duplicates(subset=["article_id"]).sample(n=min(target_size, len(subset)), random_state=random_state)
    subset = subset.reset_index(drop=True)
    subset["catalog_index"] = range(len(subset))
    return subset


def save_subset(subset: pd.DataFrame, output_parquet_path: Path | None = None, metadata_path: Path | None = None) -> None:
    """Persist the filtered catalog and a lightweight metadata file."""
    output_parquet_path = output_parquet_path or DATA_DIR / "products.parquet"
    metadata_path = metadata_path or DATA_DIR / "products_metadata.csv"

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output_parquet_path, index=False)

    metadata_columns = [
        "article_id",
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
    ]
    export_df = subset.copy()
    for col in metadata_columns:
        if col not in export_df.columns:
            export_df[col] = ""
    export_df[metadata_columns].to_csv(metadata_path, index=False)


def main() -> None:
    print("Loading H&M catalog...")
    raw = load_dataset("Qdrant/hm_ecommerce_products", split="train")

    catalog = build_catalog_frame(raw)

    subset = select_products_subset(
        catalog,
        selected_departments=DEFAULT_SELECTED_DEPARTMENTS,
        target_size=TARGET_SIZE,
    )
    save_subset(subset)

    print(f"Saved {len(subset)} balanced products")
    print(subset["department_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
