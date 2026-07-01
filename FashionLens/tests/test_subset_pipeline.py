import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.create_subset import build_catalog_frame, select_products_subset
from src.download_images import clear_downloaded_images


class SubsetPipelineTests(unittest.TestCase):
    def test_select_products_subset_balances_selected_departments(self):
        df = pd.DataFrame(
            [
                {"article_id": "1", "prod_name": "Blue dress", "department_name": "Dress", "image_url": "https://example.com/1.jpg"},
                {"article_id": "2", "prod_name": "Red dress", "department_name": "Dress", "image_url": "https://example.com/2.jpg"},
                {"article_id": "3", "prod_name": "Black blazer", "department_name": "Blouse", "image_url": "https://example.com/3.jpg"},
                {"article_id": "4", "prod_name": "Beige trousers", "department_name": "Trouser", "image_url": "https://example.com/4.jpg"},
                {"article_id": "5", "prod_name": "Soft knit", "department_name": "Knitwear", "image_url": "https://example.com/5.jpg"},
                {"article_id": "6", "prod_name": "Other item", "department_name": "Unknown", "image_url": "https://example.com/6.jpg"},
            ]
        )

        subset = select_products_subset(df, selected_departments=["Dress", "Blouse", "Trouser", "Knitwear"], target_size=4)

        self.assertEqual(len(subset), 4)
        self.assertTrue(set(subset["department_name"].unique()) <= {"Dress", "Blouse", "Trouser", "Knitwear"})
        self.assertTrue(subset["article_id"].is_unique)

    def test_build_catalog_frame_includes_rich_metadata(self):
        class DummyRaw:
            def __init__(self):
                self.column_names = [
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

            def __getitem__(self, item):
                if item == "article_id":
                    return ["1", "2"]
                if item == "prod_name":
                    return ["Blue dress", "Black blazer"]
                if item == "product_type_name":
                    return ["Dress", "Blazer"]
                if item == "product_group_name":
                    return ["Garment Upper body", "Garment Upper body"]
                if item == "perceived_colour_value_name":
                    return ["Blue", "Black"]
                if item == "perceived_colour_master_name":
                    return ["Blue", "Black"]
                if item == "department_name":
                    return ["Dress", "Blouse"]
                if item == "section_name":
                    return ["Womens Everyday Collection", "Mens Everyday Collection"]
                if item == "detail_desc":
                    return ["A blue dress", "A black blazer"]
                if item == "text_to_embed":
                    return ["blue summer dress", "black tailored blazer"]
                if item == "image_url":
                    return ["https://example.com/dress.jpg", "https://example.com/blazer.jpg"]
                raise KeyError(item)

        catalog = build_catalog_frame(DummyRaw())

        self.assertIn("product_group_name", catalog.columns)
        self.assertIn("detail_desc", catalog.columns)
        self.assertIn("text_to_embed", catalog.columns)
        self.assertEqual(catalog.loc[0, "detail_desc"], "A blue dress")

    def test_clear_downloaded_images_removes_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir)
            (image_dir / "one.jpg").write_bytes(b"x")
            (image_dir / "two.jpg").write_bytes(b"y")

            clear_downloaded_images(image_dir)

            self.assertFalse(any(image_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
