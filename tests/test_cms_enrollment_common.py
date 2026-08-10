from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cms_enrollment_common import (  # noqa: E402
    extract_resource_downloads,
    filter_resource_downloads,
    resolve_catalog_dataset,
    select_latest_distribution,
)


def test_resolve_catalog_dataset_matches_landing_page_without_data_suffix():
    catalog_payload = {
        "dataset": [
            {
                "title": "Other Dataset",
                "landingPage": "https://data.cms.gov/other/path",
                "modified": "2026-01-01",
            },
            {
                "title": "Medicare Fee-For-Service  Public Provider Enrollment",
                "landingPage": (
                    "https://data.cms.gov/provider-characteristics/"
                    "medicare-provider-supplier-enrollment/"
                    "medicare-fee-for-service-public-provider-enrollment"
                ),
                "modified": "2026-04-20",
            },
        ]
    }

    dataset = resolve_catalog_dataset(
        catalog_payload,
        landing_page=(
            "https://data.cms.gov/provider-characteristics/"
            "medicare-provider-supplier-enrollment/"
            "medicare-fee-for-service-public-provider-enrollment/data"
        ),
    )

    assert dataset is not None
    assert dataset["title"] == "Medicare Fee-For-Service  Public Provider Enrollment"


def test_select_latest_distribution_prefers_latest_resources_api_entry():
    dataset_payload = {
        "distribution": [
            {
                "title": "Older release",
                "modified": "2026-01-20",
                "resourcesAPI": "https://example.test/older",
            },
            {
                "title": "Latest release",
                "description": "latest",
                "modified": "2026-04-20",
                "resourcesAPI": "https://example.test/latest",
            },
        ]
    }

    distribution = select_latest_distribution(dataset_payload)

    assert distribution is not None
    assert distribution["resourcesAPI"] == "https://example.test/latest"


def test_filter_resource_downloads_keeps_only_enrollment_related_csv_assets():
    resource_payload = {
        "data": [
            {
                "name": "Main enrollment",
                "downloadURL": "https://example.test/PPEF_Enrollment_Extract_2026.04.01.csv",
            },
            {
                "name": "Data dictionary",
                "downloadURL": "https://example.test/PPEF_Data_Dictionary.pdf",
            },
            {
                "name": "Methodology",
                "downloadURL": "https://example.test/Methodology.pdf",
            },
            {
                "name": "Additional NPIs",
                "downloadURL": "https://example.test/PPEF_Additional_NPIs_2026.04.01.csv",
            },
            {
                "name": "Reassignment",
                "downloadURL": "https://example.test/PPEF_Reassignment_Extract_2026.04.01.csv",
            },
            {
                "name": "Address Sub-File Q1 2026",
                "downloadURL": "https://example.test/current_addresses.csv",
            },
            {
                "name": "Historical",
                "downloadURL": (
                    "https://example.test/"
                    "PECOS_Public_Provider_Main_Historical_Files_CY2021-CY2022_2025.08.08.zip"
                ),
            },
        ]
    }

    filtered = filter_resource_downloads(
        extract_resource_downloads(resource_payload),
        [r"ENROLL", r"REASSIGN", r"PRACTICE", r"LOCATION"],
    )

    assert [asset.filename for asset in filtered] == [
        "PPEF_Enrollment_Extract_2026.04.01.csv",
        "PPEF_Reassignment_Extract_2026.04.01.csv",
        "current_addresses.csv",
    ]


class CmsEnrollmentCommonTestCase(unittest.TestCase):
    def test_catalog_landing_page_resolution(self):
        test_resolve_catalog_dataset_matches_landing_page_without_data_suffix()

    def test_latest_distribution_resolution(self):
        test_select_latest_distribution_prefers_latest_resources_api_entry()

    def test_resource_asset_filtering(self):
        test_filter_resource_downloads_keeps_only_enrollment_related_csv_assets()


if __name__ == "__main__":
    unittest.main()
