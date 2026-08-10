"""Unit and loader tests for the unified CMS provider API."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.api import STORE, app, format_provider
from src.data_store import DataStore, prepare_organization_search_columns, prepare_provider_search_columns


def provider_fixture() -> pd.DataFrame:
    rows = [
        {
            "baseline_view": "current_active_primary",
            "provider_id": "provider-1",
            "npi": "1234567890",
            "provider_full_name": "Jane Q Doe",
            "first_name": "Jane",
            "middle_name": "Q",
            "last_name": "Doe",
            "taxonomy_code_primary": "207RC0000X",
            "taxonomy_desc_primary": "Cardiovascular Disease",
            "taxonomy_codes_all": "207RC0000X | 207R00000X",
            "taxonomy_descs_all": "Cardiovascular Disease | Internal Medicine",
            "specialty_normalized": "Cardiology",
            "specialty_group": "Cardiology",
            "provider_type_group": "Physician",
            "provider_practice_address_1": "10 Old Road",
            "provider_practice_city": "St Paul",
            "provider_practice_state": "MN",
            "provider_practice_zip5": "55101",
            "provider_practice_phone": "6515550100",
            "mapped_clinic_id": "clinic-1",
            "mapped_clinic_name": "North Clinic",
            "mapped_system_id": "system-1",
            "mapped_system_name": "North Health",
            "mapped_practice_address_1": "100 Main Street",
            "mapped_practice_city": "Minneapolis",
            "mapped_practice_state": "MN",
            "mapped_practice_zip5": "55401",
            "mapped_practice_phone": "6125550100",
            "mapping_confidence_tier": "high",
            "mapping_method": "reassignment",
            "mapping_score": 12.5,
            "mapping_margin_score": 5.0,
            "mapping_signal_count": 3,
            "relationship_source": "PPEF_REASSIGNMENT",
            "is_active_relationship": True,
            "cms_current_active_flag": True,
            "cms_currentness_tier": "current_active_primary",
            "cms_currentness_reason": "active_relationship_high_confidence_primary_mapping",
            "cms_active_as_of_date": "2026-05-05",
            "ppef_snapshot_date": "2026-05-05",
            "pecos_snapshot_date": "2026-05-05",
            "npi_registry_snapshot_date": "2026-05-05",
            "provider_identity_source": "PPEF+NPI",
            "specialty_source": "NPI Registry",
            "phone_source": "NPI Registry",
            "address_source": "NPI Registry",
            "clinic_source": "PECOS",
            "system_source": "PECOS",
        },
        {
            "baseline_view": "current_active_primary",
            "provider_id": "provider-2",
            "npi": "0987654321",
            "provider_full_name": "Alex Smith",
            "first_name": "Alex",
            "last_name": "Smith",
            "taxonomy_code_primary": "207Q00000X",
            "taxonomy_desc_primary": "Family Medicine",
            "specialty_normalized": "Family Medicine",
            "specialty_group": "Primary Care",
            "provider_practice_city": "Duluth",
            "provider_practice_state": "MN",
            "provider_practice_zip5": "55802",
            "mapped_clinic_id": "clinic-2",
            "mapped_clinic_name": "Lake Clinic",
            "mapped_system_id": "system-2",
            "mapped_system_name": "Lake Health",
            "mapped_practice_city": "Duluth",
            "mapped_practice_state": "MN",
            "mapped_practice_zip5": "55802",
            "cms_current_active_flag": True,
        },
    ]
    return prepare_provider_search_columns(pd.DataFrame(rows))


def clinic_fixture() -> pd.DataFrame:
    return prepare_organization_search_columns(
        pd.DataFrame(
            [
                {
                    "clinic_id": "clinic-1",
                    "clinic_name": "North Clinic",
                    "clinic_display_name": "North Clinic",
                    "system_id": "system-1",
                    "system_display_name": "North Health",
                    "practice_address_1": "100 Main Street",
                    "practice_city": "Minneapolis",
                    "practice_state": "MN",
                    "practice_zip5": "55401",
                    "provider_count": 25,
                    "is_hospital": False,
                }
            ]
        ),
        "clinic",
    )


def system_fixture() -> pd.DataFrame:
    return prepare_organization_search_columns(
        pd.DataFrame(
            [
                {
                    "system_id": "system-1",
                    "system_name": "North Health",
                    "system_display_name": "North Health",
                    "practice_address_1": "100 Main Street",
                    "practice_city": "Minneapolis",
                    "practice_state": "MN",
                    "practice_zip5": "55401",
                    "provider_count_entity": 125,
                    "is_hospital": True,
                },
                {
                    "system_id": "system-2",
                    "system_name": "Lake Health",
                    "system_display_name": "Lake Health",
                    "practice_city": "Duluth",
                    "practice_state": "MN",
                    "practice_zip5": "55802",
                    "provider_count_entity": 50,
                    "is_hospital": False,
                },
            ]
        ),
        "system",
    )


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()
        STORE.providers = provider_fixture()
        STORE.clinics = clinic_fixture()
        STORE.systems = system_fixture()
        STORE.source = "cms"
        STORE.baseline_view = "current_active_primary"
        STORE.metadata = {"ppef_snapshot_date": "2026-05-05"}

    def tearDown(self):
        STORE.clear()

    def test_health_reports_cms_components(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["data_loaded"])
        self.assertEqual(payload["data_source"], "cms")
        self.assertEqual(payload["baseline_view"], "current_active_primary")
        self.assertEqual(payload["total_providers"], 2)
        self.assertEqual(payload["total_clinics"], 1)
        self.assertEqual(payload["total_systems"], 2)
        self.assertEqual(payload["total_hospitals"], 1)

    def test_provider_lookup_returns_compatibility_and_cms_fields(self):
        response = self.client.get("/api/providers/1234567890")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["provider_name"], "Jane Q Doe")
        self.assertEqual(payload["organization_name"], "North Clinic")
        self.assertEqual(payload["primary_specialty"], "Cardiology")
        self.assertEqual(payload["affiliation"]["system"]["name"], "North Health")
        self.assertEqual(payload["affiliation"]["practice_location"]["postal_code"], "55401")
        self.assertTrue(payload["currentness"]["active"])

    def test_invalid_npi_is_rejected(self):
        response = self.client.get("/api/providers/not-an-npi")
        self.assertEqual(response.status_code, 400)

    def test_specialty_search_uses_normalized_specialty(self):
        response = self.client.get("/api/providers/search/specialty?specialty=Cardiology")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["npi"], "1234567890")

    def test_location_search_can_use_affiliation_location(self):
        response = self.client.get(
            "/api/providers/search/location?city=Minneapolis&state=MN&location_source=affiliation"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 1)

        response = self.client.get(
            "/api/providers/search/location?city=Minneapolis&state=MN&location_source=provider"
        )
        self.assertEqual(response.get_json()["count"], 0)

    def test_provider_name_filters_by_system(self):
        response = self.client.get("/api/providers/search/name?name=Jane&hospital=North%20Health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 1)

    def test_clinic_and_system_routes(self):
        clinic = self.client.get("/api/clinics/search/name?clinic=North")
        system = self.client.get("/api/systems/search/name?system=Lake")
        hospital = self.client.get("/api/hospitals/search/name?hospital=North")
        non_hospital = self.client.get("/api/hospitals/search/name?hospital=Lake")
        self.assertEqual(clinic.get_json()["results"][0]["entity_type"], "clinic")
        self.assertEqual(system.get_json()["results"][0]["entity_type"], "system")
        self.assertEqual(hospital.get_json()["count"], 1)
        self.assertEqual(non_hospital.get_json()["count"], 0)

    def test_invalid_limit_returns_400(self):
        response = self.client.get("/api/providers/search/name?name=Jane&limit=bad")
        self.assertEqual(response.status_code, 400)

    def test_data_not_loaded_returns_503(self):
        STORE.clear()
        response = self.client.get("/api/providers/search/name?name=Jane")
        self.assertEqual(response.status_code, 503)


class DataStoreLoaderTestCase(unittest.TestCase):
    def test_cms_loader_filters_view_and_loads_rollups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            processed = data_dir / "processed_data"
            processed.mkdir()
            providers = pd.concat(
                [
                    provider_fixture(),
                    provider_fixture().assign(
                        baseline_view="all_mapped",
                        provider_id=lambda frame: frame["provider_id"] + "-all",
                        npi=["1111111111", "2222222222"],
                    ),
                ],
                ignore_index=True,
            )
            providers.to_parquet(
                processed / "ppef_individuals_comparison_baseline_20260505.parquet", index=False
            )
            clinic_fixture().drop(columns=[column for column in clinic_fixture().columns if column.startswith("_")]).to_parquet(
                processed / "pecos_clinic_rollup_20260505.parquet", index=False
            )
            system_fixture().drop(columns=[column for column in system_fixture().columns if column.startswith("_")]).to_parquet(
                processed / "pecos_system_rollup_20260505.parquet", index=False
            )
            store = DataStore()
            environment = {
                "API_DATA_SOURCE": "cms",
                "CMS_BASELINE_VIEW": "current_active_primary",
                "ALLOW_LEGACY_NPI_FALLBACK": "false",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertTrue(store.load(data_dir))
            self.assertEqual(len(store.providers), 2)
            self.assertEqual(len(store.clinics), 1)
            self.assertEqual(len(store.systems), 2)
            self.assertEqual(store.source, "cms")

    def test_format_provider_handles_missing_optional_fields(self):
        row = provider_fixture().iloc[1]
        payload = format_provider(row)
        self.assertEqual(payload["npi"], "0987654321")
        self.assertIsNone(payload["mapping"]["score"])


if __name__ == "__main__":
    unittest.main()
