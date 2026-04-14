"""
Test script for NPI Registry API

This script demonstrates how to use all API endpoints.
Run the API server first: python src/api.py
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:5000"


def print_response(response: requests.Response, title: str):
    """Pretty print API response."""
    print("\n" + "=" * 80)
    print(f"TEST: {title}")
    print("=" * 80)
    print(f"URL: {response.url}")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse (formatted):")
        print(json.dumps(data, indent=2))
    else:
        print(f"\nError: {response.text}")

    print("=" * 80)


def test_health_check():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/api/health")
    print_response(response, "Health Check")
    return response.json() if response.status_code == 200 else None


def test_search_by_location():
    """Test searching providers by location."""
    # Test 1: Search with distance filter
    params = {
        "city": "Minneapolis",
        "state": "MN",
        "specialty": "Cardiology",
        "distance_miles": 30,
        "limit": 5,
    }
    response = requests.get(f"{BASE_URL}/api/providers/search/location", params=params)
    print_response(response, "Search by Location (with distance)")

    # Test 2: Search without distance filter (all in area)
    params = {
        "city": "Minneapolis",
        "state": "MN",
        "specialty": "Family Medicine",
        "limit": 5,
    }
    response = requests.get(f"{BASE_URL}/api/providers/search/location", params=params)
    print_response(response, "Search by Location (no distance filter)")


def test_search_by_specialty():
    """Test searching providers by specialty."""
    params = {"specialty": "Cardiology", "state": "MN", "limit": 5}
    response = requests.get(f"{BASE_URL}/api/providers/search/specialty", params=params)
    print_response(response, "Search by Specialty")

    return response.json() if response.status_code == 200 else None


def test_get_provider_by_npi(npi: str):
    """Test getting a specific provider by NPI."""
    response = requests.get(f"{BASE_URL}/api/providers/{npi}")
    print_response(response, f"Get Provider by NPI ({npi})")


def test_search_by_state():
    """Test searching providers by state."""
    params = {"limit": 5}
    response = requests.get(f"{BASE_URL}/api/providers/search/state/MN", params=params)
    print_response(response, "Search by State (MN)")


def test_search_by_postal_code():
    """Test searching providers by postal code."""
    params = {"limit": 5}
    response = requests.get(
        f"{BASE_URL}/api/providers/search/postal_code/55401", params=params
    )
    print_response(response, "Search by Postal Code (55401)")


def test_error_cases():
    """Test error handling."""
    # Test 1: Missing required parameters
    response = requests.get(f"{BASE_URL}/api/providers/search/location")
    print_response(response, "Error: Missing Parameters")

    # Test 2: Provider not found
    response = requests.get(f"{BASE_URL}/api/providers/9999999999")
    print_response(response, "Error: Provider Not Found")

    # Test 3: Invalid endpoint
    response = requests.get(f"{BASE_URL}/api/invalid")
    print_response(response, "Error: Invalid Endpoint")


def run_all_tests():
    """Run all API tests."""
    print("\n" + "🧪" * 40)
    print("NPI REGISTRY API - TEST SUITE")
    print("🧪" * 40)
    print("\nMake sure the API server is running: python src/api.py")
    print("\nStarting tests...\n")

    try:
        # Test 1: Health check
        health = test_health_check()

        if not health or not health.get("data_loaded"):
            print("\n❌ ERROR: Data not loaded. Please load data first.")
            return

        # Test 2: Search by location
        test_search_by_location()

        # Test 3: Search by specialty
        specialty_results = test_search_by_specialty()

        # Test 4: Get specific provider (if we have results)
        if specialty_results and specialty_results.get("results"):
            first_npi = specialty_results["results"][0]["npi"]
            test_get_provider_by_npi(first_npi)

        # Test 5: Search by state
        test_search_by_state()

        # Test 6: Search by postal code
        test_search_by_postal_code()

        # Test 7: Error cases
        test_error_cases()

        print("\n" + "✅" * 40)
        print("ALL TESTS COMPLETED")
        print("✅" * 40)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running: python src/api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()
