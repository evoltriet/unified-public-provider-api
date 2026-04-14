"""
NPI Registry - Automated Download and Split by Entity Type (NO MAPPING)

Features:
1. Download official NPPES file from CMS
2. Split by Entity Type Code (1=Individual, 2=Organization)
3. Save as separate optimized parquets
4. Keep all columns intact for processing step

Note: Organization mapping is handled in hash_based_processing.py as optional processing
"""

import requests
import pandas as pd
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
API_VERSION = "2.1"
DOWNLOAD_PAGE = "https://download.cms.gov/nppes/NPI_Files.html"


class NPIRegistryDownloader:
    """Automated NPI Registry downloader with entity type splitting."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.parquet_dir = self.data_dir / "parquet"
        self.parquet_dir.mkdir(exist_ok=True)

    def download_official_file(self, force_redownload=False):
        """
        Automatically download the official NPPES data file from CMS.

        Parameters:
        - force_redownload: If True, download even if file exists

        Returns:
        - Path to the downloaded CSV file
        """
        logger.info("=" * 80)
        logger.info("DOWNLOADING OFFICIAL NPPES DATA FILE")
        logger.info("=" * 80)

        possible_months = [
            datetime.now().strftime("%B_%Y"),
            (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%B_%Y"),
        ]

        zip_filename = None
        download_url = None

        for month in possible_months:
            test_url = (
                f"https://download.cms.gov/nppes/NPPES_Data_Dissemination_{month}.zip"
            )
            logger.info(f"Trying: {test_url}")
            try:
                response = requests.head(test_url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    download_url = test_url
                    zip_filename = f"NPPES_Data_Dissemination_{month}.zip"
                    logger.info(f"✓ Found file: {zip_filename}")
                    break
            except:
                continue

        if not download_url:
            logger.warning(
                "Could not find file using standard pattern. Scraping download page..."
            )
            download_url, zip_filename = self._scrape_download_link()

        zip_path = self.raw_dir / zip_filename

        if zip_path.exists() and not force_redownload:
            logger.info(f"File already exists: {zip_path}")
            logger.info(
                f"File size: {zip_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB"
            )
            logger.info("Use force_redownload=True to download again")
            return self._extract_csv_from_zip(zip_path)

        logger.info(f"\nDownloading from: {download_url}")
        logger.info(f"Destination: {zip_path}")
        logger.info(f"Expected size: ~7 GB (may take 15-60 minutes)")
        logger.info("=" * 80 + "\n")

        try:
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"\n✓ Download complete: {zip_path}")
            logger.info(
                f" File size: {zip_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB"
            )
            return self._extract_csv_from_zip(zip_path)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nAlternative: Download manually from:")
            logger.info("https://download.cms.gov/nppes/NPI_Files.html")
            raise

    def _scrape_download_link(self):
        """Scrape the CMS download page to find the latest file link."""
        try:
            logger.info("Fetching download page...")
            response = requests.get(DOWNLOAD_PAGE, timeout=30)
            response.raise_for_status()
            pattern = r'href="(https://download\.cms\.gov/nppes/NPPES_Data_Dissemination_\w+_\d{4}\.zip)"'
            matches = re.findall(pattern, response.text)
            if matches:
                download_url = matches[0]
                filename = download_url.split("/")[-1]
                logger.info(f"✓ Found download link: {filename}")
                return download_url, filename
            else:
                raise Exception("Could not find download link on page")
        except Exception as e:
            logger.error(f"Failed to scrape download page: {e}")
            raise

    def _extract_csv_from_zip(self, zip_path):
        """Extract the CSV file from the ZIP archive."""
        logger.info(f"\nExtracting CSV from ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_files = [
                f
                for f in zip_ref.namelist()
                if f.endswith(".csv") and "npidata" in f.lower()
            ]
            if not csv_files:
                raise Exception("No CSV file found in ZIP archive")

            csv_filename = csv_files[0]
            logger.info(f"Found CSV: {csv_filename}")
            csv_path = self.raw_dir / csv_filename

            if not csv_path.exists():
                logger.info("Extracting... (this may take 5-10 minutes)")
                zip_ref.extract(csv_filename, self.raw_dir)

            logger.info(f"✓ CSV extracted: {csv_path}")
            logger.info(
                f" File size: {csv_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB"
            )
            return csv_path

    def convert_csv_to_parquet(self, csv_file=None, chunk_size=50000):
        """
        Convert the massive NPI CSV file to Parquet format and SPLIT by Entity Type.

        Creates separate parquets for individuals and organizations.
        Organization mapping is handled separately in hash_based_processing.py
        """
        if csv_file is None:
            csv_files = list(self.raw_dir.glob("npidata*.csv"))
            if not csv_files:
                raise Exception(
                    "No CSV file found. Run download_official_file() first."
                )
            csv_file = csv_files[0]

        csv_file = Path(csv_file)

        logger.info("=" * 80)
        logger.info("CONVERTING CSV TO PARQUET AND SPLITTING BY ENTITY TYPE")
        logger.info("=" * 80)
        logger.info(f"Input: {csv_file}")
        logger.info(f"Chunk size: {chunk_size:,} rows")
        logger.info("Output will be 2 files:")
        logger.info("  - npi_individuals_YYYYMMDD.parquet")
        logger.info("  - npi_organizations_YYYYMMDD.parquet")
        logger.info("")
        logger.info(
            "Note: Organization mapping is optional in hash_based_processing.py"
        )
        logger.info("This will take 30-60 minutes...")
        logger.info("=" * 80 + "\n")

        date_str = datetime.now().strftime("%Y%m%d")
        individuals_file = self.parquet_dir / f"npi_individuals_{date_str}.parquet"
        organizations_file = self.parquet_dir / f"npi_organizations_{date_str}.parquet"

        individuals_chunks = []
        organizations_chunks = []

        try:
            for i, chunk in enumerate(
                pd.read_csv(
                    csv_file,
                    chunksize=chunk_size,
                    dtype=str,
                    low_memory=False,
                    na_values=[""],
                    keep_default_na=True,
                )
            ):
                logger.info(f"Processing chunk {i + 1} ({len(chunk):,} records)...")

                # Clean the chunk
                chunk = self._clean_chunk(chunk)

                # Split by Entity Type Code (1=Individual, 2=Organization)
                individuals = chunk[chunk["Entity Type Code"].astype(str) == "1"].copy()
                organizations = chunk[
                    chunk["Entity Type Code"].astype(str) == "2"
                ].copy()

                if len(individuals) > 0:
                    individuals_chunks.append(individuals)
                    logger.info(f"  - {len(individuals):,} individuals")

                if len(organizations) > 0:
                    organizations_chunks.append(organizations)
                    logger.info(f"  - {len(organizations):,} organizations")

                del chunk

            # Combine chunks
            logger.info("\n" + "=" * 80)
            logger.info("COMBINING CHUNKS")
            logger.info("=" * 80)

            if individuals_chunks:
                logger.info("Combining individual provider chunks...")
                df_individuals = pd.concat(individuals_chunks, ignore_index=True)
                logger.info(f"Total individuals: {len(df_individuals):,}")
            else:
                df_individuals = pd.DataFrame()

            if organizations_chunks:
                logger.info("Combining organization provider chunks...")
                df_organizations = pd.concat(organizations_chunks, ignore_index=True)
                logger.info(f"Total organizations: {len(df_organizations):,}")
            else:
                df_organizations = pd.DataFrame()

            # Save files
            logger.info("\n" + "=" * 80)
            logger.info("SAVING PARQUET FILES")
            logger.info("=" * 80)

            if len(df_individuals) > 0:
                df_individuals.to_parquet(
                    individuals_file, compression="snappy", index=False
                )
                size_mb = individuals_file.stat().st_size / 1024 / 1024
                logger.info(
                    f"✓ Individuals: {individuals_file.name} ({size_mb:.2f} MB)"
                )

            if len(df_organizations) > 0:
                df_organizations.to_parquet(
                    organizations_file, compression="snappy", index=False
                )
                size_mb = organizations_file.stat().st_size / 1024 / 1024
                logger.info(
                    f"✓ Organizations: {organizations_file.name} ({size_mb:.2f} MB)"
                )

            logger.info("\n" + "=" * 80)
            logger.info("CONVERSION COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Individuals: {len(df_individuals):,} providers")
            logger.info(f"Organizations: {len(df_organizations):,} providers")
            logger.info("")
            logger.info("Next steps (optional):")
            logger.info("  1. Run: python scripts/hash_based_processing.py")
            logger.info("  2. Choose: entity_mapping mode to add organization names")
            logger.info("=" * 80)

            return individuals_file, organizations_file

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

    def _clean_chunk(self, df):
        """Clean and optimize a chunk of data."""
        logger.debug("Cleaning data types...")

        # Optimize categorical columns where applicable
        for col in df.columns:
            if df[col].dtype == "object":
                num_unique = df[col].nunique()
                num_rows = len(df)

                if (
                    num_unique > 0
                    and num_unique < num_rows * 0.05
                    and num_unique < 1000
                ):
                    try:
                        df[col] = df[col].astype("category")
                    except:
                        pass

        return df

    def full_download_and_convert(self, force_redownload=False):
        """
        One-step function: Download and convert to split Parquet files.
        """
        logger.info("\n" + "=" * 80)
        logger.info("FULL AUTOMATED NPI REGISTRY DOWNLOAD & SPLIT")
        logger.info("=" * 80)
        logger.info("This will:")
        logger.info("1. Download official NPPES file (~7 GB, 15-60 min)")
        logger.info("2. Extract CSV from ZIP (~15 GB)")
        logger.info("3. Split by Entity Type into 2 parquets (~700 MB total)")
        logger.info("=" * 80 + "\n")

        response = input("Press Enter to continue or Ctrl+C to cancel...")

        csv_file = self.download_official_file(force_redownload=force_redownload)
        individuals_file, organizations_file = self.convert_csv_to_parquet(csv_file)

        logger.info("\n" + "🎉" * 40)
        logger.info("SUCCESS! NPI registry split and saved locally!")
        logger.info("🎉" * 40)

        return individuals_file, organizations_file


def main():
    """Main function."""
    downloader = NPIRegistryDownloader(data_dir="data")

    print("\n" + "=" * 80)
    print("NPI REGISTRY - AUTOMATED DOWNLOAD WITH ENTITY TYPE SPLIT")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Download official NPPES file from CMS")
    print("2. Split into INDIVIDUALS and ORGANIZATIONS")
    print("3. Save as optimized parquet files")
    print("\nOrganization mapping is optional in hash_based_processing.py")
    print("\nOutput: Two parquet files (individuals + organizations)")
    print("Total time: 45-120 minutes")
    print("Disk space needed: ~25 GB temporarily, ~1 GB final")
    print("=" * 80 + "\n")

    choice = input(
        "Choose:\n1. Full download and convert (recommended)\n2. Just download ZIP\n3. Convert existing CSV\n\nEnter 1, 2, or 3: "
    )

    if choice == "1":
        downloader.full_download_and_convert()
    elif choice == "2":
        csv_file = downloader.download_official_file()
        print(f"\n✓ Download complete: {csv_file}")
        print("Run with option 3 to convert to Parquet")
    elif choice == "3":
        individuals_file, organizations_file = downloader.convert_csv_to_parquet()
        print(f"\n✓ Conversion complete:")
        print(f"  - {individuals_file}")
        print(f"  - {organizations_file}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess

        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm

    main()
