import logging
import polars as pl
from pathlib import Path
import sys
from polars.exceptions import ColumnNotFoundError

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_DIR = Path(__file__).parent / "cytodata_cache"
PLATEMAP_CACHE_FILE = CACHE_DIR / "platemap_cache.parquet"
OUTPUT_CSV_FILE = Path(__file__).parent / "available_compounds_report.csv"


def main():
    if not PLATEMAP_CACHE_FILE.exists():
        logger.error(f"Cache file not found: {PLATEMAP_CACHE_FILE}")
        logger.error("Please run one of the 'use_case_*.py' scripts at least once to build the cache.")
        return

    try:
        logger.info(f"Loading platemap data from {PLATEMAP_CACHE_FILE}...")
        df = pl.read_parquet(PLATEMAP_CACHE_FILE)
    except Exception as e:
        logger.error(f"Failed to read cache file: {e}")
        return

    logger.info("Analyzing available compounds...")

    try:
        compound_report = (
            df.group_by("Metadata_EOS")
            .agg(
                pl.col("Metadata_Concentration").unique().sort().alias("Available_Concentrations"),
                pl.col("Metadata_Plate").unique().sort().alias("Available_Plates"),
                pl.len().alias("Total_Well_Count")
            )
            .sort("Metadata_EOS")  # Sort by EOS ID
        )

        # Print a summary to the console *before* flattening for CSV
        logger.info("--- Summary of Available Compounds (Top 50) ---")
        with pl.Config(
                tbl_rows=50,
                tbl_cols=5,
                tbl_width_chars=120,
                fmt_str_lengths=50
        ):
            print(compound_report.head(50))

        logger.info("--------------------------------------------------")

        # --- THIS IS THE FIX ---
        # We must cast the elements *inside* the list[i64] to str *before* joining.
        compound_report_csv = compound_report.with_columns(
            pl.col("Available_Concentrations").list.eval(pl.element().cast(pl.String)).list.join(", ").alias(
                "Available_Concentrations_Str"),
            pl.col("Available_Plates").list.join(", ").alias("Available_Plates_Str")
        ).drop("Available_Concentrations", "Available_Plates")  # Drop the original list columns
        # --- END FIX ---

        # Save the *flattened* report to a CSV file
        compound_report_csv.write_csv(OUTPUT_CSV_FILE)
        logger.info(f"Successfully saved full report to: {OUTPUT_CSV_FILE}")

        logger.info(f"A full list of {len(compound_report)} unique compounds was saved to {OUTPUT_CSV_FILE}")

    except ColumnNotFoundError as e:
        logger.error(f"Missing a required column: {e}")
        logger.error("The cache file might be corrupted. Try clearing the cache and re-running a use case script.")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    main()