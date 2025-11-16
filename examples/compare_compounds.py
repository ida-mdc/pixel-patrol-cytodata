import logging
import sys
from pathlib import Path

import polars as pl
from pixel_patrol_base.api import (
    export_project, show_report, import_project,
    # show_report is no longer needed here
)
from pixel_patrol_cytodata.cytodata_project import CytoDataProject, CytoDataSettings

# --- Setup Logging ---
# Configure logging to show info level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration ---
S3_DATA_ROOT = "https://cellpainting-gallery.s3.amazonaws.com/cpg0036-EU-OS-bioactives"
SAMPLE_PER_GROUP = 50
PROJECT_NAME = "CytoData_Compare_Compounds"
# These are the folders to scan for 'load_data.csv' files
MANIFEST_FOLDERS_TO_SCAN = [
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0036-EU-OS-bioactives/FMP/workspace/load_data_csv/",
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0036-EU-OS-bioactives/IMTM/workspace/load_data_csv/",
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0036-EU-OS-bioactives/MEDINA/workspace/load_data_csv/",
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0036-EU-OS-bioactives/USC/workspace/load_data_csv/",
]

Idarubicin_ID = "EOS101077"
Nocodazole_ID = "EOS100913"
DMSO_ID = "EOS100001"


def main():
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)
    exported_project_path = output_directory / f"{PROJECT_NAME}.zip"

    if not exported_project_path.exists():

        project = CytoDataProject(
            name=PROJECT_NAME,
            base_dir=S3_DATA_ROOT,
            loader="bioio-s3"
        )

        project.add_paths(MANIFEST_FOLDERS_TO_SCAN)

        filters = {
            f"{Idarubicin_ID}_FMP": (
                    (pl.col("Metadata_EOS") == Idarubicin_ID)
                    & (pl.col("Metadata_Provider") == "FMP")
            ),
            f"{Nocodazole_ID}_FMP": (
                    (pl.col("Metadata_EOS") == Nocodazole_ID)
                    & (pl.col("Metadata_Provider") == "FMP")
            ),
            f"{DMSO_ID}_FMP": (
                    (pl.col("Metadata_EOS") == DMSO_ID)
                    & (pl.col("Metadata_Provider") == "FMP")
            ),
            f"{Idarubicin_ID}_IMTM": (
                    (pl.col("Metadata_EOS") == Idarubicin_ID)
                    & (pl.col("Metadata_Provider") == "IMTM")
            ),
            f"{Nocodazole_ID}_IMTM": (
                    (pl.col("Metadata_EOS") == Nocodazole_ID)
                    & (pl.col("Metadata_Provider") == "IMTM")
            ),
            f"{DMSO_ID}_IMTM": (
                    (pl.col("Metadata_EOS") == DMSO_ID)
                    & (pl.col("Metadata_Provider") == "IMTM")
            ),
            f"{Idarubicin_ID}_MEDINA": (
                    (pl.col("Metadata_EOS") == Idarubicin_ID)
                    & (pl.col("Metadata_Provider") == "MEDINA")
            ),
            f"{Nocodazole_ID}_MEDINA": (
                    (pl.col("Metadata_EOS") == Nocodazole_ID)
                    & (pl.col("Metadata_Provider") == "MEDINA")
            ),
            f"{DMSO_ID}_MEDINA": (
                    (pl.col("Metadata_EOS") == DMSO_ID)
                    & (pl.col("Metadata_Provider") == "MEDINA")
            ),
            f"{Idarubicin_ID}_USC": (
                    (pl.col("Metadata_EOS") == Idarubicin_ID)
                    & (pl.col("Metadata_Provider") == "USC")
            ),
            f"{Nocodazole_ID}_USC": (
                    (pl.col("Metadata_EOS") == Nocodazole_ID)
                    & (pl.col("Metadata_Provider") == "USC")
            ),
            f"{DMSO_ID}_USC": (
                    (pl.col("Metadata_EOS") == DMSO_ID)
                    & (pl.col("Metadata_Provider") == "USC")
            )
        }

        settings = CytoDataSettings(
            cmap="viridis",
            selected_file_extensions="all",  # This setting isn't used by this project type
            comparison_filters=filters,
            sample_per_group=SAMPLE_PER_GROUP
        )
        project.set_settings(settings)

        logger.info("Starting project processing... This will now run the deep processing loop.")
        project.process_records()

        if project.get_records_df() is None or project.get_records_df().is_empty():
            logger.error("Project processing failed or returned no records. Exiting.")
            return

        logger.info("Successfully processed files, ran processors, and filtered groups.")
        logger.info(f"Final DataFrame shape: {project.get_records_df().shape}")
        logger.info(f"\n{project.get_records_df().head()}")

        try:
            logger.info(f"Exporting project to {exported_project_path}...")
            export_project(project, exported_project_path)
            logger.info(f"Project export complete: {exported_project_path}")
        except Exception as e:
            logger.error(f"Failed to export project: {e}")
    else:
        project = import_project(exported_project_path)

    show_report(project)

if __name__ == "__main__":
    # cache_dir = Path.cwd() / "cytodata_cache"
    # if cache_dir.exists():
    #     import shutil
    #     logger.info(f"Clearing cache directory: {cache_dir}")
    #     shutil.rmtree(cache_dir)

    main()