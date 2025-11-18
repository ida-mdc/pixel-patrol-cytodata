# compare_compounds_local.py
import logging
import sys
from pathlib import Path

import polars as pl  # noqa: F401 (PP uses Polars DataFrames)
from pixel_patrol_base.api import export_project, show_report
from pixel_patrol_base.core.project_settings import Settings as PP_settings
from pixel_patrol_cytodata.cytodata_project import CytoDataProject
from pixel_patrol_cytodata.settings import CytoDataSettings

# --- Config (edit these) ---
PROJECT_NAME   = "CytoData_Local_HepG2"
BASE_DIR       = "/data/datasets/01_Bioactives_images/input"
MANIFEST_CSV   = "manifest_filtered_HepG2_B1004-B1006_R1-2_IMTM-USC_minprov2_20251118_143043.csv"
SAMPLE_PER_GRP = 300  # larger sample (local I/O is faster)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def main():
    out_dir = Path(__file__).parent / "exported_projects"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{PROJECT_NAME}.zip"

    # Project: local mode, driven by filtered manifest list
    project = CytoDataProject(
        name=PROJECT_NAME,
        base_dir=BASE_DIR,
        loader="bioio-s3",     # same loader; it handles local paths too
    )

    # PixelPatrol UI/report settings (optional)
    project.set_settings(PP_settings(
        cmap="rainbow",        # tweak if you like
        # n_example_files=DEFAULT_N_EXAMPLE_FILES (uses PP-base default)
    ))

    # CytoData pipeline settings (local mode, workers, sampling)
    project.set_cd_settings(CytoDataSettings(
        data_source="local",
        local_prefix=BASE_DIR,
        sample_per_group=SAMPLE_PER_GRP,
        max_parallel_workers=8,
        # comparison_filters=None  # process once, slice later
    ))

    # Feed the 1-column CSV of manifest directories (your 8 dirs)
    project.add_manifest_dirs_from_csv(MANIFEST_CSV)

    logger.info("Processing records…")
    project.process_records()
    df = project.get_records_df()
    if df is None or df.is_empty():
        logger.error("No records produced. Check MANIFEST_CSV/BASE_DIR.")
        return

    logger.info(f"Processed records: shape={df.shape}")
    logger.info("Exporting project zip…")
    export_project(project, export_path)
    logger.info(f"Exported: {export_path}")

    # Open the interactive report
    show_report(project)

if __name__ == "__main__":
    main()
