import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union, Iterable, Optional, Set, Dict, Any, Tuple

import polars as pl
import requests
# ---
from tqdm.auto import tqdm

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.project_settings import Settings as PP_settings
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins

from pixel_patrol_cytodata.settings import CytoDataSettings, S3_NAMESPACE

logger = logging.getLogger(__name__)

def _find_s3_files(s3_folder_url: str, file_suffixes: List[str]) -> List[str]:
    """
    Recursively scans an S3 URL and finds all files ending with
    any of the specified suffixes.
    """
    if not s3_folder_url.startswith("https://"):
        raise ValueError(f"Path is not a valid S3 URL: {s3_folder_url}")
    try:
        base_url_part, prefix = s3_folder_url.split(".com/", 1)
        base_url = base_url_part + ".com/"
    except ValueError:
        raise ValueError(f"Could not parse S3 URL: {s3_folder_url}")

    file_urls = []

    def get_files(current_prefix: str):
        """Recursively get files."""
        try:
            response = requests.get(base_url, params={"prefix": current_prefix, "delimiter": "/"})
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Find files (Contents)
            for content in root.findall(f".//{S3_NAMESPACE}Contents"):
                key_element = content.find(f"{S3_NAMESPACE}Key")
                if key_element is not None:
                    key = key_element.text
                    if key and any(key.lower().endswith(suffix) for suffix in file_suffixes):
                        file_urls.append(f"{base_url}{key}")

            # Find subdirectories (CommonPrefixes)
            for common_prefix in root.findall(f".//{S3_NAMESPACE}CommonPrefixes"):
                prefix_element = common_prefix.find(f"{S3_NAMESPACE}Prefix")
                if prefix_element is not None:
                    subdir_prefix = prefix_element.text
                    if subdir_prefix:
                        get_files(subdir_prefix)

        except requests.RequestException as e:
            logger.error(f"Failed to access S3 URL {base_url} with prefix {current_prefix}: {e}")
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML response from S3: {e}")

    get_files(prefix)
    return file_urls


def _standardize_well_format(well: Optional[str]) -> Optional[str]:
    """
    Safely standardizes well names from 'A1' to 'A01'.
    Handles None, 'A01' (already padded), and 'A1'.
    """
    if well is None:
        return None
    if not isinstance(well, str) or len(well) < 2:
        return well  # Malformed, return as-is

    try:
        letter = well[0].upper()
        number = int(well[1:])
        return f"{letter}{number:02d}"
    except ValueError:
        logger.debug(f"Could not parse well format: {well}")
        return well


class CytoDataProject:
    """
    Manages state and data for a CytoData project, focusing on S3-based
    Cell Painting gallery datasets (e.g., cpg0036).
    """

    def __init__(self, name: str, base_dir: str, loader: str, use_local: bool = True):
        self.name: str = name
        self.base_dir: str = base_dir
        self.paths: Set[str] = set()
        self.use_local = use_local
        self.loader: Optional[PixelPatrolLoader] = discover_loader(loader)
        self.settings: CytoDataSettings = CytoDataSettings()
        self.pp_settings: PP_settings = PP_settings()
        self.records_df: Optional[pl.DataFrame] = None
        self.cache_dir: str = str(Path.cwd() / "cytodata_cache")
        Path(self.cache_dir).mkdir(exist_ok=True)

        # --- DISCOVER PROCESSORS ON INIT ---
        self.processors: List[PixelPatrolProcessor] = discover_processor_plugins()
        logger.info(f"Discovered {len(self.processors)} processor plugins.")

        if not self.loader:
            logger.error(f"Could not find or initialize loader: {loader}")
            raise ValueError(f"Loader '{loader}' not found.")

        logger.info(f"Project Core: Project '{name}' initialized.")
        logger.info(f"Cache directory: {self.cache_dir}")

    def add_paths(self, paths: Union[str, Iterable[str]]) -> None:
        if isinstance(paths, str):
            paths = [paths]
        self.paths.update(paths)
        logger.info(f"Project Core: Paths updated. Total paths count: {len(self.paths)}.")

    def set_settings(self, settings: PP_settings) -> None:
        if not isinstance(settings, PP_settings):
            logger.warning("Settings are not an instance of PP project settings. "
                           "Using as-is, but this may cause issues.")

        self.pp_settings = settings
        logger.info(f"Project Core: PP Project settings updated for '{self.name}'.")

    def set_cd_settings(self, settings: CytoDataSettings) -> None:
        self.settings = settings

        self.use_local = (self.settings.data_source == "local")
        logger.info(f"CytoData settings updated. use_local={self.use_local}")

    def _scan_s3_for_manifests(self) -> List[str]:
        """Scans all registered S3 paths for 'load_data.csv' files."""
        all_manifest_paths = []
        logger.info("Scanning for manifest files (load_data.csv)...")
        for s3_path in self.paths:
            if "load_data_csv" not in s3_path:
                logger.warning(f"Path {s3_path} does not seem to be a manifest folder. Skipping.")
                continue
            logger.info(f"Scanning for manifests in: {s3_path}")
            all_manifest_paths.extend(_find_s3_files(s3_path, [".csv"]))
        return all_manifest_paths

    def _scan_s3_for_platemaps(self) -> List[str]:
        """
        Scans all potential metadata locations for platemaps,
        including .csv, .tsv, and .txt files.
        """
        all_platemap_paths = set()
        file_types_to_scan = [".csv", ".tsv", ".txt"]

        logger.info("No platemap cache found. Scanning *all* metadata locations...")

        if not self.base_dir:
            logger.error("Base directory (base_dir) is not set. Cannot find platemaps.")
            return []

        toplevel_platemap_folder = f"{self.base_dir}/Metadata/platemaps"
        logger.info(f"Scanning for platemaps in: {toplevel_platemap_folder}")
        all_platemap_paths.update(
            _find_s3_files(toplevel_platemap_folder, file_types_to_scan)
        )

        providers = set()
        for path in self.paths:
            try:
                provider = path.split("/")[4]
                providers.add(provider)
            except IndexError:
                logger.warning(f"Could not parse provider from path: {path}")

        if not providers:
            logger.warning("Could not determine any providers. Only scanning top-level metadata.")

        for provider in providers:
            workspace_platemap_folder = f"{self.base_dir}/{provider}/workspace/metadata/"
            logger.info(f"Scanning for platemaps in: {workspace_platemap_folder}")
            all_platemap_paths.update(
                _find_s3_files(workspace_platemap_folder, file_types_to_scan)
            )

        return list(all_platemap_paths)

    def _robust_standardize(self, df: pl.DataFrame, alias_priority_map: Dict[str, List[str]]) -> pl.DataFrame:
        """
        Robustly renames columns based on a priority list of aliases.
        """
        try:
            df = df.rename({col: col.strip() for col in df.columns})
        except Exception as e:
            logger.warning(f"Could not strip whitespace from column names: {e}")

        logger.debug(f"Standardizing cols. Cleaned: {df.columns}")
        cols_to_rename = {}
        df_cols_set = set(df.columns)
        df_cols_lower = {col.lower(): col for col in df.columns}

        for standard_name, aliases in alias_priority_map.items():
            if standard_name in df_cols_set:
                logger.debug(f"Found standard column '{standard_name}'. No rename needed.")
                continue

            found_alias_original_case = None
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower in df_cols_lower:
                    found_alias_original_case = df_cols_lower[alias_lower]
                    break

            if found_alias_original_case:
                for other_alias in aliases:
                    other_alias_lower = other_alias.lower()
                    if other_alias_lower == found_alias_original_case.lower():
                        continue
                    if other_alias_lower in df_cols_lower:
                        ambiguous_col_original_case = df_cols_lower[other_alias_lower]
                        logger.warning(
                            f"Ambiguous columns found for '{standard_name}'. "
                            f"Using '{found_alias_original_case}' "
                            f"and ignoring '{ambiguous_col_original_case}'."
                        )
                if standard_name not in cols_to_rename.values():
                    if found_alias_original_case != standard_name:
                        cols_to_rename[found_alias_original_case] = standard_name
                else:
                    logger.warning(
                        f"Duplicate rename target. Cannot rename '{found_alias_original_case}' to "
                        f"'{standard_name}' as another col is already targeting it."
                    )

        if cols_to_rename:
            logger.debug(f"Rename map: {cols_to_rename}")
            try:
                df = df.rename(cols_to_rename)
            except pl.DuplicateError as e:
                logger.error(
                    f"DuplicateError during rename: {e}. "
                    f"Cols to rename: {cols_to_rename}. "
                    f"DF Cols: {df.columns}"
                )
                try:
                    cols_to_drop = cols_to_rename.keys()
                    logger.warning(f"Attempting recovery by dropping source columns: {cols_to_drop}")
                    return df.drop(list(cols_to_drop))
                except Exception as drop_e:
                    logger.error(f"Recovery failed: {drop_e}. Returning original df.")
                    return df
        return df

    def _standardize_manifest_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Renames known aliases in a single manifest DataFrame.
        """
        alias_map: Dict[str, List[str]] = {
            "Metadata_Plate": ["Metadata_Plate", "Metadata_Platename", "Plate", "plate", "plate_name"],
            "Metadata_Well": ["Metadata_Well", "Well", "well", "well_id"],
            "Metadata_Site": ["Metadata_Site", "Site", "site"],
            "Metadata_Provider": ["Metadata_Provider", "Provider"],
            "PathName_DNA": ["PathName_DNA", "PathName_OrigDNA", "PathName_DAPI"],
            "PathName_AGP": ["PathName_AGP", "PathName_OrigAGP", "PathName_ConcanavalinA"],
            "PathName_RNA": ["PathName_RNA", "PathName_OrigRNA", "PathName_SYTO14"],
            "PathName_ER": ["PathName_ER", "PathName_OrigER", "PathName_ER_EGFP"],
            "PathName_Mito": ["PathName_Mito", "PathName_OrigMito", "PathName_MitoTracker"],
            "FileName_DNA": ["FileName_DNA", "FileName_OrigDNA", "FileName_DAPI"],
            "FileName_AGP": ["FileName_AGP", "FileName_OrigAGP", "FileName_ConcanavalinA"],
            "FileName_RNA": ["FileName_RNA", "FileName_OrigRNA", "FileName_SYTO14"],
            "FileName_ER": ["FileName_ER", "FileName_OrigER", "FileName_ER_EGFP"],
            "FileName_Mito": ["FileName_Mito", "FileName_OrigMito", "FileName_MitoTracker"],
        }
        return self._robust_standardize(df, alias_map)

    def _get_manifest_df(self) -> pl.DataFrame:
        """Loads all manifest files, standardizes, and merges them."""
        cache_path = Path(self.cache_dir) / "manifest_cache.parquet"

        if cache_path.exists():
            try:
                logger.info(f"Loading manifests from cache: {cache_path}")
                return pl.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load manifest cache: {e}. Re-scanning S3.")

        if self.use_local:
            all_manifest_paths = self._scan_local_for_manifests()
        else:
            all_manifest_paths = self._scan_s3_for_manifests()


        if not all_manifest_paths:
            logger.error("No manifest files (load_data.csv) found during scan.")
            return pl.DataFrame()

        logger.info(f"Found {len(all_manifest_paths)} total manifest files.")

        manifest_dfs = []
        for path in tqdm(all_manifest_paths, desc="Reading manifests"):

            try:
                if self.use_local:
                    df = pl.read_csv(path, ignore_errors=True)
                else:
                    df = pl.read_csv(path, storage_options={"anon": True}, ignore_errors=True)

                df = self._standardize_manifest_cols(df)

                if "Metadata_Provider" not in df.columns:
                    try:
                        provider = path.split("/")[4]
                        df = df.with_columns(pl.lit(provider).alias("Metadata_Provider"))
                    except IndexError:
                        logger.warning(f"Could not parse provider from path: {path}")

                df = df.with_columns(pl.lit(path).alias("Metadata_ManifestPath"))
                logger.debug(f"Manifest {Path(path).name} cols after std: {df.columns}")
                manifest_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read or standardize manifest {path}: {e}")

        if not manifest_dfs:
            logger.error("No manifest data could be loaded after processing all paths.")
            return pl.DataFrame()

        try:
            manifest_df = pl.concat(manifest_dfs, how="diagonal_relaxed")
        except Exception as e:
            logger.error(f"Failed to concatenate manifest DataFrames: {e}")
            return pl.DataFrame()

        try:
            manifest_df.write_parquet(cache_path)
            logger.info(f"Manifest cache saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write manifest cache: {e}")

        return manifest_df

    def _standardize_platemap_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Renames known aliases in a single platemap DataFrame based on log evidence.
        """
        alias_map: Dict[str, List[str]] = {
            "Metadata_Plate": ["Metadata_Plate", "plate_map_name"],
            "Metadata_Well": ["Metadata_Well", "well_position", "Well", "well"],
            "Metadata_Provider_Platemap": ["Metadata_Partner", "Metadata_Provider", "Provider"],
            "Metadata_BroadSample": ["Metadata_BroadSample", "broad_sample", "BROAD_SAMPLE", "BroadSample"],
            "Metadata_PerkinElmerID": ["Metadata_PerkinElmerID", "pert_id", "PerkinElmerID", "pert_id_broad"],
            "Metadata_Name": ["Metadata_Name", "pert_name", "Name", "Compound_Name"],
            "Metadata_EOS": ["Metadata_EOS", "EOS_ID", "EOS", "eos_id"],
        }
        return self._robust_standardize(df, alias_map)

    def _get_platemap_df(self) -> pl.DataFrame:
        """
        Loads all platemaps from S3 paths, merges them, and applies renaming.
        """
        cache_path = Path(self.cache_dir) / "platemap_cache.parquet"

        if cache_path.exists():
            try:
                logger.info(f"Loading platemaps from cache: {cache_path}")
                return pl.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load platemap cache: {e}. Re-scanning S3.")

        if self.use_local:
            all_platemap_paths = self._scan_local_for_platemaps()
        else:
            all_platemap_paths = self._scan_s3_for_platemaps()
        if not all_platemap_paths:
            logger.error("No platemap files found during scan.")
            return pl.DataFrame()

        logger.info(f"Found {len(all_platemap_paths)} total platemap files. Filtering for valid platemaps...")

        platemap_dfs = []

        for path in tqdm(all_platemap_paths, desc="Reading platemaps"):
            try:
                path_lower = path.lower()
                is_s3 = path_lower.startswith("s3://") or path_lower.startswith("https://")

                read_args = dict(ignore_errors=True)
                if not self.use_local and is_s3:
                    read_args["storage_options"] = {"anon": True}

                if path_lower.endswith(".tsv"):
                    df = pl.read_csv(path, separator="\t", **read_args)

                elif path_lower.endswith(".txt"):
                    logger.debug(f"Reading space-delimited file: {path}")
                    df = pl.read_csv(path, separator=" ", has_header=True, **read_args)

                elif path_lower.endswith(".csv"):
                    df = pl.read_csv(path, separator=",", **read_args)

                else:
                    logger.debug(f"Skipping file with unhandled extension: {path}")
                    continue

            except Exception as e:
                logger.warning(f"Failed to read file {path}: {e}")
                continue

            if df is None or df.is_empty():
                continue

            try:
                df_std = self._standardize_platemap_cols(df)

                required_cols = ["Metadata_Well", "Metadata_EOS", "Metadata_Plate"]
                if not all(col in df_std.columns for col in required_cols):
                    logger.debug(f"Skipping platemap {Path(path).name}: Missing one of {required_cols}. "
                                 f"Standardized columns: {df_std.columns}")
                    continue

                logger.debug(f"Platemap {Path(path).name} cols after std: {df_std.columns}")
                platemap_dfs.append(df_std)

            except Exception as e:
                logger.warning(f"Failed to *standardize* platemap {path}: {e}")

        if not platemap_dfs:
            logger.error("No valid, well-level platemap data could be loaded after processing all paths.")
            return pl.DataFrame()

        logger.info(f"Successfully loaded and standardized {len(platemap_dfs)} well-level platemaps.")

        try:
            platemap_df = pl.concat(platemap_dfs, how="diagonal_relaxed")
        except Exception as e:
            logger.error(f"Failed to concatenate platemap DataFrames: {e}")
            return pl.DataFrame()

        try:
            platemap_df.write_parquet(cache_path)
            logger.info(f"Platemap cache saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write platemap cache: {e}")

        return platemap_df

    def _apply_comparison_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the filters defined in settings.comparison_filters
        and creates the 'imported_path_short' column for dashboard compatibility.
        """
        if not self.settings.comparison_filters:
            logger.warning("No comparison filters found in settings.")
            return df

        if not isinstance(self.settings.comparison_filters, dict):
            logger.error("settings.comparison_filters must be a dictionary.")
            return df

        logger.info(f"Applying {len(self.settings.comparison_filters)} filters...")

        group_expressions = []
        for group_name, filter_expr in self.settings.comparison_filters.items():
            if not isinstance(filter_expr, pl.Expr):
                logger.error(f"Filter for group '{group_name}' is not a Polars expression.")
                continue

            group_expressions.append(
                pl.when(filter_expr).then(pl.lit(group_name)).otherwise(pl.lit(None))
            )

        if not group_expressions:
            logger.error("No valid filter expressions were created.")
            return df

        try:
            df = df.with_columns(
                pl.coalesce(group_expressions).alias("imported_path_short")
            )
            df = df.with_columns(
                pl.coalesce(group_expressions).alias("imported_path")
            )
            df = df.filter(pl.col("imported_path_short").is_not_null())

        except Exception as e:
            logger.error(f"Error applying filter expressions: {e}")
            logger.error(f"DataFrame columns: {df.columns}")

        return df

    def _clean_join_keys(self, df: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
        """
        Cleans the join keys on a DataFrame to ensure joins will succeed.
        - Strips whitespace from all string keys
        - Standardizes well padding (e.g., A1 -> A01)
        """
        logger.debug(f"Cleaning join keys: {keys}")

        for key in keys:
            if key in df.columns and df.get_column(key).dtype == pl.String:
                df = df.with_columns(
                    pl.col(key).str.strip_chars().alias(key)
                )

        if "Metadata_Well" in keys and "Metadata_Well" in df.columns:
            try:
                df = df.with_columns(
                    pl.col("Metadata_Well").map_elements(
                        _standardize_well_format,
                        return_dtype=pl.String
                    ).alias("Metadata_Well")
                )
                logger.debug("Successfully standardized Metadata_Well padding (e.g., A1 -> A01).")
            except Exception as e:
                logger.warning(f"Could not standardize Metadata_Well padding: {e}. "
                               f"This may cause join failure if padding is inconsistent.")

        return df

    # --- NEW HELPER FOR PARALLEL PROCESSING ---
    def _process_single_record(self, path: str) -> Dict[str, Any]:
        """
        Helper function to load and process a *single* file.
        This will be called by the parallel executor.
        """
        if self.loader is None:
            return {"path": path}  # Should be impossible if called from _run_deep_processing

        try:
            # 1. Load the file using the loader
            art = self.loader.load(path)
            if art is None:
                logger.warning(f"Loader failed for path: {path}")
                return {"path": path}  # Return path to join

            # 2. Run all discovered processors on the loaded record
            processor_metadata = {}
            for P in self.processors:
                if not is_record_matching_processor(art, P.INPUT):
                    continue

                try:
                    out = P.run(art)
                    if isinstance(out, dict):
                        processor_metadata.update(out)
                    else:
                        art = out  # Allow for chained processors
                        if hasattr(art, 'meta'):
                            processor_metadata.update(art.meta)
                except Exception as e:
                    logger.warning(f"Processor {P.NAME} failed for {path}: {e}")

            return {"path": path, **processor_metadata}

        except Exception as e:
            logger.error(f"Critical error during processing for {path}: {e}")
            return {"path": path}  # Return path to join

    # --- END HELPER ---

    def _run_deep_processing(self) -> pl.DataFrame:
        """
        Runs the deep processing loop (load image, run processors) on the
        final, sampled list of records *in parallel*.
        """
        if self.records_df is None or self.records_df.is_empty():
            logger.warning("No records to process. Skipping deep processing.")
            return pl.DataFrame()

        if self.loader is None:
            logger.error("No loader set. Cannot run deep processing.")
            return pl.DataFrame()

        # Get the final list of paths to process
        try:
            if self.use_local:
                self.records_df = self.records_df.with_columns(
                    pl.when(pl.col("URL_OrigDNA").str.starts_with(self.settings.s3_prefix))
                    .then(
                        pl.col("URL_OrigDNA")
                        .str.replace(self.settings.s3_prefix, self.settings.local_prefix, literal=True)
                    )
                    .otherwise(pl.col("URL_OrigDNA"))
                    .alias("URL_OrigDNA")
                )

            self.records_df = self.records_df.with_columns([
                pl.col("URL_OrigDNA").alias("path"),
                pl.col("URL_OrigDNA").str.split("/").list.last().alias("name")
            ])
            self.records_df = self.records_df.with_columns([
                pl.col("name").str.split(".").list.last().alias("file_extension"),
                # Add 'size_bytes' as null to satisfy dashboard schema
                pl.lit(0, dtype=pl.Int64).alias("size_bytes")
            ])
            # --- END FIX ---

            paths_to_process = self.records_df["path"].to_list()
        except pl.ColumnNotFoundError:
            logger.error("Fatal: 'URL_OrigDNA' column not found. Cannot determine paths for processing.")
            return pl.DataFrame()

        logger.info(f"Starting deep processing loop for {len(paths_to_process)} files...")

        processor_rows = []

        with ThreadPoolExecutor(max_workers=self.settings.max_parallel_workers) as executor:
            futures = {executor.submit(self._process_single_record, path): path for path in paths_to_process}

            for future in tqdm(as_completed(futures), total=len(paths_to_process), desc="Running processors",
                               unit="file"):
                try:
                    result = future.result()
                    processor_rows.append(result)
                except Exception as e:
                    path = futures[future]
                    logger.error(f"A future failed for path {path}: {e}")
                    processor_rows.append({"path": path})  # Append empty row

        if not processor_rows:
            logger.warning("Deep processing ran but returned no processor metadata.")
            return pl.DataFrame()

        return pl.DataFrame(processor_rows)

    def process_records(self) -> "CytoDataProject":
        """
        Main processing pipeline:
        1. Build the list of files by joining manifests and platemaps.
        2. Filter this list down to the groups of interest.
        3. Sample the filtered list.
        4. Run the deep processing (load image + run processors) on the sampled list.
        5. Join the processor results back to the main DataFrame.
        """
        if self.loader is None:
            logger.error("No loader set. Cannot process records.")
            return self

        manifest_df = self._get_manifest_df()
        platemap_df = self._get_platemap_df()

        if manifest_df.is_empty() or platemap_df.is_empty():
            logger.error("Manifest or Platemap DataFrame is empty. Cannot proceed.")
            return self

        logger.info("Creating 'Metadata_Plate_Base' join key on ManifestDF...")
        try:
            manifest_df = manifest_df.with_columns(
                pl.col("Metadata_Plate").str.split("_R").list.get(0).alias("Metadata_Plate_Base")
            )
        except Exception as e:
            logger.error(f"Failed to create 'Metadata_Plate_Base' column: {e}")
            return self

        logger.info("Cleaning join keys on ManifestDF...")
        manifest_keys_to_clean = ["Metadata_Plate_Base", "Metadata_Well"]
        manifest_df = self._clean_join_keys(manifest_df, manifest_keys_to_clean)

        logger.info("Cleaning join keys on PlatemapDF...")
        platemap_keys_to_clean = ["Metadata_Plate", "Metadata_Well"]
        platemap_df = self._clean_join_keys(platemap_df, platemap_keys_to_clean)

        try:
            self.records_df = manifest_df.join(
                platemap_df,
                left_on=["Metadata_Plate_Base", "Metadata_Well"],
                right_on=["Metadata_Plate", "Metadata_Well"],
                how="left"
            )

            self.records_df = (
                self.records_df.group_by("URL_OrigDNA", maintain_order=True)
                .agg([
                    pl.all().exclude("URL_OrigDNA").first()  # take the first non-null per column
                ])
            )

            self.records_df = self.records_df.drop("Metadata_Plate_Base")

            logger.info(f"Manifest and Platemap joined. Total records: {len(self.records_df)}")

        except Exception as e:
            logger.error(f"An unexpected error occurred during join: {e}")
            return self

        if self.records_df.is_empty():
            logger.warning("No records found after joining manifest and platemap data.")
            return self

        if self.settings and self.settings.comparison_filters:
            logger.info("Applying comparison filters to joined DataFrame...")
            self.records_df = self._apply_comparison_filters(self.records_df)

            if self.records_df.is_empty():
                logger.error("DataFrame is empty after applying filters. No groups found.")
                return self

            logger.info("Group counts after filtering:")
            logger.info(f"\n{self.records_df['imported_path_short'].value_counts(sort=True)}")

            if self.settings.sample_per_group > 0:
                n_samples = self.settings.sample_per_group
                logger.info(f"Sampling DataFrame to {n_samples} records per group...")

                self.records_df = self.records_df.group_by("imported_path_short").map_groups(
                    lambda group_df: group_df.sample(
                        n=min(n_samples, len(group_df)),
                        with_replacement=False,  # Ensure we don't pick the same file twice
                        seed=42
                    )
                )

                logger.info("Sampling complete. New group counts:")
                logger.info(f"\n{self.records_df['imported_path_short'].value_counts(sort=True)}")

        else:
            logger.info("No comparison filters set; proceeding without grouping.")
            if self.settings.sample_per_group and self.settings.sample_per_group > 0:
                n = min(self.settings.sample_per_group, len(self.records_df))
                logger.info(f"Sampling {n} records (global).")
                self.records_df = self.records_df.sample(n=n, with_replacement=False, seed=42)

        processor_df = self._run_deep_processing()

        if not processor_df.is_empty():
            logger.info(f"Records df shape: {self.records_df.shape}")
            logger.info(f"Records df unique paths: {self.records_df.select(pl.col('path').n_unique()).item()}")
            logger.info(f"Processor df shape: {processor_df.shape}")
            logger.info(f"Processor df unique paths: {processor_df.select(pl.col('path').n_unique()).item()}")

            logger.info("Joining processor metadata back to main DataFrame...")
            try:
                self.records_df = self.records_df.join(
                    processor_df,
                    on="path",
                    how="left"
                )
            except Exception as e:
                logger.error(f"Failed to join processor metadata: {e}")
        else:
            logger.warning("No processor metadata was generated. Final DataFrame will not be enriched.")

        return self

    def add_manifest_dirs_from_csv(self, csv_path: str) -> None:
        df = pl.read_csv(csv_path)
        if "manifest_dir" not in df.columns:
            raise ValueError("CSV must contain a 'manifest_dir' column")
        self.add_paths(df["manifest_dir"].to_list())  # reuse existing API

    def _scan_local_for_manifests(self) -> list[str]:
        """Find local load_data.csv files."""
        files = []
        for p in self.paths:
            mp = Path(p) / "load_data.csv"
            if mp.exists():
                files.append(str(mp))
        return files

    def _scan_local_for_platemaps(self) -> list[str]:
        files = set()
        for p in self.paths:
            parts: Tuple[str, ...] = Path(p).parts  # or: parts: tuple[str, ...] = Path(p).parts
            if "workspace" in parts:
                wi = parts.index("workspace")
                prov = parts[wi - 1] if wi > 0 else None
            else:
                prov = None
            if prov is None:
                continue
            meta = Path(self.base_dir) / prov / "workspace" / "metadata"
            for ext in ("*.csv", "*.tsv", "*.txt"):
                files.update(map(str, meta.rglob(ext)))
        return list(files)


    # --- Getters ---
    def get_name(self) -> str:
        return self.name

    def get_paths(self) -> Set[str]:
        return self.paths

    def get_settings(self) -> PP_settings:
        return self.pp_settings

    def get_cd_settings(self) -> CytoDataSettings:
        return self.settings

    def get_records_df(self) -> Optional[pl.DataFrame]:
        return self.records_df

    def get_loader(self) -> Optional[PixelPatrolLoader]:
        return self.loader
