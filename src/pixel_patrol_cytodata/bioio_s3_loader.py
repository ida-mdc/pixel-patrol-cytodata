import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import bioio_imageio
import polars as pl
from bioio import BioImage
from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_loader_bio.plugins.loaders._utils import is_zarr_store
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import _extract_metadata

logger = logging.getLogger(__name__)


def _load_bioio_image(source: Union[str, Path]) -> Optional[BioImage]:
    """
    Attempts to load an image from a path or S3 URL using bioio.

    --- FIX ---
    The default bioio OME-TIFF reader hangs on remote S3 files when
    reading metadata. We will *force* it to use the imageio
    reader, which is more robust for remote files.
    """
    storage_options = {}
    if isinstance(source, str) and (source.startswith("https://") or source.startswith("s3://")):
        # Use s3fs anonymous login
        storage_options = {"anon": True}

    try:
        return BioImage(source, reader=bioio_imageio.Reader, fs_kwargs=storage_options)
    except Exception as e:
        logger.warning(f"Could not load '{source}' with BioImage (imageio reader): {e}")
        return None


class BioIoS3Loader:
    """
    A Pixel Patrol loader that uses bioio to load images from local paths or S3 URLs.
    This loader is designed to work with the CytoDataProject class.

    Note: This loader doesn't load image data itself, but rather metadata.
    The actual image loading would be handled by a processor.
    """
    NAME = "bioio-s3"
    SUPPORTED_EXTENSIONS: Set[str] = {"czi", "tif", "tiff", "ome.tif", "nd2", "lif", "jpg", "jpeg", "png", "bmp",
                                      "ome.zarr"}
    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str, "dim_names": list, "n_images": int, "num_pixels": int,
        "shape": pl.List(pl.UInt32), "ndim": int, "channel_names": list, "dtype": str,
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        (r"^pixel_size_[A-Za-z]$", float), (r"^[A-Za-z]_size$", int),
    ]
    FOLDER_EXTENSIONS: Set[str] = {"zarr", "ome.zarr"}

    def is_folder_supported(self, folder_path: Union[str, Path]) -> bool:
        return is_zarr_store(folder_path)

    def load(self, path: Union[str, Path]) -> Optional[Record]:
        """
        Loads metadata *and* pixel data from an S3 URL or local path.
        Returns a Record object.
        """
        try:
            path_str = str(path)
            if path_str.startswith("https://"):
                path_str = path_str.replace("https://cellpainting-gallery.s3.amazonaws.com/",
                                            "s3://cellpainting-gallery/")

            img = _load_bioio_image(path_str)
            if img is None:
                return None

            metadata = _extract_metadata(img)

            array_data = img.dask_data

            return record_from(array_data, metadata, kind="intensity")

        except Exception as e:
            logger.warning(f"Failed to load metadata or data from {path}: {e}")
            return None