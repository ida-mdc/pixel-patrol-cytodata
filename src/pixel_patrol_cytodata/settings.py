from dataclasses import dataclass

S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"

@dataclass
class CytoDataSettings:
    comparison_filters: dict | None = None
    sample_per_group: int = 50
    data_source: str = "auto"  # "auto" | "s3" | "local"
    s3_prefix: str = "s3://cellpainting-gallery/cpg0036-EU-OS-bioactives"
    local_prefix: str = "/data/datasets/01_Bioactives_images/input"
    max_parallel_workers: int = 8