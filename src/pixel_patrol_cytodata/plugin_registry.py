from pixel_patrol_cytodata.bioio_s3_loader import BioIoS3Loader


def register_loader_plugins():
    return [
        BioIoS3Loader
    ]
