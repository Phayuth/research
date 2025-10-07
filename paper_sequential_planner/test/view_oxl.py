import objaverse.xl as oxl
import pandas as pd
from util import rsrcpath

annotations_path = rsrcpath + "objv"
download_path = rsrcpath + "down3d"

annotations = oxl.get_annotations(download_dir=annotations_path)
meta = annotations["metadata"]
cansobj = annotations[meta.str.contains("phone")]


def handle_found_object(local_path, file_identifier, sha256, metadata):
    print(
        "\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
        f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n",
    )


objdown = oxl.download_objects(
    cansobj,
    download_dir=download_path,
    processes=8,
    handle_found_object=handle_found_object,
)
