import os
from turtle import down
import click
from huggingface_hub import hf_hub_download
from segm.utils.download import download
import tarfile
from pathlib import Path

DOWNLOAD_URL = [
    (
        "https://huggingface.co/datasets/imraazib/BingRGB/resolve/main/train.tar", 
        "e493f5ba1436c2d1f9329aa347bbb7f0a83c2371"
    ),

    (
       "https://huggingface.co/datasets/imraazib/BingRGB/resolve/main/val.tar",
       "079902d6334df9187ebd7e9a4f8fcf4e0c95b31e"
    )
]

METADATA_URL = [
    (
        "https://huggingface.co/datasets/imraazib/BingRGB/resolve/main/train_metadata.json",
        "e51d7cf73e9a1ed959ca4c08e70e98a2010348aa"
    ),
    (
        "https://huggingface.co/datasets/imraazib/BingRGB/resolve/main/val_metadata.json",
        "7556e428e81e5393f4ab3cea0590fd6c8e1a4a2a"
    )
]

# checksum  = "f440cc40356f286690dda492ddba5980dfa89e4b"

def download_metadata(download_dir, overwrite=False):
    for url, checksum in METADATA_URL:
        fname = download(url, path = download_dir, overwrite=overwrite, sha1_hash = checksum)
    
    print(f"Downloaded metadata from hugingface repo.")



def download_bingrgb(path, overwrite=False):
    download_dir = path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    for url, checksum in DOWNLOAD_URL:
        raw_data = download(url, path = download_dir, overwrite=overwrite, sha1_hash=checksum)
        extract_to = path.resolve()
        print(extract_to)
        with tarfile.open(raw_data, 'r:*') as tar:
            tar.extractall(path = str(path))

    print(f"Downloaded and unzipped BingRGB dataset from hugingface dataset repo.")

@click.command(help="Initialize ADE20K dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):
    dataset_dir = Path(download_dir) / "BingRGB"
    download_bingrgb(dataset_dir, overwrite=False)
    download_metadata(dataset_dir, overwrite=False)

if __name__ == "__main__":
    main()
