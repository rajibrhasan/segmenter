import os
import click
from huggingface_hub import hf_hub_download
from segm.utils.download import download
import tarfile

download_url = "https://huggingface.co/datasets/imraazib/BingRGB/resolve/main/BingRGB.tar"
checksum  = "f440cc40356f286690dda492ddba5980dfa89e4b"

def download_bingrgb(download_dir, overwrite=False):
    os.makedirs(download_dir, exist_ok=True)
    raw_data = download(download_url, path = download_dir, overwrite=overwrite, sha1_hash=checksum)
    extract_to = os.path.dirname(os.path.abspath(download_dir))
    with tarfile.open(raw_data, 'r:*') as tar:
        tar.extractall(path = extract_to)

    print(f"Downloaded and unzipped BingRGB dataset from hugingface dataset repo.")

@click.command(help="Initialize ADE20K dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):
    download_bingrgb(download_dir, overwrite=False)

if __name__ == "__main__":
    main()
