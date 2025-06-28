from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="BingRGB.tar",
    path_in_repo="BingRGB.tar",
    repo_id="imraazib/BingRGB",
    repo_type="dataset",
)
