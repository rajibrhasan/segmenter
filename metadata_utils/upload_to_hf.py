from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="/teamspace/studios/this_studio/segmenter/train.tar",
    path_in_repo="train.tar",
    repo_id="imraazib/BingRGB",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="/teamspace/studios/this_studio/segmenter/val.tar",
    path_in_repo="val.tar",
    repo_id="imraazib/BingRGB",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="/teamspace/studios/this_studio/segmenter/Dataset/BingRGB/train_metadata.json",
    path_in_repo="train_metadata.json",
    repo_id="imraazib/BingRGB",
    repo_type="dataset",
)

api.upload_file(
    path_or_fileobj="/teamspace/studios/this_studio/segmenter/Dataset/BingRGB/val_metadata.json",
    path_in_repo="val_metadata.json",
    repo_id="imraazib/BingRGB",
    repo_type="dataset",
)