import numpy as np
import json
from pathlib import Path
from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir


RGB_CUSTOM_CONFIG_PATH = Path(__file__).parent / "config" / "bing_rgb.py"
RGB_CUSTOM_CATS_PATH = Path(__file__).parent / "config" / "bing_rgb.yml"


def format_patch_text(data):
    template = (
    '"Location": "{Location}", "Upazila": "{Upazila}", "District": "{District}", '
    '"Population Density": {Population_Density}, "Literacy": {Literacy}, "Region Type": "{Region_Type}", '
    'Distance to district sadar is {distance_to_district_sadar}, distance to upazila sadar is {distance_to_upazila_sadar}, '
    '{inside_district_sadar_text}, {inside_upazila_sadar_text}'
    )
    inside_district_sadar_text = (
        "inside district sadar" if data["inside_district_sadar"] else "not inside district sadar"
    )
    inside_upazila_sadar_text = (
        "inside upazila sadar" if data["inside_upazila_sadar"] else "not inside upazila sadar"
    )
    
    return template.format(
        Location=data["Location"],
        Upazila=data["Upazila"],
        District=data["District"],
        Population_Density=data["Population Density"],
        Literacy=data["Literacy"],
        Region_Type=data["Region Type"],
        distance_to_district_sadar=data["distance_to_district_sadar"],
        distance_to_upazila_sadar=data["distance_to_upazila_sadar"],
        inside_district_sadar_text=inside_district_sadar_text,
        inside_upazila_sadar_text=inside_upazila_sadar_text
    )


class BingRGBDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, tokenizer, **kwargs):
        super().__init__(image_size, crop_size, split, RGB_CUSTOM_CONFIG_PATH, **kwargs)
        self.names, self.colors = utils.dataset_cat_description(RGB_CUSTOM_CATS_PATH)
        self.n_cls = len(self.names)
        self.ignore_label = 255
        self.reduce_zero_label = False
        self.tokenizer = tokenizer 
        metadata_path = f'{self.config.data_root}/{split}_metadata.json'        

        with open(metadata_path, "r") as f:
            self.captions = json.load(f)
       

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "BingRGB"  # Your dataset root directory
        config.data_root = path

        config.data[self.split]["data_root"] = path
        config = super().update_default_config(config)

        return config

    def test_post_process(self, labels):
        # Return labels directly or implement remapping if needed
        return labels

    def __getitem__(self, idx):
        data = self.dataset[idx]

        train_splits = ["train", "trainval"]

        # Load image and segmentation
        if self.split in train_splits:
            im = data["img"].data
            seg = data["gt_semantic_seg"].data.squeeze(0)

            #Mapping background to 255
            seg[seg == 0] = 255
            seg[seg != 255] -= 1
        else:
            im = [im.data for im in data["img"]]
            seg = None
        

        if self.split in train_splits:
            file_name = data["img_metas"].data["ori_filename"]
        else:
            file_name = data["img_metas"][0].data["ori_filename"]

        # Get text for the sample and tokenize
        text_caption = format_patch_text(self.captions.get(file_name))
        # example_text = "This patch is located in Shyampur union, under Tejgaon Development Circle Upazila of Dhaka District. Shyampur union has a population of 214,000, with a density of 16,525 people per square kilometer. The area of the union is 12.95 square kilometers, and the literacy rate is 97.0%. This patch is 10 km away from the nearest district center and 10 km away from the nearest upazila center. This patch is located outside of the District Sadar. This patch is located outside of the Upazila Sadar."
        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(
                text_caption,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
        else:
            tokenized_text = None

        out = dict(im=im, text=tokenized_text)
        if self.split in train_splits:
            out["segmentation"] = seg
        else:
            im_metas = [meta.data for meta in data["img_metas"]]
            out["im_metas"] = im_metas
            out["colors"] = self.colors

        return out
