# dataset settings
dataset_type = "BingRGBDataset"  # Or your own registered dataset class name
data_root = "BingRGB/"

custom_imports = dict(
    imports=['segm.data.mmseg.custom_bing_rgb'],  # Python path (relative to PYTHONPATH)
    allow_failed_imports=False
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # You can replace with your dataset's mean/std
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
crop_size = (512, 512)
max_ratio = 2

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=False),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_metadata_path = "train_metadata.json",
    val_metadata_path = "val_metadata.json",
    test_metadata_path = "val_metadata.json",
    
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        ann_dir="train/gts",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/images",
        ann_dir="val/gts",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/images",
        ann_dir="val/gts",
        pipeline=test_pipeline,
    ),
)
