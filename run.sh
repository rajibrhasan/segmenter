python -m segm.train --log-dir seg_base_mask --dataset bing_rgb \
  --backbone vit_base_patch16_384 --text_encoder clip_vit_base_patch16 --decoder mask_transformer