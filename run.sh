python -m segm.train --log-dir checkpoints/seg_clip_base_mask --dataset bing_rgb \
  --backbone vit_base_patch16_clip_224 --text_encoder clip_vit_base_patch16 --decoder mask_transformer