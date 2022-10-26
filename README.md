# HADM
Human Attention based Driving Model (HADM)

## Step To Use

1.  ````shell
    python3 downsample.py
      --image_dir 'data/images/gazemap_images'
      
2.  ````shell
    python3 features.py \
      --source 'data/images/ds_images' \
      --weights yolov5s.pt \
      --conf 0.25  \
      --save-txt \
      --save-conf \
      --features 'data/features'

3.  ````shell
    python3 grid.py \
      --gazemaps 'data/images/gazemap_images/ds_images' \
      --grids 'data/grids/test_grid.txt' \
      --gridheight 16 \
      --gridwidth 16
