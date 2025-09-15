## MPDD One-Class Segmentation

This repository assumes the user has already downloaded here:https://github.com/stepanje/MPDD?tab=readme-ov-file and follows the following set up:

data_root/
├── bracket_black/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/scratch/*.png
│   ├── test/hole/*.png
│   └── ground_truth/
│       ├── scratch/*.png
│       └── hole/*.png
├── bracket_silver/
│   ├── train/good/*.png
│   └── ...
└── gear_shaft/
    ├── train/good/*.png
    └── ...


Provides data setup, process, and evaluation of 2 complementary training modes;

1. Plain U-Net (per metal)
    -Reconstruction based autoencoder from scratch that trains separately for each class.
    - Learns only "good" class from training images, learning distributions for 'normal' cases
    - Aims to detect anomolies in test images via residual errors
    
2. Shared U-Net (multi-metal with class condition)
    -Trains shared autoencoder across all classes
    - Uses metal_id conditioning to adapt reconstruction to different metal appearances
    

## Installation

git clone https://github.com/yourusername/mpdd_segmentation.git
cd mpdd_segmentation
pip install -r requirements.txt


`run.py` trains plain_unet on single metal defined in bash, and trains shared U-Net over all metals, then compares performance on selected metal.

`python run.py \
  --data_root /path/to/anomaly_dataset \
  --metal bracket_black \
  --img_size 256 \
  --batch_size 16 \
  --epochs 40 \
  --lr 1e-3 \
  --device cpu \
  --save_dir results`

