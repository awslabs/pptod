gdown "https://drive.google.com/u/1/uc?export=download&confirm=3gY1&id=1o2pi8Oe6pdirxvCllf260GFOuKLfXzsu"
mv epoch_9_best_ckpt.zip ./small/full_training/
cd ./small/full_training/
unzip epoch_9_best_ckpt.zip
rm epoch_9_best_ckpt.zip