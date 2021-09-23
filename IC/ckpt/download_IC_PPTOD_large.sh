gdown "https://drive.google.com/u/1/uc?export=download&confirm=hn2z&id=1NKCxORsjp4zca2YBnRhpbLWmIFBtTT9O"
mv epoch_70_best_ckpt.zip ./large/full_training/
cd ./large/full_training/
unzip epoch_70_best_ckpt.zip
rm epoch_70_best_ckpt.zip