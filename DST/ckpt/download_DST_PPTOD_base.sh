gdown "https://drive.google.com/u/1/uc?export=download&confirm=JR9v&id=108vA2eLBt6hPBT_9jXC5_lX0H2V2iwCq"
mv epoch_4_best_ckpt.zip ./base/full_training/
cd ./base/full_training/
unzip epoch_4_best_ckpt.zip
rm epoch_4_best_ckpt.zip