wget https://pptod.s3.amazonaws.com/DST/epoch_10_best_ckpt.zip
mv epoch_10_best_ckpt.zip ./large/full_training/
cd ./large/full_training/
unzip epoch_10_best_ckpt.zip
rm epoch_10_best_ckpt.zip
