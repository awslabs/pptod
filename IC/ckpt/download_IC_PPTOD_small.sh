wget https://pptod.s3.amazonaws.com/IC/epoch_50_best_ckpt.zip
mv epoch_50_best_ckpt.zip ./small/full_training/
cd ./small/full_training/
unzip epoch_50_best_ckpt.zip
rm epoch_50_best_ckpt.zip