gdown "https://drive.google.com/u/1/uc?export=download&confirm=bEC1&id=15KIaCd0Y8EGYYW4x0cBH2--fwAJ-AxCI"
mv epoch_3_best_ckpt.zip ./small/full_training/
cd ./small/full_training/
unzip epoch_3_best_ckpt.zip
rm epoch_3_best_ckpt.zip