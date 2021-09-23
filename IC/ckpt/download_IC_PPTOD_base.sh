gdown "https://drive.google.com/u/1/uc?export=download&confirm=8gZG&id=1OmLQbZpUMsLZowTCMYRnDQ45kkxviPOj"
mv epoch_66_best_ckpt.zip ./base/full_training/
cd ./base/full_training/
unzip epoch_66_best_ckpt.zip
rm epoch_66_best_ckpt.zip