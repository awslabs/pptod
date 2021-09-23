gdown "https://drive.google.com/u/1/uc?export=download&confirm=F5J6&id=1AYB5snpx8uqKeZOF5RDr_VUskCAGWFVG"
mv epoch_50_best_ckpt.zip ./small/full_training/
cd ./small/full_training/
unzip epoch_50_best_ckpt.zip
rm epoch_50_best_ckpt.zip