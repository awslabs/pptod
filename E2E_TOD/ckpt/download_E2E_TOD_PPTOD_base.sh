gdown "https://drive.google.com/u/1/uc?export=download&confirm=Ojzi&id=172L5_GNFuWjcfBVdI4Tto7dAiYL4bYRE"
mv epoch_6_best_ckpt.zip ./base/full_training/
cd ./base/full_training/
unzip epoch_6_best_ckpt.zip
rm epoch_6_best_ckpt.zip