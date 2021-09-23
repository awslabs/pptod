gdown "https://drive.google.com/u/1/uc?export=download&confirm=vdoi&id=1kH-6Vbe2u8eT-QWIOYM_eS_6PbsNbp8q"
mv epoch_5_best_ckpt.zip ./large/full_training/
cd ./large/full_training/
unzip epoch_5_best_ckpt.zip
rm epoch_5_best_ckpt.zip