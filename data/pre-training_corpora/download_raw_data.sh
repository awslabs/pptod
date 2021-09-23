gdown "https://drive.google.com/u/0/uc?export=download&confirm=o9iZ&id=1EnGX0UF4KW6rVBKMF3fL-9Q2ZyFKNOIy"
unzip TODBERT_dialog_datasets.zip
mv dialog_datasets raw_data
rm TODBERT_dialog_datasets.zip
gdown "https://drive.google.com/u/1/uc?id=1GOS019dAQbcvbgnKeFTc4LTdHIthyy-9&export=download"
unzip raw_intent_classification_data.zip
mv raw_intent_classification_data ./raw_data
rm raw_intent_classification_data.zip
rm -r __MACOSX