wget -r --no-parent https://github.com/PolyAI-LDN/task-specific-datasets/archive/refs/heads/master.zip
mv ./github.com/PolyAI-LDN/task-specific-datasets/archive/refs/heads/master.zip .
rm -r github.com
unzip master.zip
rm master.zip
mv ./task-specific-datasets-master/banking_data/categories.json .
mv ./task-specific-datasets-master/banking_data/test.csv .
mv ./task-specific-datasets-master/banking_data/train.csv .
rm -r task-specific-datasets-master