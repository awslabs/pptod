CUDA_VISIBLE_DEVICES=4,5,6,7 python ../../../inference.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-large\
    --pretrained_path ../../../ckpt/large/full_training/\
    --output_save_path ../../../inference_result/large/full_training/\
    --number_of_gpu 4\
    --batch_size_per_gpu 8