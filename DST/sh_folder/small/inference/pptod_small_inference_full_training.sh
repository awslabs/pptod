CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-small\
    --pretrained_path ../../../ckpt/small/full_training/\
    --output_save_path ../../../inference_result/small/full_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 128