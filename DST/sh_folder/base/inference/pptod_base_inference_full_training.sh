CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-base\
    --pretrained_path ../../../ckpt/base/full_training/\
    --output_save_path ../../../inference_result/base/full_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 32