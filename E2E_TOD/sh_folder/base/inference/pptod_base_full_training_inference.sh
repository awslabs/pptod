CUDA_VISIBLE_DEVICES=6,7 python ../../../inference_pptod.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --model_name t5-base\
    --pretrained_path ../../../ckpt/base/full_training/\
    --output_save_path ../../../inference_result/base/full_training/\
    --number_of_gpu 2\
    --batch_size_per_gpu 32