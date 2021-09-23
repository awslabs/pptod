CUDA_VISIBLE_DEVICES=4,5 python ../../../inference_pptod.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --model_name t5-small\
    --pretrained_path ../../../ckpt/small/full_training/\
    --output_save_path ../../../inference_result/small/full_training/\
    --number_of_gpu 2\
    --batch_size_per_gpu 64