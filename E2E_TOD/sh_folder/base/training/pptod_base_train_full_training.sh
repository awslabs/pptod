CUDA_VISIBLE_DEVICES=4,5 python ../../../learn.py\
    --data_path_prefix ../../../../data/multiwoz/data/\
    --model_name t5-base\
    --pretrained_path ../../../../checkpoints/base/\
    --ckpt_save_path ../../../ckpt/base/full_training/\
    --epoch_num 60\
    --gradient_accumulation_steps 16\
    --number_of_gpu 2\
    --batch_size_per_gpu 4