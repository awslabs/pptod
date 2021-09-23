CUDA_VISIBLE_DEVICES=0,1 python ../../../learn.py\
    --data_path_prefix ../../../../data/multiwoz/data/multi-woz-fine-processed/\
    --model_name t5-small\
    --pretrained_path ../../../../checkpoints/small/\
    --ckpt_save_path ../../../ckpt/small/full_training/\
    --epoch_num 50\
    --gradient_accumulation_steps 4\
    --number_of_gpu 2\
    --batch_size_per_gpu 16