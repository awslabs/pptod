CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --data_prefix ../../../../data/banking77/\
    --model_name t5-small\
    --format_mode ic\
    --pretrained_path ../../../ckpt/small/full_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 128\
    --save_path ../../../inference_result/small/full_training