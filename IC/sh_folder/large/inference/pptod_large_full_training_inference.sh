CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --data_prefix ../../../../data/banking77/\
    --model_name t5-large\
    --format_mode bs\
    --pretrained_path ../../../ckpt/large/full_training/\
    --number_of_gpu 1\
    --batch_size_per_gpu 16\
    --save_path ../../../inference_result/large/full_training