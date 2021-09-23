CUDA_VISIBLE_DEVICES=4,5,6,7 python pretrain.py\
    --dataset_prefix_path ../data/pre-training_corpora/tokenized_pretraining_corpora/\
    --model_name t5-small\
    --batch_size_per_gpu 32\
    --number_of_gpu 4\
    --gradient_accumulation_steps 1\
    --save_steps 5000\
    --save_path ../checkpoints/\
    --save_ckpt_name small

