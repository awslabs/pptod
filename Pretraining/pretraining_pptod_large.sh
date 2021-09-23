CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pretrain.py\
    --dataset_prefix_path ../data/pre-training_corpora/tokenized_pretraining_corpora/\
    --model_name t5-large\
    --batch_size_per_gpu 2\
    --number_of_gpu 8\
    --gradient_accumulation_steps 8\
    --save_steps 5000\
    --save_path ../checkpoints/\
    --save_ckpt_name large

