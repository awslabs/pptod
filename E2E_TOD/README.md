# End-to-End Task-Oriented Dialogue Modelling

### 1. Model checkpoints of the full training experiment:

|               | Inform        |Success|BLEU|Combined Score|
| :-------------: |:-------------:|:-----:|:-----:|:-----:|
|[small](https://pptod.s3.amazonaws.com/E2E/epoch_9_best_ckpt.zip) |87.80|75.30 | **19.89**|101.44|
| [base](https://pptod.s3.amazonaws.com/E2E/epoch_6_best_ckpt.zip)|**89.20**| **79.40**|18.62 |**102.92**|
| [large](https://pptod.s3.amazonaws.com/E2E/epoch_5_best_ckpt.zip)|82.60| 74.10|19.21 |97.56|


Download and unzip the pretrained checkpoint under the "./ckpt/X/full_training/" directory. 

Here, X is in ['small', 'base', 'large'].

Alternatively, you can run the following commands to download the trained models.

#### (1) Downloading Pre-trained PPTOD-small Checkpoint:
```yaml
cd ckpt
chmod +x ./download_E2E_TOD_PPTOD_small.sh
./download_E2E_TOD_PPTOD_small.sh
```

#### (2) Downloading Pre-trained PPTOD-base Checkpoint:
```yaml
cd ckpt
chmod +x ./download_E2E_TOD_PPTOD_base.sh
./download_E2E_TOD_PPTOD_base.sh
```

#### (3) Downloading Pre-trained PPTOD-large Checkpoint:
```yaml
cd ckpt
chmod +x ./download_E2E_TOD_PPTOD_large.sh
./download_E2E_TOD_PPTOD_large.sh
```

### 2. Perform inference using pretrained checkpoints:
```yaml
cd ./sh_folder/X/inference/ 
chmod +x ./pptod_X_full_training_inference.sh
./pptod_X_full_training_inference.sh
```
**[Note]** If you need our model's predictions, we also provide the predicted results of our models under the "./inference_result/X/full_training/" directory.

Here, X is in ['small', 'base', 'large'] and some key parameters are described below:

```yaml
--use_db_as_input: Whether use DB result as input. It should be set as the same value as the 
                   --use_db_as_input argument in the training script.
                   
--pretrained_path: The path that stores the model from training. Should be the same value as the 
                   --ckpt_save_path argument in the training script.
                   
--output_save_path: The directory to save the predicted result.
```

### 3. Training
To train a new model, you can use the provided scripts.

```yaml
cd ./sh_folder/X/training/ 
chmod +x ./pptod_X_full_training.sh
./pptod_X_full_training.sh
```
Here, X is in ['small', 'base', 'large'] and some key parameters are described below:

```yaml
--use_db_as_input: Whether include DB search result when generating the dialogue act and system response. 
                   (With DB input, the model performs better. Without DB input, the model can generates 
                   the belief state, dialogue act, and system response in a fully paralleled way.)
                   
--train_data_ratio: The portion of training data used, default value is 1.0, meaning 100% of training data.
                    For different few-shot settings, you can set this argument to different values. For 
                    example, when train_data_ratio equals to 0.01, the model is trained with 1% of training data.
                    
--gradient_accumulation_steps: How many forward computations between two gradient updates.

--number_of_gpu: Number of avaliable GPUs.

--batch_size_per_gpu: The batch size for each GPU.
```
**[Note 1]** The few-shot training samples are randomly selected, thus the results from different runs may not be the same.

**[Note 2]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We recommend
you to set the actual batch size value as 128.
