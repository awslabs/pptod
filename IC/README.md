# Intent Classification

### 1. Model checkpoints for the full training experiment:

|               | PPTOD-small         |PPTOD-base|PPTOD-large|
|:-------------:|:-------------:|:-----:|:-----:|
| Model Checkpoints | [full-train-small](https://pptod.s3.amazonaws.com/IC/epoch_50_best_ckpt.zip) | [full-train-base](https://pptod.s3.amazonaws.com/IC/epoch_66_best_ckpt.zip) |[full-train-large](https://pptod.s3.amazonaws.com/IC/epoch_70_best_ckpt.zip) |
| Classification Accuracy | 93.24% | 93.76% | 94.38% |

Download and unzip the pretrained checkpoint under the "./ckpt/X/full_training/" directory. 

Here, X is in ['small', 'base', 'large'].


#### (1) Downloading Pre-trained PPTOD-small Checkpoint:
```yaml
cd ckpt
chmod +x ./download_IC_PPTOD_small.sh
./download_IC_PPTOD_small.sh
```

#### (2) Downloading Pre-trained PPTOD-base Checkpoint:
```yaml
cd ckpt
chmod +x ./download_IC_PPTOD_base.sh
./download_IC_PPTOD_base.sh
```

#### (3) Downloading Pre-trained PPTOD-large Checkpoint:
```yaml
cd ckpt
chmod +x ./download_IC_PPTOD_large.sh
./download_IC_PPTOD_large.sh
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
--pretrained_path: The path that stores the model from training. Should be the same value as the 
                   --ckpt_save_path argument in the training script.
                   
--save_path: The directory to save the predicted result.
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
--datapoints_per_intent: How many training samples for each intent class. When performing full training, just set
                         it as a large value, e.g. 10000. For different few-shot settings, you can set this argument 
                         to different values. For example, when setting it 10, the model is trained with 10 samples 
                         per intent class.
                    
--gradient_accumulation_steps: How many forward computations between two gradient updates.

--number_of_gpu: Number of avaliable GPUs.

--batch_size_per_gpu: The batch size for each GPU.
```

**[Note 1]** The few-shot training samples are randomly selected, thus the results from different runs may not be the same.

**[Note 2]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. For PPTOD-small
and PPTOD-base, we recommend the actual batch size value is set as 128. For PPTOD-large, the recommended value is 64.

