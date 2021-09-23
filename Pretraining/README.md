### Code for Dialogue Multi-Task Pre-training

**[Note]** Before pre-training PPTOD, please make sure you have downloaded and proccessed the pre-training corpora as described [here](https://github.com/yxuansu/PPTOD_Private/tree/main/data).

To pre-train PPTOD from script, you can run the following command.

```yaml
chmod +x ./pretraining_pptod_X.sh
./pretraining_pptod_X.sh
```
Here, X is in ['small', 'base', 'large'] and some key parameters are described below:

```yaml
--use_nlu: Whether to include pre-training data that is annotated for NLU task. The default value is True.

--use_bs: Whether to include pre-training data that is annotated for DST task. The default value is True.

--use_da: Whether to include pre-training data that is annotated for POL task. The default value is True.

--use_nlg: Whether to include pre-training data that is annotated for NLG task. The default value is True.

--gradient_accumulation_steps: How many forward computations between two gradient updates.

--number_of_gpu: Number of avaliable GPUs.

--batch_size_per_gpu: The batch size for each GPU.
```

**[Note]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We recommend
you to set the actual batch size value as 128. All PPTOD models are trained on a single machine with 8 x Nvidia V100 GPUs (8 x 32GB memory).
