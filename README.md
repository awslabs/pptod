## Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System
**Authors**: Yixuan Su, Lei Shu, Elman Mansimov, Arshit Gupta, Deng Cai, Yi-An Lai, and Yi Zhang

Code of our PPTOD paper: [Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System](https://arxiv.org/pdf/2109.14739.pdf)

### Introduction:
Pre-trained language models have been recently shown to benefit task-oriented dialogue (TOD) systems. Despite their success, existing methods often formulate this task as a cascaded generation problem which can lead to error accumulation across different sub-tasks and greater data annotation overhead. In this study, we present PPTOD, a unified model that seamlessly supports both task-oriented dialogue understanding and response generation in a plug-and-play fashion. In addition, we introduce a new dialogue multi-task pre-training strategy that allows the model to learn the primary TOD task completion skills from heterogeneous dialog corpora. We extensively test our model on three benchmark TOD tasks, including end-to-end dialogue modelling, dialogue state tracking, and intent classification. Results show that PPTOD creates new state-of-the-art on all evaluated tasks in both full training and low-resource scenarios. Furthermore, comparisons against previous SOTA methods show that the responses generated by PPTOD are more factually correct and semantically coherent as judged by human annotators.

![Alt text](https://github.com/awslabs/pptod/blob/main/overview.png)

### Main Results:
The following table shows our models performances on end-to-end dialogue modelling (Inform, Success, BLEU, and Combined Score) on MultiWOZ 2.0. It also shows the dialogue state tracking (DST) results on MultiWOZ 2.0 and intent classification accuracy on Banking77.

|               | Inform        |Success|BLEU|Combined Score|DST Joint Accuracy|Intent Classification Accuracy|
| :-------------: |:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|PPTOD-small |87.80|75.30 | **19.89**|101.44|51.50|93.27|
| PPTOD-base|**89.20**| **79.40**|18.62 |**102.92**|53.37|93.86|
| PPTOD-large|82.60| 74.10|19.21 |97.56|**53.89**|**94.08**|


### Citation:
If you find our paper and resources useful, please kindly cite our paper:

```bibtex
@article{su2021multitask,
   author = {Yixuan Su and
             Lei Shu and
             Elman Mansimov and
             Arshit Gupta and
             Deng Cai and
             Yi{-}An Lai and
             Yi Zhang},
   title     = {Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System},
   journal   = {CoRR},
   volume    = {abs/2109.14739},
   year      = {2021},
   url       = {https://arxiv.org/abs/2109.14739},
   eprinttype = {arXiv},
   eprint    = {2109.14739}
}
```

## Example Usage:
In the following, we provide an example of how to use the pre-trained PPTOD to address different TOD tasks (**without fine-tuning on any downstream task**). We assume you have downloaded the pptod-small checkpoint (you can find instructions below).
```python
# load the pre-trained PPTOD-small
import torch
from transformers import T5Tokenizer
model_path = r'./checkpoints/small/'
tokenizer = T5Tokenizer.from_pretrained(model_path)
from E2E_TOD.modelling.T5Model import T5Gen_Model
from E2E_TOD.ontology import sos_eos_tokens
special_tokens = sos_eos_tokens
model = T5Gen_Model(model_path, tokenizer, special_tokens, dropout=0.0, 
        add_special_decoder_token=True, is_training=False)
model.eval()
```
```python
# prepare some pre-defined tokens and task-specific prompts
sos_context_token_id = tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
eos_context_token_id = tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
pad_token_id, sos_b_token_id, eos_b_token_id, sos_a_token_id, eos_a_token_id, \
sos_r_token_id, eos_r_token_id, sos_ic_token_id, eos_ic_token_id = \
tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
'<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>', '<sos_d>', '<eos_d>'])
bs_prefix_text = 'translate dialogue to belief state:'
bs_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
da_prefix_text = 'translate dialogue to dialogue action:'
da_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(da_prefix_text))
nlg_prefix_text = 'translate dialogue to system response:'
nlg_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(nlg_prefix_text))
ic_prefix_text = 'translate dialogue to user intent:'
ic_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ic_prefix_text))
```
```python
# an example dialogue context
dialogue_context = "<sos_u> can i reserve a five star place for thursday night at 3:30 for 2 people <eos_u> <sos_r> i'm happy to assist you! what city are you dining in? <eos_r> <sos_u> seattle please. <eos_u>"
context_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dialogue_context))
```
```python
# predict belief state 
input_id = bs_prefix_id + [sos_context_token_id] + context_id + [eos_context_token_id]
input_id = torch.LongTensor(input_id).view(1, -1)
x = model.model.generate(input_ids = input_id, decoder_start_token_id = sos_b_token_id,
            pad_token_id = pad_token_id, eos_token_id = eos_b_token_id, max_length = 128)
print (model.tokenized_decode(x[0]))
# the predicted result is
# <sos_b> [restaurant] rating five star date thursday night start time 3:30 number of people 2 city seattle <eos_b>
```
```python
# the 

```
 
### 1. Environment Setup:
```yaml
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. PPTOD Checkpoints:
You can download checkpoints of PPTOD with different configurations here.

| PPTOD-small       | PPTOD-base          | PPTOD-large  |
| :-------------: |:-------------:| :-----:|
| [here](https://pptod.s3.amazonaws.com/Pretrain/small.zip)      | [here](https://pptod.s3.amazonaws.com/Pretrain/base.zip) | [here](https://pptod.s3.amazonaws.com/Pretrain/large.zip) |

To use PPTOD, you should download the checkpoint you want and unzip it in the ./checkpoints directory.

Alternatively, you can run the following commands to download the PPTOD checkpoints.

#### (1) Downloading Pre-trained PPTOD-small Checkpoint:
```yaml
cd checkpoints
chmod +x ./download_pptod_small.sh
./download_pptod_small.sh
```

#### (2) Downloading Pre-trained PPTOD-base Checkpoint:
```yaml
cd checkpoints
chmod +x ./download_pptod_base.sh
./download_pptod_base.sh
```

#### (3) Downloading Pre-trained PPTOD-large Checkpoint:
```yaml
cd checkpoints
chmod +x ./download_pptod_large.sh
./download_pptod_large.sh
```

### 3. Data Preparation:
The detailed instruction for preparing the pre-training corpora and the data of downstream TOD tasks are provided in the ./data folder.

### 4. Dialogue Multi-Task Pre-training:
To pre-train a PPTOD model from scratch, please refer to details provided in ./Pretraining directory.

### 5. Benchmark TOD Tasks:
#### (1) End-to-End Dialogue Modelling:
To perform End-to-End Dialogue Modelling using PPTOD, please refer to details provided in ./E2E_TOD directory. 

#### (2) Dialogue State Tracking:
To perform Dialogue State Tracking using PPTOD, please refer to details provided in ./DST directory. 

#### (3) Intent Classification:
To perform Intent Classification using PPTOD, please refer to details provided in ./IC directory. 


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

