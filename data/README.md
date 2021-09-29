# Pre-training Corpora Preparation:

We thank the [Salesforce team](https://github.com/jasonwu0731/ToD-BERT) for gathering up the publicly available human-written multi-turn dialogue corpora. We download the raw data from their released google drive link. To run the pre-training data preparation scripts, please first install gdown library as:
```yaml
pip3 install gdown
```

Then, run the following commands.
```yaml
cd pre-training_corpora
chmod +x ./download_raw_data.sh
./download_raw_data.sh

chmod +x ./process_pre-training_data.sh
./process_pre-training_data.sh

chmod +x ./post-processing_pre-training_data.sh
./post-processing_pre-training_data.sh
```




# Preparation of Benchmark TOD Task Datasets:
## 1. MultiWOZ Data:
The MultiWOZ dataset is used for both end-to-end task-oriented dialogue modelling and dialogue state tracking tasks.
### (1) Preparation:
To acquire the processed dataset, you can run the following commands. 
```yaml
cd ./multiwoz/
chmod +x ./data_preparation.sh
./data_preparation.sh
```
Take a coffee, this process will take around 60 minutes.

### (2) Data Format:
```json
[
    {
        "dial_id": "PMUL1170",
        "user": "i need to take a train out of cambridge , i will be leaving town on wednesday .",
        "resp": "there are [value_choice] trains out of [value_departure] on [value_day] . do you have a departure time in mind ?",
        "bspn": "[train] day wednesday departure cambridge",
        "aspn": "[train] [inform] choice departure day [request] leave",
        "turn_num": 0,
        "db": "[db_3]",
    },
    {
        "dial_id": "PMUL1170",
        "user": "<sos_u> i would like to go to peterborough and leave after 12:45 , i have to attend a meeting beforehand . <eos_u>",
        "resp": "<sos_r> [value_id] leaves at [value_leave] on [value_day] . will that work for you ? <eos_r>",
        "bspn": "<sos_b> [train] day wednesday departure cambridge leave 12:45 destination peterborough <eos_b>",
        "aspn": "<sos_a> [train] [inform] day leave id <eos_a>",
        "turn_num": 1,
        "db": "<sos_db> [db_3] <eos_db>",
    },
    ...
]
```
We use json to store the data. Each dialogue session is represented as a list of turns. Each turn is represented as a dictionary that contains the following fields:

* **dial_id** - The unique ID for the dialogue session instance. 
* **user** - The user's utterance.
* **resp** - The delexicalized reference system response.
* **bspn** - The belief state.
* **aspn** - The system action.
* **turn_num** - This argument indicates the turn position in the dialogue session, e.g., if turn_num = 0 means this is the very first turn in the whole dialogue session.
* **db** - The database query result.

## 2. Banking77 Data:
The Banking77 dataset is used for intent classification task.
### (1) Preparation:
To get the dataset, you can run the following commands.
```yaml
cd ./banking77/
chmod +x ./banking77_preparation.sh
./banking77_preparation.sh
```
### (2) Data Format:
Each data sample consists of two parts: (1) the user's utterance and (2) the user's intent class label. An example is provided in the following.
```yaml
The user's uttrance: How much can I top-up to my card at a time?
The user's intent class label: [top_up_limits]
```
