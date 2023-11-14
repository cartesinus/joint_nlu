# JointNLU

JointNLU is a Python library for training a Joint Natural Language Understanding (NLU) model using the Hugging Face Transformers library. The Joint NLU model aims to simultaneously perform intent classification and slot filling tasks in a single forward pass.

## JointNLUModel

The `JointNLUModel` is a PyTorch module for the Joint NLU task which uses a pretrained transformer model as its encoder, and includes separate classification layers for intents and slots.

### Attributes

- `encoder`: A transformer model from Hugging Face's transformers library pre-trained on the given model_name.
- `intent_id2label`: A mapping from intent IDs to intent labels.
- `intent_label2id`: A mapping from intent labels to intent IDs.
- `slot_id2label`: A mapping from slot IDs to slot labels.
- `slot_label2id`: A mapping from slot labels to slot IDs.
- `intent_classifier`: A linear layer for intent classification tasks, projecting encoder outputs to intent space.
- `slot_classifier`: A linear layer for slot filling tasks, projecting encoder outputs to slot space.

### Usage

```python
from joint_nlu import JointNLUModel

model = JointNLUModel('bert-base-uncased', num_intents=10, num_slots=20, intent_id2label, intent_label2id, slot_id2label, slot_label2id)
```

## Training

To train the model, use the `CustomTrainer` class. This class extends the Hugging Face's Trainer class, customized to compute losses for both intent classification and slot filling tasks. It includes methods to handle the custom data collator and implements hooks for additional functionality during the training process.

```python
from joint_nlu import CustomTrainer, DataCollatorForJointIntentAndSlotFilling

data_collator = DataCollatorForJointIntentAndSlotFilling(tokenizer)
trainer = CustomTrainer(model=model, args=training_args, data_collator=data_collator, ...)
trainer.train()
```

## Dataset Filtering

JointNLU supports dataset filtering based on specified column values. This feature allows users to include or exclude specific data points from the training process based on the values in certain columns (e.g., scenario or intent).

To use dataset filtering, update the configuration file to include filters for the desired columns. The filters are specified as a dictionary, where keys are column names, and values are lists of accepted values for each column.

```json
{
  "filters": {
    "scenario": ["calendar", "travel"],
    "intent": ["book_flight", "schedule_meeting"]
  }
}
```

In this example, the dataset will be filtered to include only the rows where the scenario column contains either "calendar" or "travel", and the intent column contains either "book_flight" or "schedule_meeting".

## Pushing to Hugging Face Hub

To enable easy sharing and deployment of your trained Joint NLU model, our framework supports direct uploading to the Hugging Face Hub. To use this feature, ensure that you have git-lfs installed and that you are logged in to the Hugging Face Hub.

In your training configuration (config.json), set the "push_to_hub" option to true under the "trainer" section. This instructs the training script to automatically push your model to the specified repository on the Hugging Face Hub upon successful training and evaluation.

```json
"trainer": {
    ...
    "push_to_hub": true
    ...
}
```

When this option is enabled, and provided you have sufficient permissions for the specified repository, the model will be pushed to the Hugging Face Hub at the end of the training process. This facilitates easy access, version control, and sharing of your NLU model within the community.

### Temporary Authentication Method

**Warning:** The following method involves using an environment variable to pass your Hugging Face token. This method should be considered temporary and may not be the safest option. Ensure your development environment is secure before using this approach. A more secure implementation will be provided in future updates.

To push models to the Hugging Face Hub using an environment variable, follow these steps before running train.py script.

```python
import os
from huggingface_hub import notebook_login, HfFolder

notebook_login()
token = HfFolder.get_token()
os.environ['HF_TOKEN'] = token
```

Run the train.py script normally. The script will automatically use the token from the environment variable for authentication when pushing the model to the Hugging Face Hub.

Note: This method currently places the responsibility of token security on the user. It's recommended to use this method only if you are confident in the security of your development environment.
