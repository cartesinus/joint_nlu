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

Please refer to the source code and comments for more detailed information.
