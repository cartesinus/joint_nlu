# -*- coding: utf-8 -*-
"""
CustomTrainer(Trainer): An extension of Hugging Face's Trainer class, customized to compute
losses for both intent classification and slot filling tasks. It includes methods to handle the
custom data collator and implements hooks for additional functionality during the training
process.
"""
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer
from .data_collator import DataCollatorForJointIntentAndSlotFilling


class CustomTrainer(Trainer):
    """
    A custom trainer class that extends the Hugging Face Trainer to compute the loss for
    a joint intent classification and slot filling model.

    This trainer handles the training process for models that output both intent and slot
    predictions and computes the loss for each output separately. The losses are then
    combined to create a total loss for the backpropagation step.

    Methods:
        compute_loss: Computes the loss for intent classification and slot filling tasks.
        get_train_dataloader: Returns a DataLoader for the training data with a custom collate
            function.
        on_epoch_end: Custom actions to perform at the end of each epoch.
        get_eval_dataloader: Returns a DataLoader for the evaluation data with a custom collate
            function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for both intent classification and slot filling tasks.

        Parameters:
            model (nn.Module): The model being trained.
            inputs (dict): A dictionary of inputs to the model. Must include 'intent' and 'iob' for
                labels.
            return_outputs (bool, optional): If set to True, the method also returns model outputs.
                Defaults to False.

        Returns:
            tuple or torch.Tensor: A tuple containing the loss and model outputs if return_outputs
                is True. Otherwise, returns a torch.Tensor representing the loss.
        """
        labels = inputs.pop("intent")
        slot_labels = inputs.pop("iob")

        # intent_logits, slot_logits = model(**inputs)
        outputs = model(**inputs)
        intent_logits = outputs["intent_logits"]
        slot_logits = outputs["slot_logits"]

        intent_loss = F.cross_entropy(intent_logits, labels)
        slot_loss = F.cross_entropy(slot_logits.view(-1, model.config.num_slots),
                                                     slot_labels.view(-1))

        loss = intent_loss + slot_loss  # You can also weigh these losses

        outputs = {"intent_logits": intent_logits, "slot_logits": slot_logits,
                   "intent_labels": labels, "slot_labels": slot_labels
                   }
        return [loss, outputs] if return_outputs else loss

    def get_train_dataloader(self):
        """
        Overrides the Trainer method to return a DataLoader with a custom collate function
        for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training data.
        """
        # Override to modify collate_fn
        return DataLoader(
            self.train_dataset,
            collate_fn=DataCollatorForJointIntentAndSlotFilling(tokenizer=self.tokenizer),
            batch_size=self.args.train_batch_size,
            shuffle=True,
            drop_last=True  # Skip the last incomplete batch
        )

    def get_eval_dataloader(self, eval_dataset):
        """
        Overrides the Trainer method to return a DataLoader with a custom collate function
        for the evaluation dataset.

        Parameters:
            eval_dataset: The dataset to evaluate on.

        Returns:
            DataLoader: The DataLoader object for the evaluation data.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # Override to modify collate_fn
        return DataLoader(
            eval_dataset,
            collate_fn=DataCollatorForJointIntentAndSlotFilling(tokenizer=self.tokenizer),
            batch_size=self.args.eval_batch_size,
        )
