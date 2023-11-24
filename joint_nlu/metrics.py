# -*- coding: utf-8 -*-
"""
This utility module provides a collection of functions designed to evaluate JointNLU model.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def semantic_accuracy(intent_true, intent_pred, slot_true, slot_pred):
    """
    Calculate the semantic accuracy for intent and slot predictions.

    Semantic accuracy is a metric that considers a prediction to be correct
    only if both the intent and slot predictions match their respective true labels.
    This function computes the semantic accuracy across a batch of predictions,
    counting a prediction as correct only if both intent and slot are accurately predicted.

    Parameters:
    - intent_true (List[int]): A list of true intent labels.
    - intent_pred (List[int]): A list of predicted intent labels.
    - slot_true (List[List[int]]): A list of lists containing true slot labels for each token.
    - slot_pred (List[List[int]]): A list of lists containing predicted slot labels for each token.

    Returns:
    - float: The semantic accuracy as a proportion of correct predictions over the total number of
        predictions.

    Example:
    >>> intent_true = [1, 2, 3]
    >>> intent_pred = [1, 2, 4]
    >>> slot_true = [[1, 0], [2, 2], [3, 0]]
    >>> slot_pred = [[1, 0], [2, 2], [4, 0]]
    >>> semantic_accuracy(intent_true, intent_pred, slot_true, slot_pred)
    0.6667  # 2 out of 3 predictions are correct for both intent and slots

    Note:
    - The function assumes that the length of the true and predicted lists for both intents and
        slots are equal.
    - The slot_true and slot_pred lists are expected to be aligned with the intent_true and
        intent_pred lists.
    """
    correct_count = 0
    for i, intent_label in enumerate(intent_true):
        if intent_label == intent_pred[i]:
            if slot_true[i] == slot_pred[i]:
                correct_count += 1

    return correct_count / len(intent_true)


def compute_metrics(eval_pred):
    """
    Computes metrics for model evaluation including intent accuracy, intent F1 score,
    slot F1 score, and semantic accuracy.

    This function processes the predictions and labels provided during the evaluation
    step of the model training, calculating accuracy and F1 scores for both intents
    and slots. Additionally, it computes the semantic accuracy which combines both
    intent and slot predictions.

    Args:
        eval_pred: A tuple containing model predictions and labels for intents and slots.

    Returns:
        dict: A dictionary containing the computed metrics: intent_accuracy,
              intent_f1_macro, slot_f1, and semantic_accuracy.
    """
    predictions, _ = eval_pred

    intent_predictions = predictions[0]
    slot_predictions = predictions[1]
    intent_labels = predictions[2]
    slot_labels = predictions[3]

    # Intent metrics
    intent_predictions = np.argmax(intent_predictions, axis=1)
    intent_acc = accuracy_score(intent_labels, intent_predictions)
    intent_f1 = f1_score(intent_labels, intent_predictions, average="macro")

    # Slot metrics
    slot_predictions = np.argmax(slot_predictions, axis=2)

    # Filter out 'pad_token_label_id' (-100)
    true_slot_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions.tolist(), slot_labels.tolist())
    ]
    true_slot_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_predictions.tolist(), slot_labels.tolist())
    ]

    mlb = MultiLabelBinarizer()
    true_slot_labels_bin = mlb.fit_transform(true_slot_labels)
    true_slot_predictions_bin = mlb.transform(true_slot_predictions)

    slot_f1 = f1_score(true_slot_labels_bin, true_slot_predictions_bin, average='macro')
    sem_acc = semantic_accuracy(intent_labels, intent_predictions, true_slot_labels,
                                true_slot_predictions)

    return {
        "intent_accuracy": intent_acc,
        "intent_f1_macro": intent_f1,
        "slot_f1": slot_f1,
        "semantic_accuracy": sem_acc,
    }
