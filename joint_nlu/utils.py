# -*- coding: utf-8 -*-
"""
This utility module provides a collection of functions designed to facilitate data preparation and
processing tasks for natural language understanding (NLU) models. The functions included cover a
range of operations such as converting sentences to a specific annotation format, calculating
accuracy metrics, and manipulating label sequences.

The module's primary focus is to transform raw text data into formats suitable for training
machine learning models, specifically those involved in intent classification and slot filling
tasks. Additionally, it includes functions to evaluate the performance of NLU models by comparing
predicted outputs against true labels.

The functions in this module are intended to be used in a pre-processing pipeline where text data
is annotated, formatted, and evaluated. They handle the intricacies of NLU data manipulation,
allowing for streamlined data preparation workflows.

Functions:
- Conversion to BIO and flat tag annotation schemes.
- Semantic accuracy computation for joint intent and slot predictions.
- Label sequence padding and truncation to uniform lengths.
- IOB token extraction from datasets.

Note:
The module does not need to be initialized and its functions can be directly applied to data as
needed. It is recommended to use these utilities in the context of a larger data processing
pipeline to ensure optimal model performance.
"""
from typing import List, Dict
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def convert_to_bio(sentence):
    """
    Converts a sentence with annotated slots into the BIO (Beginning, Inside, Outside) tagging
    format.

    The function expects the input sentence to have slots marked with square brackets.
    For example, the input "Book a [flight] from [city : New York] to [city : Los Angeles]"
    would be converted to the BIO format as "O O B-flight O B-city I-city O O B-city I-city".

    Parameters:
    - sentence (str): A string representing the sentence with words and annotated slots.

    Returns:
    - str: A string where each word is followed by a space and a BIO tag indicating whether the
        word is Outside any slot (O), at the Beginning of a slot (B-), or Inside a slot (I-).

    Example:
    >>> sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
    >>> convert_to_bio(sentence)
    'O O B-flight O B-city I-city O O B-city I-city'

    Note:
    - The function assumes that slots are well-formed and correctly annotated.
    - Words are separated by spaces, and slots are indicated by square brackets.
    - The colon ':' within brackets is used to separate the slot type from its value.
    - Tags are generated in lowercase regardless of the case of the slot type in the input.
    """
    bio = ""
    in_slot = False
    slot = ""
    raw_sentence = ""
    for word in sentence.split(' '):
        word = word.strip()
        if word.startswith('['):
            in_slot = True
            b_slot = True
            slot = word[1:].lower()
        elif word == ':' and in_slot:
            continue
        elif word.endswith(']'):
            in_slot = False
            if b_slot:
                b_slot = False
                bio += "B-" + slot.lower() + " "
            else:
                bio += "I-" + slot.lower() + " "
            raw_sentence += word[:-1] + " "
        elif in_slot:
            if b_slot:
                b_slot = False
                bio += "B-" + slot.lower() + " "
            else:
                bio += "I-" + slot.lower() + " "
            raw_sentence += word + " "
        else:
            bio += "O "
            raw_sentence += word + " "

    return bio.strip()


def convert_to_flattag(sentence):
    """
    Converts a sentence with annotated slots into a flat tagging format.

    This function processes a sentence where slots are annotated with square brackets
    and assigns a single tag to all words belonging to a slot. Unlike the BIO format,
    this function does not distinguish between the beginning or inside of slots.
    For example, the input "Book a [flight] from [city : New York] to [city : Los Angeles]"
    would be converted to "O O flight O city city O O city city".

    Parameters:
    - sentence (str): A string representing the sentence with words and annotated slots.

    Returns:
    - str: A string where each word is followed by a space and a tag indicating its slot type,
           or 'O' if it is not part of any slot.

    Example:
    >>> sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
    >>> convert_to_flattag(sentence)
    'O O flight O city city O O city city'

    Note:
    - The function assumes that slots are well-formed and correctly annotated.
    - Words are separated by spaces, and slots are indicated by square brackets.
    - The colon ':' within brackets is used to separate the slot type from its value.
    - All slot tags are converted to lowercase.
    """
    flattag = ""
    in_slot = False
    slot = ""
    raw_sentence = ""
    for word in sentence.split(' '):
        word = word.strip()
        if word.startswith('['):
            in_slot = True
            b_slot = True
            slot = word[1:].lower()
        elif word == ':' and in_slot:
            continue
        elif word.endswith(']'):
            in_slot = False
            if b_slot:
                b_slot = False
                flattag += slot.lower() + " "
            else:
                flattag += slot.lower() + " "
            raw_sentence += word[:-1] + " "
        elif in_slot:
            if b_slot:
                b_slot = False
                flattag += slot.lower() + " "
            else:
                flattag += slot.lower() + " "
            raw_sentence += word + " "
        else:
            flattag += "O "
            raw_sentence += word + " "

    return flattag.strip()


def get_all_iob_tokens(dataset):
    """
    Extracts a unique list of IOB tokens from a dataset.

    This function iterates over a collection of sentences with tokens labeled
    in the IOB format (Inside, Outside, Beginning) and compiles a unique list
    of all IOB tokens present across the dataset.

    Parameters:
    - dataset (list of str): A list of strings, with each string representing
                             a sentence where words are tagged in IOB format.

    Returns:
    - list: A list of unique IOB tokens found in the dataset.

    Example:
    >>> dataset = ["B-person O B-location I-location", "O B-person O O"]
    >>> get_all_iob_tokens(dataset)
    ['B-person', 'O', 'B-location', 'I-location']

    Note:
    - The dataset is expected to be a list where each element is a string
      of space-separated IOB tokens.
    - This function does not preserve the order of appearance; it only ensures uniqueness.
    """
    uniq_iob = []
    for x in dataset:
        for iob_token in x.split(' '):
            if not iob_token in uniq_iob:
                uniq_iob.append(iob_token)
    return uniq_iob


# Configuration for each dataset
DATASET_CONFIGS = {
    "AmazonScience/massive": {
        "keep_columns": ["utt", "intent", "annot_utt", "ner_tags"],
        "rename_column": {"utt": "text"},
        "process_function": convert_to_flattag
    },
    # Add other datasets here...
}

def preprocess_dataset(dataset_id, dataset_configs):
    # Get the configuration for the dataset
    config = DATASET_CONFIGS.get(dataset_id)
    if not config:
        raise ValueError(f"No configuration found for dataset: {dataset_id}")

    # process individuell datasets
    proc_lan_dataset_list=[]
    iob = []

    for lang in dataset_configs:
        # load dataset for language
        lang_ds = load_dataset(dataset_id, lang)
        # only keep the specified columns
        lang_ds = lang_ds.remove_columns([col for col in lang_ds["train"].column_names
                                          if col not in config["keep_columns"]])

        # rename the columns as specified
        for old_name, new_name in config["rename_column"].items():
            lang_ds = lang_ds.rename_column(old_name, new_name)

        # Get the function to process the dataset
        process_function = config["process_function"]

        iob_uniq = get_all_iob_tokens([process_function(s) for s in lang_ds["train"]["annot_utt"]
                                           + lang_ds["test"]["annot_utt"]
                                           + lang_ds["validation"]["annot_utt"]])
        iob = ClassLabel(num_classes=len(iob_uniq), names=iob_uniq)

        #tokens
        lang_ds["train"] = lang_ds["train"].add_column("tokens",
                [x.split() for x in lang_ds["train"]["text"]])
        lang_ds["test"] = lang_ds["test"].add_column("tokens",
            [x.split() for x in lang_ds["test"]["text"]])
        lang_ds["validation"] = lang_ds["validation"].add_column("tokens",
                [x.split() for x in lang_ds["validation"]["text"]])

        #iob
        lang_ds["train"] = lang_ds["train"].add_column("iob_tokens",
            [process_function(s).split() for s in lang_ds["train"]["annot_utt"]])
        lang_ds["test"] = lang_ds["test"].add_column("iob_tokens",
            [process_function(s).split() for s in lang_ds["test"]["annot_utt"]])
        lang_ds["validation"] = lang_ds["validation"].add_column("iob_tokens",
            [process_function(s).split() for s in lang_ds["validation"]["annot_utt"]])
        lang_ds["train"] = lang_ds["train"].add_column("iob",
                [iob.str2int(s) for s in lang_ds["train"]["iob_tokens"]])
        lang_ds["test"] = lang_ds["test"].add_column("iob",
                [iob.str2int(s) for s in lang_ds["test"]["iob_tokens"]])
        lang_ds["validation"] = lang_ds["validation"].add_column("iob",
                [iob.str2int(s) for s in lang_ds["validation"]["iob_tokens"]])

        proc_lan_dataset_list.append(lang_ds)


    # concat single splits into one
    train_dataset = concatenate_datasets([ds["train"] for ds in proc_lan_dataset_list])
    eval_dataset = concatenate_datasets([ds["validation"] for ds in proc_lan_dataset_list])
    test_dataset = concatenate_datasets([ds["test"] for ds in proc_lan_dataset_list])

    dataset = DatasetDict({"train": train_dataset, "validation": eval_dataset,
        "test": test_dataset})
    return dataset, iob


def pad_or_truncate_labels(label_ids: List[int], max_length: int) -> List[int]:
    """
    Pads with -100 or truncates the list of label IDs to a maximum length.

    This function takes a list of label IDs and either pads it with -100
    to reach the desired maximum length, or truncates it if it exceeds
    the maximum length. Padding with -100 is a common practice when handling
    labels for token classification tasks in models such as BERT, where -100
    is used to indicate that a token should not be considered for loss calculation.

    Parameters:
    - label_ids (List[int]): A list of integer label IDs.
    - max_length (int): The maximum length to pad or truncate the label_ids list to.

    Returns:
    - List[int]: The list of label IDs adjusted to the maximum length.

    Example:
    >>> label_ids = [1, 2, 3]
    >>> max_length = 5
    >>> pad_or_truncate_labels(label_ids, max_length)
    [1, 2, 3, -100, -100]

    Note:
    - If the length of label_ids is less than max_length, -100 is appended until
      the list reaches max_length.
    - If the length of label_ids is greater than max_length, the list is truncated.
    """
    # Pad or truncate labels to match max_length
    if len(label_ids) < max_length:
        label_ids += [-100] * (max_length - len(label_ids))
    elif len(label_ids) > max_length:
        label_ids = label_ids[:max_length]
    return label_ids


def tokenize_and_process_labels(examples: Dict, tokenizer) -> Dict:
    # Tokenize all examples
    tokenized_inputs = tokenizer(examples["annot_utt"], truncation=True, padding="max_length",
                                 max_length=512)

    # Initialize list to store label ids
    labels = []

    # Process labels
    for i, label in enumerate(examples["iob"]):
        label_ids = []
        for word, lab in zip(examples["tokens"][i], label):
            # Tokenize each word and label pair
            word_ids = tokenizer(word)["input_ids"]
            label_ids.extend([lab] * len(word_ids))

        # Pad or truncate labels to match max_length
        label_ids = pad_or_truncate_labels(label_ids, 512)

        # Append label ids to the list
        labels.append(label_ids)  # Corrected indentation

    # Add processed labels to tokenized_inputs
    tokenized_inputs["labels"] = labels

    # Process iob
    iobs = []
    for iob in examples["iob"]:
        padded_iob = pad_or_truncate_labels(iob, 512)
        iobs.append(padded_iob)

    tokenized_inputs["iob"] = iobs

    return tokenized_inputs

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
    for i in enumerate(intent_true):
        if intent_true[i] == intent_pred[i]:
            if slot_true[i] == slot_pred[i]:
                correct_count += 1

    return correct_count / len(intent_true)


def compute_metrics(eval_pred):
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
