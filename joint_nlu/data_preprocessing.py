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
from datasets import load_dataset, DatasetDict, ClassLabel


def convert_to_bio(sentence):
    """
    Converts a sentence with annotated slots into the BIO (Beginning, Inside, Outside) tagging
    format.

    The function expects the input sentence to have slots marked with square brackets.
    For example, the input "Book a [flight] from [city : New York] to [city : Los Angeles]"
    would be converted to the BIO format as "o o o o b-city i-city o b-city i-city".

    Parameters:
    - sentence (str): A string representing the sentence with words and annotated slots.

    Returns:
    - str: A string where each word is followed by a space and a BIO tag indicating whether the
        word is Outside any slot (O), at the Beginning of a slot (B-), or Inside a slot (I-).

    Example:
    >>> sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
    >>> convert_to_bio(sentence)
    'o o o o b-city i-city o b-city i-city'

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
        if word.startswith('[') and word.endswith(']'):
            bio += "o "
            raw_sentence += word + " "
        elif word.startswith('['):
            in_slot = True
            b_slot = True
            if word.endswith(':'):
                slot = word[1:-1].lower()
            else:
                slot = word[1:].lower()
        elif word == ':' and in_slot:
            continue
        elif word.endswith(']') or word.endswith('],'):
            in_slot = False
            if b_slot:
                b_slot = False
                bio += "b-" + slot.lower() + " "
            else:
                bio += "i-" + slot.lower() + " "
            raw_sentence += word[:-1] + " "
        elif in_slot:
            if b_slot:
                b_slot = False
                bio += "b-" + slot.lower() + " "
            else:
                bio += "i-" + slot.lower() + " "
            raw_sentence += word + " "
        else:
            bio += "o "
            raw_sentence += word + " "

    return bio.strip()


def convert_to_flattag(sentence):
    """
    Converts a sentence with annotated slots into a flat tagging format.

    This function processes a sentence where slots are annotated with square brackets
    and assigns a single tag to all words belonging to a slot. Unlike the BIO format,
    this function does not distinguish between the beginning or inside of slots.
    For example, the input "Book a [flight] from [city : New York] to [city : Los Angeles]"
    would be converted to "o o flight o city city o o city city".

    Parameters:
    - sentence (str): A string representing the sentence with words and annotated slots.

    Returns:
    - str: A string where each word is followed by a space and a tag indicating its slot type,
           or 'O' if it is not part of any slot.

    Example:
    >>> sentence = "Book a [flight] from [city : New York] to [city : Los Angeles]"
    >>> convert_to_flattag(sentence)
    'o o o o city city o city city'

    Note:
    - The function assumes that slots are well-formed and correctly annotated.
    - Words are separated by spaces, and slots are indicated by square brackets.
    - The colon ':' within brackets is used to separate the slot type from its value.
    - All slot tags are converted to lowercase.
    """
    bio = convert_to_bio(sentence)
    return bio.replace('b-', '').replace('i-', '')


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
    for token in dataset:
        for iob_token in token.split(' '):
            if not iob_token in uniq_iob:
                uniq_iob.append(iob_token)
    return uniq_iob


def apply_filters_to_dataset(dataset, filters):
    """
    Applies filters to a dataset based on specified column values.

    Parameters:
    - dataset (Dataset): The dataset to be filtered.
    - filters (dict): A dictionary where keys are column names and values are lists of
      accepted values for the corresponding column.

    Returns:
    - Dataset: The filtered dataset.
    """
    for filter_col, filter_values in filters.items():
        if isinstance(dataset.features[filter_col], ClassLabel):
            class_mapping = dataset.features[filter_col].names
            filter_indices = [class_mapping.index(val) for val in filter_values
                                                        if val in class_mapping]
            check_value = lambda example: example[filter_col] in filter_indices
        else:
            check_value = lambda example: example[filter_col] in filter_values

        dataset = dataset.filter(check_value)

    return dataset


# Configuration for each dataset
DATASET_CONFIGS = {
    "AmazonScience/massive": {
        "keep_columns": ["utt", "intent", "annot_utt", "ner_tags"],
        "rename_column": {"utt": "text"},
        "iob_column": "annot_utt",
        "process_function": convert_to_bio
    },
    "custom.py": {
        "keep_columns": ["utterance", "intent", "bio"],
        "rename_column": {"utterance": "text"},
        "iob_column": "bio",
    }
}


def preprocess_dataset(dataset_id, dataset_configs, split_ratio, filters=None):
    """
    Preprocesses a multilingual dataset for NLU tasks based on specified configurations and split
    ratios.

    Parameters:
        dataset_id (str): The identifier of the dataset.
        dataset_configs (List[str]): List of configurations for different languages in the dataset.
        split_ratio (dict): Ratios for splitting the dataset into train, valid, and testing sets.

    Returns:
        Tuple[DatasetDict, ClassLabel]: A tuple containing the preprocessed dataset and the IOB
            label class.
    """
    # Get the configuration for the dataset
    config = DATASET_CONFIGS.get(dataset_id)
    if not config:
        raise ValueError(f"No configuration found for dataset: {dataset_id}")

    # Default split ratio if not provided
    default_split_ratio = {
        "train": "100%",
        "validation": "100%",
        "test": "100%"
    }

    # Use the provided split ratio or the default one
    split_ratio = split_ratio if split_ratio else default_split_ratio

    # process individuell datasets
    iob = []

    for lang in dataset_configs:
        split_config = {
            "train": f"train[:{split_ratio['train']}]",
            "validation": f"validation[:{split_ratio['validation']}]",
            "test": f"test[:{split_ratio['test']}]"
        }

        # load dataset for language
        lang_ds_train = load_dataset(dataset_id, lang, split=split_config['train'])
        lang_ds_validation = load_dataset(dataset_id, lang, split=split_config['validation'])
        lang_ds_test = load_dataset(dataset_id, lang, split=split_config['test'])

        # Apply filters if specified
        if filters:
            lang_ds_train = apply_filters_to_dataset(lang_ds_train, filters)
            lang_ds_validation = apply_filters_to_dataset(lang_ds_validation, filters)
            lang_ds_test = apply_filters_to_dataset(lang_ds_test, filters)

        # only keep the specified columns
        lang_ds_train = lang_ds_train.remove_columns(
            [col for col in lang_ds_train.column_names if col not in config["keep_columns"]]
        )
        lang_ds_validation = lang_ds_validation.remove_columns(
            [col for col in lang_ds_validation.column_names if col not in config["keep_columns"]]
        )
        lang_ds_test = lang_ds_test.remove_columns(
            [col for col in lang_ds_test.column_names if col not in config["keep_columns"]]
        )

        # rename the columns as specified
        for old_name, new_name in config["rename_column"].items():
            lang_ds_train = lang_ds_train.rename_column(old_name, new_name)
            lang_ds_validation = lang_ds_validation.rename_column(old_name, new_name)
            lang_ds_test = lang_ds_test.rename_column(old_name, new_name)

        # Get the function to process the dataset
        process_function = config.get("process_function")
        iob_column = config["iob_column"]

        if process_function and iob_column != "bio":
            iob_uniq = get_all_iob_tokens([process_function(s) for s in lang_ds_train[iob_column]
                                           + lang_ds_test[iob_column]
                                           + lang_ds_validation[iob_column]])

            iob = ClassLabel(num_classes=len(iob_uniq), names=iob_uniq)

            lang_ds_train = lang_ds_train.add_column("iob",
                [iob.str2int(process_function(s).split()) for s in lang_ds_train[iob_column]])
            lang_ds_test = lang_ds_test.add_column("iob",
                [iob.str2int(process_function(s).split()) for s in lang_ds_test[iob_column]])
            lang_ds_validation = lang_ds_validation.add_column("iob",
                [iob.str2int(process_function(s).split()) for s in lang_ds_validation[iob_column]])
        else:
            # Directly use the existing IOB column
            iob_uniq = get_all_iob_tokens(list(lang_ds_train[iob_column]
                                           + lang_ds_test[iob_column]
                                           + lang_ds_validation[iob_column]))

            iob = ClassLabel(num_classes=len(iob_uniq), names=iob_uniq)

            lang_ds_train = lang_ds_train.add_column("iob",
                    [iob.str2int(s.split()) for s in lang_ds_train[iob_column]])
            lang_ds_test = lang_ds_test.add_column("iob",
                    [iob.str2int(s.split()) for s in lang_ds_test[iob_column]])
            lang_ds_validation = lang_ds_validation.add_column("iob",
                    [iob.str2int(s.split()) for s in lang_ds_validation[iob_column]])

        #tokens
        lang_ds_train = lang_ds_train.add_column("tokens",
                [x.split() for x in lang_ds_train["text"]])
        lang_ds_test = lang_ds_test.add_column("tokens",
                [x.split() for x in lang_ds_test["text"]])
        lang_ds_validation = lang_ds_validation.add_column("tokens",
                [x.split() for x in lang_ds_validation["text"]])

    dataset = DatasetDict({"train": lang_ds_train, "validation": lang_ds_validation,
        "test": lang_ds_test})
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
    """
    Tokenizes and processes labels for NLU tasks.

    This function tokenizes the input text and processes the corresponding IOB (Inside, Outside,
    Beginning) labels to align them with the tokenized input. It ensures that each token has a
    corresponding label, either by extending labels to match the tokenized text or by
    padding/truncating them to a specified maximum length.

    Parameters:
        examples (Dict): A dictionary containing the text and labels. Expected keys are 'annot_utt'
            for annotated utterances, 'iob' for IOB labels, and 'tokens' for tokenized text.
        tokenizer: The tokenizer used for processing the text.

    Returns:
        Dict: A dictionary with tokenized inputs and processed labels. Adds keys 'labels' and 'iob'
            to the input dictionary, containing the processed labels aligned with the tokenized
            text.
    """
    # Tokenize all examples
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length",
                                 max_length=128)

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
        label_ids = pad_or_truncate_labels(label_ids, 128)

        # Append label ids to the list
        labels.append(label_ids)  # Corrected indentation

    # Add processed labels to tokenized_inputs
    tokenized_inputs["labels"] = labels

    # Process iob
    iobs = []
    for iob in examples["iob"]:
        padded_iob = pad_or_truncate_labels(iob, 128)
        iobs.append(padded_iob)

    tokenized_inputs["iob"] = iobs

    return tokenized_inputs
