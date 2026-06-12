from functools import cached_property
from logging import getLogger
from typing import Any, Dict, List

from datasets import load_dataset
import random
import json
import os
import re

from prompting.misc_utils import deterministic_hash, deterministic_random
from prompting.models import GenerationOutput, Prompt

logger = getLogger(__name__)


############################################################################################################
############################################################################################################


BB_TASKS = ['implicatures', 'question_selection', 'logical_fallacy_detection', 'presuppositions_as_nli', 'sports_understanding', 
            'navigate', 'epistemic_reasoning', 'dyck_languages', 'tense', 'gender_inclusive_sentences_german', 'operators', 'causal_judgment', 
            'winowhy', 'linguistics_puzzles', 'ruin_names', 'snarks', 'disambiguation_qa', 'movie_recommendation', 'timedial', 'hyperbaton']

MATH_TASKS = ["aqua"]

def load_BB_data(type, task):
    bb_induce_data_path = os.path.join(os.path.dirname(__file__), '../extra_data/bigbench-raw/induce/')
    bb_eval_data_path = os.path.join(os.path.dirname(__file__), '../extra_data/bigbench-raw//execute/')
    base_path = bb_induce_data_path if type == 'induce' else bb_eval_data_path
    
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    data_collection = []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)
        data_collection.append({"text": input_, "label": output_[0]})
    
    return inputs, outputs, data_collection

def load_math_data(type, task):
    induce_data_path = {"gsm8k":"../extra_data/math/gsm8k/gsm8k_train.json",
                        "aqua":"../extra_data/math/AQuA/aqua_train_processed.json",
                        "svamp": "../extra_data/math/SVAMP/svamp_train.json",
                        "multiarith": "../extra_data/math/MultiArith/multiarith_train.json"}
    eval_data_path = {"gsm8k":"../extra_data/math/gsm8k/gsm8k_test.json",
                    "aqua":"../extra_data/math/AQuA/aqua_test_processed.json",
                    "svamp": "../extra_data/math/SVAMP/svamp_test.json",
                    "multiarith": "../extra_data/math/MultiArith/multiarith_test.json"}
    
    path = induce_data_path[task] if type == 'induce' else eval_data_path[task]
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    data_collection = []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        #
        input_, output_ = data['input'], [data['output']]
        inputs.append(input_)
        outputs.append(output_)
        data_collection.append({"text": input_, "label": output_[0]})

    return inputs, outputs, data_collection


############################################################################################################
############################################################################################################


class BaseProcessor:
    @cached_property
    def dataset(self):
        if self.dataset_name in ["ag_news", "SetFit/sst2", "trec", "amazon_polarity"]:
            if self.dataset_name == "trec":
                return load_dataset(self.dataset_name, trust_remote_code=True)
            else:
                return load_dataset(self.dataset_name)
        elif self.dataset_name in BB_TASKS:
            _, __, train_val_data = load_BB_data(type='induce', task=self.dataset_name)
            _, __, test_data = load_BB_data(type='execute', task=self.dataset_name)
            dataset_collection = {"train": train_val_data, "validation": None, "test": test_data}
            return dataset_collection
        elif self.dataset_name in MATH_TASKS:
            _, __, train_val_data = load_math_data(type='induce', task=self.dataset_name)
            _, __, test_data = load_math_data(type='execute', task=self.dataset_name)
            dataset_collection = {"train": train_val_data, "validation": None, "test": test_data}
            return dataset_collection
        else:
            raise NotImplementedError
        
    @cached_property
    def train_split(self):
        return self.dataset["train"]

    @cached_property
    def val_split(self):
        return self.dataset["validation"]

    @cached_property
    def test_split(self):
        return self.dataset["test"]

    def generate_datasets(self, seed: int, mode: str):
        logger.info(f"generating datasets using seed {seed}")

        if len(self.train_split) < 1200:
            raise Exception(f"dataset is too small, size: {len(self.train_split)}")

        shuffled = deterministic_random(seed).sample(
            range(len(self.train_split)),
            k=len(self.train_split),
        )

        labeled_train_indices = shuffled[0:100]
        labeled_val_indices = shuffled[100:200]

        if mode.startswith("unlabeled-custom-"):
            unlabeled_size = int(mode.split("-")[-1])
            assert len(shuffled) >= 200 + unlabeled_size

            unlabeled_train_indices = shuffled[200 : 200 + unlabeled_size]
            logger.info(f"set unlabeled size to {unlabeled_size}")
        else:
            unlabeled_train_indices = shuffled[200:1200]

        if mode == "labeled" or mode == "labeled-gpt3":
            self.train_dataset = [self.train_split[i] for i in labeled_train_indices]
            self.val_dataset = [self.train_split[i] for i in labeled_val_indices]
        else:
            self.train_dataset = [self.train_split[i] for i in unlabeled_train_indices]
            self.val_dataset = [self.train_split[i] for i in labeled_val_indices]

        test_size = len(self.test_split)

        if mode == "labeled-gpt3":
            test_sample_size = 1000
        else:
            test_sample_size = 10000

        if test_size > test_sample_size:
            logger.info(f"subsampling test set to {test_sample_size} examples")
            test_size = test_sample_size

        test_indices = deterministic_random(42).sample(
            range(len(self.test_split)), k=test_size
        )
        self.test_dataset = [self.test_split[i] for i in test_indices]

        dataset_hash = deterministic_hash(
            f"{labeled_train_indices}{labeled_val_indices}{unlabeled_train_indices}"
        )
        logger.info(f"dataset hash={dataset_hash}")

    def get_label_idx(self, example: Dict):
        return example["label"]

    @cached_property
    def prompt_start(self):
        return ""

    def convert_example_to_template_fields(self, example: Dict):
        return example

    def fill_train_template(self, example: Dict):
        fields = self.convert_example_to_template_fields(example)
        return self.train_template.format(**fields)

    def fill_eval_template(self, example: Dict):
        fields = self.convert_example_to_template_fields(example)
        return self.eval_template.format(**fields)

    def fill_representation_template(self, example: Dict):
        fields = self.convert_example_to_template_fields(example)
        return self.representation_template.format(**fields)

    def fill_calibration_template(self, calibration_example: Dict):
        return self.eval_template.format(**calibration_example)

    def get_training_prompt(self, train_indices: List[int], train_split: str = "train"):
        if train_split not in ("train", "val"):
            raise Exception

        data = self.train_dataset if train_split == "train" else self.val_dataset
        training_prompt = self.prompt_start + "".join(
            [self.fill_train_template(data[idx]) for idx in train_indices]
        )
        return training_prompt

    def get_probing_prompt(self, train_indices: List[int]):
        data = self.train_dataset
        template = "input: {text}\ntype: {label_text}"
        prompt = "\n\n".join(
            [
                template.format(**self.get_probing_fields(data[idx]))
                for idx in train_indices
            ]
            + ["input:"]
        )
        return prompt

    def get_probing_fields(self, example: Dict):
        return self.convert_example_to_template_fields(example)

    def create_prompts(
        self,
        train_indices: List[int],
        train_split: str = "train",
        split: str = "test",
        custom_split: List[Dict] = None,
    ):
        prompts = []
        cali_prompts = []

        training_prompt = self.get_training_prompt(
            train_indices, train_split=train_split
        )

        # Custom split
        if split != "custom" and custom_split is not None:
            raise Exception("custom_split should be None unless split=custom")

        # Define the "dataset" based on split
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        elif split == "custom":
            if custom_split is None:
                raise Exception("split=custom but custom_split is None")
            dataset = custom_split
        else:
            raise Exception(f"Unknown split {split}")

        # Fill in the evaluation examples in the "dataset"
        for eval_example in dataset:
            prompt = Prompt(training_prompt + self.fill_eval_template(eval_example))
            prompts.append(prompt)

        # Calibration prompt
        for calibrate_example in self.calibration_examples:
            calibrate_prompt = Prompt(
                training_prompt + self.fill_calibration_template(calibrate_example),
                is_calibration_prompt=True,
            )
            cali_prompts.append(calibrate_prompt)

        return prompts, cali_prompts

    def extract_predictions(
        self,
        outputs: List[GenerationOutput],
        split: str = "test",
        custom_split: List[Dict] = None,
    ) -> Dict[str, Any]:

        if split != "custom" and custom_split is not None:
            raise Exception("custom_split should be None unless split=custom")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        elif split == "custom":
            if custom_split is None:
                raise Exception("split=custom but custom_split is None")
            dataset = custom_split
        else:
            raise Exception(f"Unknown split {split}")

        if len(outputs) != len(dataset):
            raise Exception(
                f"number of predictions ({len(outputs)}) != "
                f"number of examples ({len(dataset)})"
            )
        preds = []
        labels = []
        class_dist = [0] * len(self.labels)

        for output, test_example in zip(
            outputs, map(self.convert_example_to_template_fields, dataset)
        ):
            for i, label in enumerate(self.labels):
                if output.completion.startswith(label):
                    pred = label
                    class_dist[i] += 1
                    break
            else:
                raise Exception(
                    f"Completion {output.completion} does not match any of the labels {self.labels}"
                )
            preds.append(pred)
            labels.append(test_example["label_text"])
        acc = sum(
            1 if pred == label else 0 for pred, label in zip(preds, labels)
        ) / len(labels)

        return_dict = {
            "acc": acc,
            "preds": preds,
            "labels": labels,
            "class-dist": class_dist,
        }
        return return_dict

    def subsample_train_dataset(self, samples: int, seed: int = 42):
        logger.info(f"Subsampling training dataset to {samples} rows.")
        shuffled = self.dataset["train"].shuffle(seed=seed)
        self.train_dataset = [shuffled[i] for i in range(samples)]

    @property
    def calibration_examples(self):
        return [
            {"text": "N/A"},
            {"text": "[MASK]"},
            {"text": ""},
        ]


class SST2Processor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "SetFit/sst2"
        self.train_template = "Review: {text}\n" "Sentiment: {label_text}\n\n"
        self.eval_template = "Review: {text}\n" "Sentiment:"
        #
        self.representation_template = "Review: {text}\n"

        self.labels = ["positive", "negative"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def parse_probe_example(self, s: str):
        return {"text": s, "label_text": "positive"}


class TRECProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "trec"
        self.val_split = None
        self.prompt_start = (
            "Classify the questions based on whether their answ"
            "er type is a Number, Location, Person, Description"
            ", Entity, or Abbreviation.\n\n"
        )

        self.train_template = "Question: {text}\n" "Answer Type: {label_text}\n\n"
        self.eval_template = "Question: {text}\n" "Answer Type:"
        #
        self.representation_template = "Question: {text}\n"

        self.labels = [
            "Description",
            "Entity",
            "Abbreviation",
            "Person",
            "Number",
            "Location",
        ]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def get_label_idx(self, example: Dict):
        return example["coarse_label"]

    def convert_example_to_template_fields(self, example):
        label_text = self.labels[example["coarse_label"]]
        return {"text": example["text"], "label_text": label_text}

    def parse_probe_example(self, s: str):
        return {"text": s, "coarse_label": 0}


class AmazonProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "amazon_polarity"
        self.val_split = None
        self.train_template = (
            "Title: {title}\n" "Review: {review}\n" "Sentiment: {label_text}\n\n"
        )
        self.eval_template = "Title: {title}\n" "Review: {review}\n" "Sentiment:"
        #
        self.representation_template = "Title: {title}\n" "Review: {review}\n"

        self.labels = ["negative", "positive"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def convert_example_to_template_fields(self, example):
        label_text = self.labels[example["label"]]
        return {
            "title": example["title"],
            "review": example["content"],
            "label_text": label_text,
        }

    @property
    def calibration_examples(self):
        return [
            {"title": "N/A", "review": "N/A"},
            {"title": "[MASK]", "review": "[MASK]"},
            {"title": "", "review": ""},
        ]

    def get_probing_fields(self, example: Dict):
        label_text = self.labels[example["label"]]
        return {
            "text": example["title"] + "\n" + example["content"],
            "label_text": label_text,
        }

    def parse_probe_example(self, s: str):
        if "\n" in s:
            title, review = s.split("\n", maxsplit=1)
        else:
            title, review = "", s
        return {"title": title, "content": review, "label": 0}
    

class AGNewsProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "ag_news"
        self.val_split = None
        self.train_template = "Article: {text}\n" "Answer: {label_text}\n\n"
        self.eval_template = "Article: {text}\n" "Answer:"
        #
        self.representation_template = "Article: {text}\n"

        self.labels = ["World", "Sports", "Business", "Technology"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def convert_example_to_template_fields(self, example: Dict):
        label_text = self.labels[example["label"]]
        return {"text": example["text"], "label_text": label_text}

    def parse_probe_example(self, s: str):
        return {"text": s, "label": 0}
    

########################################################################################
########################################################################################

# winowhy epistemic\_reasoning timedial hyperbaton
# Aqua
class BB_Math_Processor(BaseProcessor):
    def __init__(self, dataset_name, seed: int, mode: str):
        self.dataset_name = dataset_name
        assert dataset_name in BB_TASKS or dataset_name in MATH_TASKS
        #
        self.val_split = None
        self.train_template = "Article: {text}\n" "Answer: {label_text}\n\n"
        self.eval_template = "Article: {text}\n" "Answer:"
        #
        self.representation_template = "Article: {text}\n"
        self.set_label_mapping(data_set=dataset_name)
        ###
        self.prompt_start = f"Task Option Mapping: {str(self.label_mapping)}.\n\n"

        #
        self.generate_datasets(seed, mode)
        #
        if dataset_name in ["winowhy", "epistemic_reasoning", "hyperbaton"]:
            self.labels = self.update_label_orders_direct()
        elif dataset_name in ["timedial", "aqua"]:
            self.labels = self.update_label_orders_matching()
        else:
            raise NotImplementedError
        self.model_kwargs = {"labels": self.labels}

    def update_label_orders_direct(self):
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for task in dataset:
                task["label"] = self.label_mapping[task["label"]]

        combined = self.train_dataset + self.val_dataset + self.test_dataset
        unique_labels = list(dict.fromkeys(item['label'] for item in combined))
        unique_labels = sorted(unique_labels)
        print("Unique labels:", unique_labels)

        label_to_index = {label: index for index, label in enumerate(unique_labels)}

        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for item in dataset:
                item['label'] = label_to_index[item['label']]

        return unique_labels
    
    def update_label_orders_matching(self):
        def normalize_label(label_str):
            label_str = label_str.strip()
            
            if len(label_str) == 1 and label_str.isalpha():
                return f"({label_str.upper()})"
            
            if not label_str.startswith("("):
                label_str = f"({label_str}"
            
            match = re.match(r"\(([A-Za-z])", label_str)
            if match:
                return f"({match.group(1).upper()})"
            else:
                print("- Return the same label.")
                return label_str

        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for task in dataset:
                normalized_label = normalize_label(task["label"])
                task["label"] = self.label_mapping[normalized_label]

        unique_labels = []
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for task in dataset:
                label = task["label"]
                if label not in unique_labels:
                    unique_labels.append(label)
        unique_labels = sorted(unique_labels)
        print("Unique labels:", unique_labels)

        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        print("Label to index mapping:", label_to_index)

        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            for task in dataset:
                task["label"] = label_to_index[task["label"]]

        return unique_labels

    def convert_example_to_template_fields(self, example: Dict):
        label_text = self.labels[example["label"]]
        return {"text": example["text"], "label_text": label_text}

    def parse_probe_example(self, s: str):
        return {"text": s, "label": 0}
    
    def set_label_mapping(self, data_set):
        mapping_fict = {
            "winowhy": {"correct": "Yes", "incorrect": "No"}, 
            "epistemic_reasoning": {"entailment": "Yes", "non-entailment": "No"}, 
            "timedial": {"(A)": "A", "(B)": "B", "(C)": "C"}, 
            "hyperbaton": {"a": "A", "b": "B"}, 
            "aqua": {"(A)": "A", "(B)": "B", "(C)": "C", "(D)": "D", "(E)": "E"}
        }
        self.label_mapping = mapping_fict[data_set]

