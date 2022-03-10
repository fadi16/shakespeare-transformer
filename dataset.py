import re

import torch
from torch.utils.data import Dataset


class ShakespearianDataset(Dataset):
    def __init__(self, source_file_path, target_file_path, tokenizer, max_source_len, max_target_len, add_tokens=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_text = preprocess_file(source_file_path)
        longest_source_sequence = len(max(self.source_text, key=lambda x: len(x.split())).split())
        print("longest_source_sequence = ", longest_source_sequence)
        self.target_text = preprocess_file(target_file_path)
        longest_target_sequence = len(max(self.target_text, key=lambda x: len(x.split())).split())
        print("longest_target_sequence = ", longest_target_sequence)

        assert len(self.source_text) == len(self.target_text)

        # todo: currently zero tokens are being added, why?
        if add_tokens:
            # add missing tokens to tokenizer - mostly on the Shakespearian side
            self.update_tokenizer()

    def update_tokenizer(self):
        # from https://github.com/huggingface/tokenizers/issues/627
        all_target_text = " ".join(self.target_text)
        new_tokens = self.tokenizer.tokenize(all_target_text)
        self.tokenizer.add_tokens(new_tokens)

        # this method only adds new tokens

    def __len__(self):
        """returns the length of the dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return input ids, attention marks and target ids"""
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # clean data, make sure it's a string
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        # tokenizing source
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def preprocess_sent(sent):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sent = re.sub(r"[^a-zA-Z?.'!,¿]+", " ", sent)
    return sent


def preprocess_file(f_path):
    sents = open(f_path, "r").readlines()
    return [preprocess_sent(s) for s in sents]
