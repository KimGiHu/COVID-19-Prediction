from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import randint
from datasets import concatenate_datasets

checkpoint = 'bert-base-multilingual-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

label_binary = {'+': 1, '-':1, '1':1, '0': 0, 1:1, 0:0}

label_list = ['news_headline','article','korean','english','spanish','확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']


def load_data(language='spanish'):
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'},encoding='utf-8')
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example[language], truncation=True)

    def trim(example):
        example[language] = example[language].strip()
        return example
    

    def get_labels(example):
        example['labels'] = [label_binary[example[lab]] for lab in label_list[5:]]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def load_data_multilang(languages=['korean', 'english', 'spanish']):
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    datasets = []
    for language in languages:
        def tokenize_function(example):
            return tokenizer(example[language], truncation=True)

        def trim(example):
            example[language] = example[language].strip()
            return example
        

        def get_labels(example):
            example['labels'] = [label_binary[example[lab]] for lab in label_list[5:]]
            return example

        tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)

        tokenized_datasets = tokenized_datasets.remove_columns(
            label_list
        )
        tokenized_datasets.set_format("torch")
        datasets.append(tokenized_datasets)
    
    train_datasets = concatenate_datasets([d["train"] for d in datasets])
    eval_datasets = concatenate_datasets([d["validation"] for d in datasets])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        eval_datasets, batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader



if __name__ == '__main__':
    train_dataloader, eval_dataloader = load_data_multilang()
