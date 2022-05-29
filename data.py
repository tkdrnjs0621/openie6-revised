# Utilities for getting data
import os
import ipdb
import pdb
import random
import argparse
import pickle
import csv
import nltk
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

def pad_data(data):
    model_str='bert-base-cased'
    do_lower_case=True
    tokenizer = AutoTokenizer.from_pretrained(model_str, do_lower_case=do_lower_case)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # print("PRINTED ",data[0][0])
    
    maxlen=-1
    texts = [d['text'] for d in data]
    for t in texts:
        if(len(t)>maxlen):
            maxlen=len(t)
    padded_text = []
    for t in texts:
        add=maxlen-len(t)
        a = t.copy() + [pad_id]*add
        padded_text.append(a)
    
    
    labels = [d['labels'] for d in data]
    max_depth=5
    for i in range(len(labels)):
        pad_depth = max_depth - len(labels[i])
        num_words = len(labels[i][0])
        labels[i] = labels[i] + [[0]*num_words]*pad_depth
    
    maxlen=-1
    for l in labels:
        if(len(l[0])>maxlen):
            maxlen=len(l[0])
    padded_labels = []
    
    for l in labels:
        _l = []
        for ll in l:
            add = maxlen-len(ll)
            a = ll.copy() + [-100]*add
            _l.append(a)
        padded_labels.append(_l)
    
    
    maxlen=-1
    ws = [d['word_starts'] for d in data]
    for w in ws:
        if(len(w)>maxlen):
            maxlen=len(w)
    padded_word_starts = []
    for w in ws:
        add=maxlen-len(w)
        a = w.copy() + [0]*add
        padded_word_starts.append(a)
    
    padded_meta_data = [d['meta_data'] for d in data]
    
    padded_text=torch.tensor(padded_text)
    padded_labels=torch.tensor(padded_labels)
    padded_word_starts=torch.tensor(padded_word_starts)
    # padded_meta_data=torch.tensor(padded_meta_data)
    
    paddedD = {'text': padded_text, 'labels': padded_labels,
               'word_starts': padded_word_starts, 'meta_data': padded_meta_data}
    
#     fields = data[0][-1]
#     TEXT = fields['text'][1]
#     text_list = [ex[2].text for ex in data]
#     padded_text = torch.tensor(TEXT.pad(text_list))

#     LABELS = fields['labels'][1]
#     labels_list = [ex[2].labels for ex in data]
#     # max_depth = max([len(l) for l in labels_list])
#     max_depth = 5
#     for i in range(len(labels_list)):
#         pad_depth = max_depth - len(labels_list[i])
#         num_words = len(labels_list[i][0])
#         # print(num_words, pad_depth)
#         labels_list[i] = labels_list[i] + [[0]*num_words]*pad_depth
#     # print(labels_list)
#     padded_labels = torch.tensor(LABELS.pad(labels_list))

#     WORD_STARTS = fields['word_starts'][1]
#     word_starts_list = [ex[2].word_starts for ex in data]
#     padded_word_starts = torch.tensor(WORD_STARTS.pad(word_starts_list))

#     META_DATA = fields['meta_data'][1]
#     meta_data_list = [META_DATA.vocab.stoi[ex[2].meta_data] for ex in data]
#     padded_meta_data = torch.tensor(META_DATA.pad(meta_data_list))

#     paddedD = {'text': padded_text, 'labels': padded_labels,
#                'word_starts': padded_word_starts, 'meta_data': padded_meta_data}

    return paddedD

def process_data(
    input_path,
    model_str='bert-base-cased',
    bos_token_id=101,
    eos_token_id=102,
    do_lower_case=True
    ):
    
    label_dict = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                      'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
    
    examples, exampleDs, targets, lang_targets, orig_sentences = [], [], [], [], []
    tokenizer = AutoTokenizer.from_pretrained(model_str, do_lower_case=do_lower_case, use_fast=True, data_dir='data/pretrained_cache',
                                              add_special_tokens=False, additional_special_tokens=['[unused1]', '[unused2]', '[unused3]'])

    sentence = None
    max_extraction_length = 5

    inp_lines = open(input_path, 'r').readlines()

    new_example = True
    for line_num, line in tqdm(enumerate(inp_lines)):
        line = line.strip()
        if line == '':
            new_example = True

        if '[unused' in line or new_example:
            if sentence is not None:
                if len(targets) == 0:
                    targets = [[0]]
                    lang_targets = [[0]]
                orig_sentence = sentence.split('[unused1]')[0].strip()
                orig_sentences.append(orig_sentence)

                exampleD = {'text': input_ids, 'labels': targets[:max_extraction_length], 'word_starts': word_starts, 'meta_data': orig_sentence}
                if len(sentence.split()) <= 100:
                    exampleDs.append(exampleD)

                targets = []
                sentence = None
            # starting new example
            if line is not '':
                new_example = False
                sentence = line

                tokenized_words = tokenizer.batch_encode_plus(sentence.split())
                input_ids, word_starts, lang = [bos_token_id], [], []
                for tokens in tokenized_words['input_ids']:
                    if len(tokens) == 0: # special tokens like \x9c
                        tokens = [100]
                    word_starts.append(len(input_ids))
                    input_ids.extend(tokens)
                input_ids.append(eos_token_id)
                assert len(sentence.split()) == len(word_starts), ipdb.set_trace()
        else:
            if sentence is not None:
                target = [label_dict[i] for i in line.split()]
                target = target[:len(word_starts)]
                assert len(target) == len(word_starts), ipdb.set_trace()
                targets.append(target)

    for exampleD in exampleDs:
        examples.append(exampleD)
    return examples, orig_sentences