import sys
import os
import ipdb
import random
import numpy as np
import pickle
import copy
from typing import Dict
from collections import OrderedDict
import logging
from tqdm import tqdm
import regex as re

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import LSTM, CrossEntropyLoss
from torch.optim import Adam
from transformers import AdamW, AutoModel

# from metric import Carb

class IGL_OIE(nn.Module):

    def __init__(        
        self, 
        bert_model_str='bert-base-cased', 
        iterative_layers=2, 
        labelling_dim=300,
        dropout = 0.0,
        max_depth=5,
        meta_data_vocab=None):
        
        super(IGL_OIE, self).__init__()

        self._base_model = AutoModel.from_pretrained(bert_model_str)
        self._base_model.to("cuda")
        self._hidden_size = self._base_model.config.hidden_size

        self._iterative_transformer = self._base_model.encoder.layer[-iterative_layers:]
        self._base_model.encoder.layer = self._base_model.encoder.layer[:-iterative_layers]
       
        self._num_labels = 6
        self._dropout = nn.Dropout(p=dropout).cuda()

        self._label_embeddings = nn.Embedding(100, self._hidden_size).cuda()

        self._labelling_dim = labelling_dim
        self._labelling_layer = nn.Linear(self._labelling_dim, self._num_labels).cuda()
        self._merge_layer = nn.Linear(self._hidden_size, self._labelling_dim).cuda()

        self._loss = nn.CrossEntropyLoss()

        self._max_depth = max_depth

        self._meta_data_vocab = meta_data_vocab
        self._constD = dict()

        self.all_predictions_conj = []
        self.all_sentence_indices_conj = []
        self.all_conjunct_words_conj = []
        self.all_predictions_oie = []

    def forward(self, 
                batch,
                batch_idx=-1,
                constraints=None, 
                cweights=None):
        
        batch_labels_cuda = batch["labels"].cuda()
        batch_text_cuda = batch["text"].cuda()
        batch_word_starts_cuda = batch["word_starts"].cuda()
        
        batch_size, depth, labels_length = batch_labels_cuda.shape
        if not self.training:
            depth = self._max_depth

        loss, lstm_loss = 0, 0
        # print(len(batch["text"]))
        hidden_states, _ = self._base_model(batch_text_cuda)
        output_dict = dict()
        # (batch_size, seq_length, max_depth, num_labels)
        all_depth_scores = []

        d = 0
        while True:
            for layer in self._iterative_transformer:
                hidden_states = layer(hidden_states)[0]

            hidden_states = self._dropout(hidden_states)
            word_hidden_states = torch.gather(hidden_states, 1,batch_word_starts_cuda.unsqueeze(2).repeat(1, 1, hidden_states.shape[2]))

            if d != 0:
                greedy_labels = torch.argmax(word_scores, dim=-1)      
                label_embeddings = self._label_embeddings(greedy_labels)
                word_hidden_states = word_hidden_states + label_embeddings

            word_hidden_states = self._merge_layer(word_hidden_states)
            word_scores = self._labelling_layer(word_hidden_states)
            all_depth_scores.append(word_scores)

            d += 1
            if d >= depth:
                break
            if not self.training:
                predictions = torch.max(word_scores, dim=2)[1]
                valid_ext = False
                for p in predictions:
                    if 1 in p and 2 in p:
                        valid_ext = True
                        break
                if not valid_ext:
                    break 
        
        # (batch_size, seq_length, max_depth)
        all_depth_predictions, all_depth_confidences = [], []
        batch_size, num_words, _ = word_scores.shape
        batch_labels_cuda = batch_labels_cuda.long()
        for d, word_scores in enumerate(all_depth_scores):
            if self.training:
                batch_labels_d = batch_labels_cuda[:, d, :]
                mask = torch.ones(batch_word_starts_cuda.shape).int().type_as(hidden_states)
                loss += self._loss(word_scores.reshape(batch_size*num_words, -1),batch_labels_cuda[:, d, :].reshape(-1))
            else:
                word_log_probs = torch.log_softmax(word_scores, dim=2)
                max_log_probs, predictions = torch.max(word_log_probs, dim=2)

                padding_labels = (batch_labels_cuda[:,0,:]!=-100).float()

                sro_label_predictions = (predictions!=0).float() * padding_labels
                log_probs_norm_ext_len = (max_log_probs * sro_label_predictions) / (sro_label_predictions.sum(dim=0)+1)
                confidences = torch.exp(torch.sum(log_probs_norm_ext_len, dim=1))

                all_depth_predictions.append(predictions.unsqueeze(1))
                all_depth_confidences.append(confidences.unsqueeze(1))

        if not self.training:            
            all_depth_predictions = torch.cat(all_depth_predictions, dim=1)
            all_depth_confidences = torch.cat(all_depth_confidences, dim=1)
                
            output_dict['predictions'] = all_depth_predictions
            output_dict['scores'] = all_depth_confidences

            
            
            
        output_dict['loss'] = loss
        return output_dict