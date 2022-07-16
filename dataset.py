import sys
import os
import numpy as np
import csv
import pickle
import argparse
import random
import torch
import math
from numpy import genfromtxt

class Dataset:
    def __init__(self, ds_name):
        self.name = ds_name
        self.dir = "datasets/" + ds_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.rel2dom = {}
        self.rel2range = {}
        self.typid2entid = {}
        self.typekeys = []
        self.entid2typid = {}
        self.data = {}
        self.data["train"] = torch.as_tensor(genfromtxt(self.dir + "train2id.txt", delimiter='\t'), dtype=torch.int32)
        self.data["valid"] = torch.as_tensor(genfromtxt(self.dir + "valid2id.txt", delimiter='\t'), dtype=torch.int32)
        self.data["test"] = torch.as_tensor(genfromtxt(self.dir + "test2id.txt", delimiter='\t'), dtype=torch.int32)
        self.batch_index = 0
        self.init()

    def init(self):
        with open("datasets/" + self.name + "/" + "ent2id.pkl", 'rb') as f:
            self.ent2id = pickle.load(f)
        with open("datasets/" + self.name + "/" + "rel2id.pkl", 'rb') as f:
            self.rel2id = pickle.load(f)
        try:
            with open("datasets/" + self.name + "/" + "typid2entid.pkl", 'rb') as f:
                self.typid2entid = pickle.load(f)
                self.typekeys = self.typid2entid.keys()
        except:
            pass
        try:
            with open("datasets/" + self.name + "/" + "entid2typid.pkl", 'rb') as f:
                self.entid2typid = pickle.load(f)
        except:
            pass
        try:
            with open("datasets/" + self.name + "/" + "rel2dom.pkl", 'rb') as f:
                self.rel2dom = pickle.load(f)
        except:
            pass
        try:
            with open("datasets/" + self.name + "/" + "rel2range.pkl", 'rb') as f:
                self.rel2range = pickle.load(f)
        except:
            pass
    
    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)
                     
    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_id_ent(self, i):
    	return self.id2ent[i]
            
    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def get_id_rel(self, id):
    	return self.id2rel[i] 
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent
                     
    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]): 
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return batch
                     
    def random_negative_sampling(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        M = neg_batch.shape[0]
        corr = np.random.randint(self.num_ent() - 1, size=M)
        if side=='all':
            e_idxs = np.random.choice([0, 2], size=M)
            neg_batch[np.arange(M), e_idxs] = corr
        elif side=='tail':
            neg_batch[np.arange(M), 2] = corr
        elif side=='head':
            neg_batch[np.arange(M), 0] = corr
        return neg_batch

    def type_constrained_sampling(self, pos_batch, neg_ratio, side='all'):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if side=='all':
                if random.random() < 0.5:
                    neg_batch[i][0] = self.tc_except_dom(neg_batch[i][1], neg_batch[i][0]) #flipping head
                else:
                    neg_batch[i][2] = self.tc_except_range(neg_batch[i][1], neg_batch[i][2]) #flipping tail
            elif side=='tail':
                neg_batch[i][2] = self.tc_except_range(neg_batch[i][1], neg_batch[i][2]) #flipping tail
            elif side=='head':
                neg_batch[i][0] = self.tc_except_dom(neg_batch[i][1], neg_batch[i][0]) #flipping head
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, neg_sampler, device):
        pos_batch = self.next_pos_batch(batch_size)
        if neg_sampler == 'rns':
            neg_batch = self.random_negative_sampling(pos_batch, neg_ratio)
        elif neg_sampler == 'tcns':
            neg_batch = self.type_constrained_sampling(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        batch = torch.tensor(batch)
        return batch
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))