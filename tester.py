import torch
from dataset import Dataset
import numpy as np
from numpy import genfromtxt
import itertools
import time
from os import listdir
from os.path import isfile, join
from torch.utils import data as torch_data
from models import TransE, DistMult, ComplEx
from collections import defaultdict
import pickle
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, dataset, args, model_path, valid_or_test):
        self.args = args
        self.device = args.device
        self.model_name = args.model
        if self.model_name == 'TransE':
            self.model = TransE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        if self.model_name == 'DistMult':
            self.model = DistMult(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        if self.model_name == 'ComplEx':
            self.model = ComplEx(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.dataset = dataset
        self.rel2dom = dataset.rel2dom
        self.rel2range = dataset.rel2range
        self.entid2typid = dataset.entid2typid
        self.valid_or_test = valid_or_test
        self.batch_size = args.batch_size
        self.neg_ratio = args.neg_ratio
        self.neg_sampler = args.neg_sampler
        self.metric = args.metrics
        with open('datasets/' + self.dataset.name + "/observed_tails.pkl", 'rb') as f:
            self.all_possible_ts = pickle.load(f)
        with open('datasets/' + self.dataset.name + "/observed_heads.pkl", 'rb') as f:
            self.all_possible_hs = pickle.load(f)

    def get_observed_h(self, h, r, t):
        return(list(set(self.all_possible_hs[t.item()][r.item()]) - set([h.item()])))

    def get_observed_t(self, h, r, t):
        return(list(set(self.all_possible_ts[h.item()][r.item()]) - set([t.item()])))

    def calc_valid_loss(self):
        X_valid = (self.dataset.data[self.valid_or_test]).clone().detach()
        last_batch = False
        total_loss = 0.0
        while not last_batch:
            batch = self.dataset.next_batch_wo_shuffling(self.batch_size, neg_ratio=self.neg_ratio, neg_sampler=self.neg_sampler, device = self.device)
            last_batch = self.dataset.was_last_batch()
            pos_samples = batch[(batch[:, 3] == 1).nonzero().squeeze(1)]
            neg_samples = batch[(batch[:, 3] == -1).nonzero().squeeze(1)]
            pos_hs  = (pos_samples[:,0]).clone().detach().long().to(self.device)
            pos_rs   = (pos_samples[:,1]).clone().detach().long().to(self.device)
            pos_ts  = (pos_samples[:,2]).clone().detach().long().to(self.device)
            pos_scores = self.model.forward(pos_hs, pos_rs, pos_ts)
            neg_hs  = (neg_samples[:,0]).clone().detach().long().to(self.device)
            neg_rs   = (neg_samples[:,1]).clone().detach().long().to(self.device)
            neg_ts  = (neg_samples[:,2]).clone().detach().long().to(self.device)
            neg_scores = self.model.forward(neg_hs, neg_rs, neg_ts)
            loss = self.model._loss(pos_scores, neg_scores, self.neg_ratio)
            total_loss += loss.cpu().item()
        print("Validation loss: " + str(total_loss) + "(" + self.dataset.name + ")")
        return total_loss

    def calc_valid_mrr(self):
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        filt_mrr_h, filt_mrr_t = [], []
        X_valid = (self.dataset.data[self.valid_or_test]).clone().detach()
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        for i, (h, r, t) in enumerate(X_valid):
            heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
            rels = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])
            triplets = torch.stack((heads, rels, all_entities), dim=2).reshape(-1, 3)
            tails_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            triplets = torch.stack((all_entities, rels, tails), dim=2).reshape(-1, 3)
            heads_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            heads_predictions, tails_predictions = heads_predictions.squeeze(), tails_predictions.squeeze()
            # Filtered Scenario
            rm_idx_h = self.get_observed_h(h,r,t)
            rm_idx_t = self.get_observed_t(h,r,t)
            if self.model_name == 'DistMult' or self.model_name == 'ComplEx':
                heads_predictions[[rm_idx_h]] = - np.inf
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                indices_head = heads_predictions.argsort(descending=True)
            else:
                heads_predictions[[rm_idx_h]] = + np.inf
                tails_predictions[[rm_idx_t]] = + np.inf
                indices_tail = tails_predictions.argsort(descending=False)
                indices_head = heads_predictions.argsort(descending=False)
            filt_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
            filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
            # Filtered MR and MRR
            filt_mrr_h.append(1.0/filt_rank_h)
            filt_mrr_t.append(1.0/filt_rank_t)
        filt_mrr = np.mean(filt_mrr_h + filt_mrr_t)
        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        return filt_mrr

    def calc_valid_mrr_sem(self):
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        sem1_h, sem5_h, sem10_h, sem1_t, sem5_t,sem10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_mrr_h, filt_mrr_t = [], []
        X_valid_or_test = (self.dataset.data[self.valid_or_test]).clone().detach()
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        for i, (h, r, t) in enumerate(X_valid_or_test):
            heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
            rels = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])
            triplets = torch.stack((heads, rels, all_entities), dim=2).reshape(-1, 3)
            tails_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            triplets = torch.stack((all_entities, rels, tails), dim=2).reshape(-1, 3)
            heads_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            heads_predictions, tails_predictions = heads_predictions.squeeze(), tails_predictions.squeeze()
            if self.model_name == 'DistMult' or self.model_name == 'ComplEx':
                indices_tail = tails_predictions.argsort(descending=True)
                indices_head = heads_predictions.argsort(descending=True)
            else:
                indices_tail = tails_predictions.argsort(descending=False)
                indices_head = heads_predictions.argsort(descending=False)
            # Sem@K
            s1, s5, s10 = self.sem_at_k_head(indices_head[:10], r.item(), 10)
            sem1_h, sem5_h, sem10_h = sem1_h+s1, sem5_h+s5, sem10_h+s10
            s1, s5, s10 = self.sem_at_k_tail(indices_tail[:10], r.item(), 10)
            sem1_t, sem5_t, sem10_t = sem1_t+s1, sem5_t+s5, sem10_t+s10
            # Filtered Scenario
            rm_idx_h = self.get_observed_h(h,r,t)
            rm_idx_t = self.get_observed_t(h,r,t)
            if self.model_name == 'DistMult' or self.model_name == 'ComplEx':
                heads_predictions[[rm_idx_h]] = - np.inf
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                indices_head = heads_predictions.argsort(descending=True)
            else:
                heads_predictions[[rm_idx_h]] = + np.inf
                tails_predictions[[rm_idx_t]] = + np.inf
                indices_tail = tails_predictions.argsort(descending=False)
                indices_head = heads_predictions.argsort(descending=False)
            filt_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
            filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
            # Filtered MR and MRR
            filt_mrr_h.append(1.0/filt_rank_h)
            filt_mrr_t.append(1.0/filt_rank_t)
        filt_mrr = np.mean(filt_mrr_h + filt_mrr_t)
        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        sem_1 = 100*(sem1_h + sem1_t)/(2*X_valid_or_test.shape[0])
        sem_5 = 100*(sem5_h + sem5_t)/(2*X_valid_or_test.shape[0])
        sem_10 = 100*(sem10_h + sem10_t)/(2*X_valid_or_test.shape[0])
        for k in [1, 5, 10]:
            logger.info('Sem@{}: {}'.format(k, (eval('sem_'+str(k)))))
        return (filt_mrr, sem_1, sem_5, sem_10)


    def test(self):
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        sem1_h, sem5_h, sem10_h, sem1_t, sem5_t,sem10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_hit1_h, filt_hit1_t, filt_hit5_h, filt_hit5_t, filt_hit10_h, filt_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        raw_hit1_h, raw_hit1_t, raw_hit5_h, raw_hit5_t, raw_hit10_h, raw_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_mr_h, filt_mr_t = [], []
        raw_mr_h, raw_mr_t = [], []
        filt_mrr_h, filt_mrr_t = [], []
        raw_mrr_h, raw_mrr_t = [], []
        X_valid_or_test = (self.dataset.data[self.valid_or_test]).clone().detach()
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        for i, (h, r, t) in enumerate(X_valid_or_test):
            heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
            rels = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])
            triplets = torch.stack((heads, rels, all_entities), dim=2).reshape(-1, 3)
            tails_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            triplets = torch.stack((all_entities, rels, tails), dim=2).reshape(-1, 3)
            heads_predictions = self.model.forward((triplets[:,0]).clone().detach().long().to(self.device), \
                        (triplets[:,1]).clone().detach().long().to(self.device), (triplets[:,2]).clone().detach().long().to(self.device)).reshape(1, -1)
            heads_predictions, tails_predictions = heads_predictions.squeeze(), tails_predictions.squeeze()

            # Raw Scenario
            if self.model_name == 'DistMult' or self.model_name == 'ComplEx':
                indices_tail = tails_predictions.argsort(descending=True)
                indices_head = heads_predictions.argsort(descending=True)
            else:
                indices_tail = tails_predictions.argsort(descending=False)
                indices_head = heads_predictions.argsort(descending=False)

            # Sem@K
            if self.metric == 'sem' or self.metric == 'all':
                s1, s5, s10 = self.sem_at_k_head(indices_head[:10], r.item(), 10)
                sem1_h, sem5_h, sem10_h = sem1_h+s1, sem5_h+s5, sem10_h+s10
                s1, s5, s10 = self.sem_at_k_tail(indices_tail[:10], r.item(), 10)
                sem1_t, sem5_t, sem10_t = sem1_t+s1, sem5_t+s5, sem10_t+s10

            if self.metric == 'ranks' or self.metric == 'all':
                # Raw Ranks
                raw_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
                raw_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                # Raw MR and MRR
                raw_mr_h.append(raw_rank_h)
                raw_mrr_h.append(1.0/raw_rank_h)
                raw_mr_t.append(raw_rank_t)
                raw_mrr_t.append(1.0/raw_rank_t)
                # Raw Hits@K
                raw_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                raw_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                raw_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                raw_hit10_h += torch.where(indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                raw_hit5_h += torch.where(indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                raw_hit1_h += torch.where(indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()

                # Filtered Scenario
                rm_idx_h = self.get_observed_h(h,r,t)
                rm_idx_t = self.get_observed_t(h,r,t)
                if self.model_name == 'DistMult' or self.model_name == 'ComplEx':
                    heads_predictions[[rm_idx_h]] = - np.inf
                    tails_predictions[[rm_idx_t]] = - np.inf
                    indices_tail = tails_predictions.argsort(descending=True)
                    indices_head = heads_predictions.argsort(descending=True)
                else:
                    heads_predictions[[rm_idx_h]] = + np.inf
                    tails_predictions[[rm_idx_t]] = + np.inf
                    indices_tail = tails_predictions.argsort(descending=False)
                    indices_head = heads_predictions.argsort(descending=False)
                filt_rank_h = (indices_head == h).nonzero(as_tuple=True)[0].item() + 1
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                # Filtered MR and MRR
                filt_mr_h.append(filt_rank_h)
                filt_mrr_h.append(1.0/filt_rank_h)
                filt_mr_t.append(filt_rank_t)
                filt_mrr_t.append(1.0/filt_rank_t)
                # Filtered Hits@K
                filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                filt_hit10_h += torch.where(indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                filt_hit5_h += torch.where(indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                filt_hit1_h += torch.where(indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()

            if i%1000==0:
                print('Scored ', i+1, ' test triples.')
        print(time.time() - start)
        raw_mr = np.mean(raw_mr_h + raw_mr_t)
        filt_mr = np.mean(filt_mr_h + filt_mr_t)
        raw_mrr = np.mean(raw_mrr_h + raw_mrr_t)
        filt_mrr = np.mean(filt_mrr_h + filt_mrr_t)
        raw_hits_at_10 = (raw_hit10_h + raw_hit10_t)/(2*X_valid_or_test.shape[0])
        raw_hits_at_5 = (raw_hit5_h + raw_hit5_t)/(2*X_valid_or_test.shape[0])
        raw_hits_at_1 = (raw_hit1_h + raw_hit1_t)/(2*X_valid_or_test.shape[0])
        filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t)/(2*X_valid_or_test.shape[0])
        filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t)/(2*X_valid_or_test.shape[0])
        filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t)/(2*X_valid_or_test.shape[0])

        logger.info('{} MR: {}'.format('Raw', raw_mr))
        logger.info('{} MRR: {}'.format('Raw', raw_mrr))
        logger.info('{} MR: {}'.format('Filtered', filt_mr))
        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        logger.info('{} Hits@{}: {}'.format('Raw', 1, raw_hits_at_1*100))
        logger.info('{} Hits@{}: {}'.format('Raw', 5, raw_hits_at_5*100))
        logger.info('{} Hits@{}: {}'.format('Raw', 10, raw_hits_at_10*100))
        logger.info('{} Hits@{}: {}'.format('Filtered', 1, filtered_hits_at_1*100))
        logger.info('{} Hits@{}: {}'.format('Filtered', 5, filtered_hits_at_5*100))
        logger.info('{} Hits@{}: {}'.format('Filtered', 10, filtered_hits_at_10*100))
        if self.metric == 'sem' or self.metric == 'all':
            sem_1 = (sem1_h + sem1_t)/2
            sem_5 = (sem5_h + sem5_t)/2
            sem_10 = (sem10_h + sem10_t)/2
            for k in [1, 5, 10]:
                logger.info('Sem@{}: {}'.format(k, 100*(eval('sem_'+str(k)))/(X_valid_or_test.shape[0])))

    def sem_at_k_head(self, pred_idx, r, k=10):
        idx = pred_idx.tolist()
        if self.dataset.name == "EduKG" or self.dataset.name == "KG20C":
            idx_types = [[self.entid2typid[i]] for i in idx]
        else:
            idx_types = [self.entid2typid[i] for i in idx]
        valid_type = self.rel2dom[r]
        type_checks = [self.type_checking(lst, [valid_type]) for lst in idx_types]
        return (type_checks[0], np.mean(type_checks[:5]), np.mean(type_checks[:10])) # sem@1/5/10

    def sem_at_k_tail(self, pred_idx, r, k=10):
        idx = pred_idx.tolist()
        if self.dataset.name == "EduKG" or self.dataset.name == "KG20C":
            idx_types = [[self.entid2typid[i]] for i in idx]
        else:
            idx_types = [self.entid2typid[i] for i in idx]
        valid_type = self.rel2range[r]
        type_checks = [self.type_checking(lst, [valid_type]) for lst in idx_types]
        return (type_checks[0], np.mean(type_checks[:5]), np.mean(type_checks[:10])) # sem@1/5/10

    def type_checking(self, idx_lst, valid_ent_types):
        return (1 if len(set(idx_lst).intersection(valid_ent_types))>= 1 else 0)