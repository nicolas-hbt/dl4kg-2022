import math
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_


### TransE ###
class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=5.0):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.distance = 'l2'
        self.gamma = margin

        self.ent_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs   = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)       

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs), self.rel_embs(rs), self.ent_embs(ts)
        f = self._calc(e_hs, e_rs, e_ts).view(-1, 1)
        return f

    def _calc(self, e_hs, e_rs, e_ts):
        if self.distance == 'l1':
            out = torch.sum(torch.abs(e_hs + e_rs - e_ts), 1)
        else:
            out = torch.sqrt(torch.sum((e_hs + e_rs - e_ts)**2, 1))
        return out
    
    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(neg_ratio)
        y_neg = y_neg.view(-1)
        target = Variable(torch.from_numpy(-np.ones(P*neg_ratio, dtype=np.int32)))
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs), self.rel_embs(rs), self.ent_embs(ts)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)) / 3
        return regul
    
    def predict(self, hs, rs, ts):
        y_pred = self.forward(hs, rs, ts).view(-1, 1)
        return y_pred.data


### DistMult ###
class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=5.0):
        super(DistMult, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.distance = 'l2'
        self.gamma = margin

        self.ent_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs   = nn.Embedding(num_rel, emb_dim).to(device)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)

    def forward(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs), self.rel_embs(rs), self.ent_embs(ts)
        f = self._calc(e_hs, e_rs, e_ts)
        return f

    def _calc(self,e_hs,e_rs,e_ts):
        return torch.sum(e_hs*e_rs*e_ts,-1)
    
    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(neg_ratio)
        y_neg = y_neg.view(-1)
        target = Variable(torch.from_numpy(np.ones(P*neg_ratio, dtype=np.int32)))
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs), self.rel_embs(rs), self.ent_embs(ts)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2))/3
        return regul
    
    def predict(self, hs, rs, ts):
        y_pred = self.forward(hs, rs, ts).view(-1, 1)
        return y_pred.data

class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=5.0, lmbda=0.0):
        super(ComplEx,self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.gamma = margin
        self.lmbda = lmbda
        self.ent_re_embeddings=nn.Embedding(self.num_ent, self.emb_dim)
        self.ent_im_embeddings=nn.Embedding(self.num_ent, self.emb_dim)
        self.rel_re_embeddings=nn.Embedding(self.num_rel, self.emb_dim)
        self.rel_im_embeddings=nn.Embedding(self.num_rel, self.emb_dim)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        
    def _calc(self,e_re_h,e_im_h,r_re,r_im,e_re_t,e_im_t):
        return torch.sum(r_re*e_re_h*e_re_t + r_re*e_im_h*e_im_t + r_im*e_re_h*e_im_t - r_im*e_im_h*e_re_t,1,False)
    
    def _loss(self, y_pos, y_neg, neg_ratio):
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(neg_ratio)
        y_neg = y_neg.view(-1)
        target = Variable(torch.from_numpy(np.ones(P*neg_ratio, dtype=np.int32)))
        criterion = nn.MarginRankingLoss(margin=self.gamma)
        loss = criterion(y_pos, y_neg, target)
        return loss
    
    def forward(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs), self.ent_im_embeddings(hs)
        e_re_t, e_im_t = self.ent_re_embeddings(ts), self.ent_im_embeddings(ts)
        r_re, r_im = self.rel_re_embeddings(rs), self.rel_im_embeddings(rs)
        f = self._calc(e_re_h,e_im_h,r_re,r_im,e_re_t,e_im_t)
        return f

    def _regularization(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs), self.ent_im_embeddings(hs)
        e_re_t, e_im_t = self.ent_re_embeddings(ts), self.ent_im_embeddings(ts)
        r_re, r_im = self.rel_re_embeddings(rs), self.rel_im_embeddings(rs)
        regul = (torch.mean(e_re_h ** 2) + 
                 torch.mean(e_im_h ** 2) + 
                 torch.mean(e_re_t ** 2) +
                 torch.mean(e_im_t ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul
    
    def predict(self, hs, rs, ts):
        y_pred = self.forward(hs, rs, ts).view(-1, 1)
        return y_pred.data