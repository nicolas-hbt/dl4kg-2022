from dataset import Dataset
from models import TransE, DistMult, ComplEx
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

class Trainer:
    def __init__(self, dataset, model_name, args):
        self.device = args.device
        self.model_name = model_name
        if self.model_name == 'TransE':
            self.model = TransE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        if self.model_name == 'DistMult':
            self.model = DistMult(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        if self.model_name == 'ComplEx':
            self.model = ComplEx(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0
            while not last_batch:
                batch = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio, neg_sampler=self.args.neg_sampler, device = self.device)
                last_batch = self.dataset.was_last_batch()
                optimizer.zero_grad()
                hs  = (batch[:,0]).clone().detach().long().to(self.device)
                rs   = (batch[:,1]).clone().detach().long().to(self.device)
                ts  = (batch[:,2]).clone().detach().long().to(self.device)
                scores = self.model.forward(hs, rs, ts)
                chunks = self.args.neg_ratio + 1
                pos_scores, neg_scores = scores[:batch.shape[0]//chunks], scores[batch.shape[0]//chunks:]
                loss = self.model._loss(pos_scores, neg_scores, self.args.neg_ratio)
                if self.args.reg != 0.0 :
                    loss += self.args.reg*self.model._regularization(batch[:, 0], batch[:, 1], batch[:, 2])
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.dataset.name + ")")
        
            if epoch % self.args.save_each == 0:
                self.save_model(self.model_name, epoch)

    def save_model(self, model, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/" + model + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), directory + str(chkpnt) + ".pt")