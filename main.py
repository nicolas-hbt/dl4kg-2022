from trainer import Trainer
from tester import Tester
from dataset import Dataset
import numpy as np
import pandas as pd
import argparse
import time
import os
import torch

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=100, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-reg', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="EduKG", type=str, help="dataset")
    parser.add_argument('-model', default="TransE", type=str, help="knowledge graph embedding model")
    parser.add_argument('-emb_dim', default=10, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_sampler', default="rns", type=str, help="negative sampling strategy")
    parser.add_argument('-batch_size', default=514, type=int, help="batch size")
    parser.add_argument('-save_each', default=25, type=int, help="validate every k epochs")
    parser.add_argument('-criterion_validation', default="valid_loss", type=str, help="criterion for keeping best epoch")
    parser.add_argument('-metrics', default="sem", type=str, help="metrics to compute on test set (sem|ranks|all)")
    parser.add_argument('-pipeline', default="test", type=str, help="(train|test|both)")
    parser.add_argument('-device', default="cpu", type=str, help="(cpu|cuda:0)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset)
    model = args.model

    if args.pipeline == 'both' or args.pipeline == 'train':
        print("------- Training -------")
        start = time.time()
        trainer = Trainer(dataset, model, args)
        trainer.train()
        print("Training time: ", time.time() - start)

    if args.pipeline == 'both' or args.pipeline == 'test':
        print("------- Select best epoch on validation set -------")
        epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
        dataset = Dataset(args.dataset)
        
        best_mrr = -1.0
        best_loss = + np.inf
        results = {}
        best_epoch = "0"
        for epoch in epochs2test:
            print("Epoch n°", epoch)
            model_path = "models/" + args.dataset + "/" + model + "/" + epoch + ".pt"
            tester = Tester(dataset, args, model_path, "valid")
            start = time.time()
            if args.criterion_validation == 'valid_loss':
                valid_loss = tester.calc_valid_loss()
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
            if args.criterion_validation == 'mrr':
                filtered_mrr = tester.calc_valid_mrr()
                if filtered_mrr > best_mrr:
                    best_mrr = filtered_mrr
                    best_epoch = epoch
            if args.criterion_validation == 'mrr-sem':
                filtered_mrr, sem1, sem5, sem10 = tester.calc_valid_mrr_sem()
                if filtered_mrr > best_mrr:
                    best_mrr = filtered_mrr
                    best_epoch = epoch
                results[epoch] = {"Filtered MMR": filtered_mrr, "Sem@1": sem1, "Sem@5": sem5, "Sem@10": sem10}
                df = pd.DataFrame.from_dict(results) 
                df.to_csv ('valid_results-' + args.dataset + '-' + model +'.csv')
            print(time.time() - start)
        print("Best epoch: " + best_epoch)

        print("------- Testing on the best epoch -------")
        best_model_path = "models/" + args.dataset + "/" + model + "/" + best_epoch + ".pt"
        tester = Tester(dataset, args, best_model_path, "test")
        tester.test()