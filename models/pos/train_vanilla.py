import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from time import time
from tqdm import tqdm
import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import TranslationDataset
from model import TranslationClassifier


def train(model, dataloader, optimizer, args):
    correct_train = 0
    N = 0
    for batch in tqdm(dataloader):
        batch['input'] = batch['input'].to(args.device)
        batch['gt'] = batch['gt'].to(args.device)
        optimizer.zero_grad()
        
        # train
        pred, loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # acc
        pred = pred.detach().cpu()
        gt = batch['gt'].detach().cpu()
        
        correct_train += (pred == gt.view_as(pred)).sum().item()
        N += gt.size(0)
    return correct_train / N

def evaluate(model_train_dict, model_eval, dataloader_eval, args):
    correct_eval = 0
    N = 0
    with torch.no_grad():
        model_eval.load_state_dict(model_train_dict)
        model_eval.eval()
        for batch in dataloader_eval:
            batch['input'] = batch['input'].to(args.device)
            batch['gt'] = batch['gt'].to(args.device)
            
            # eval
            pred, loss = model_eval(batch)
            
            # acc
            pred = pred.detach().cpu()
            gt = batch['gt'].detach().cpu()
            
            correct_eval += (pred == gt.view_as(pred)).sum().item()
            N += gt.size(0)
    return correct_eval / N

def main(args):
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # read dataset
    train_path = args.data_dir / 'train_pos.json'
    test_path = args.data_dir / 'test_pos.json'
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    datasets: Dict[str, TranslationDataset] = {
        'train': TranslationDataset(train_data),
        'test': TranslationDataset(test_data)
    }
    
    # crecate DataLoader
    dataloader_train = DataLoader(datasets['train'], args.batch_size, shuffle=True, collate_fn=datasets['train'].collate_fn)
    dataloader_eval = DataLoader(datasets['test'], args.batch_size, shuffle=True, collate_fn=datasets['test'].collate_fn)

    model = TranslationClassifier(args.in_dim, args.embed_dim, args.num_layers,
                        args.dropout, args.num_classes).to(args.device)
    model_eval = TranslationClassifier(args.in_dim, args.embed_dim, args.num_layers,
                        args.dropout, args.num_classes).to(args.device)
    
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_history_train = []
    acc_history_val = []
    lr = args.lr
    best_acc = 0.
    for epoch in range(args.num_epoch):
        start_time = time()
        model.train()
        
        # TODO: Training loop - iterate over train dataloader and update model weights
        acc_train = train(model, dataloader_train, optimizer, args)
        acc_history_train.append(acc_train)
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        acc_eval = evaluate(model.state_dict(), model_eval, dataloader_eval, args)
        acc_history_val.append(acc_eval)
        
        # saving data
        elapsed = (time() - start_time) / 60
        print('[%d] time %.2f lr %f train %f eval %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    acc_train,
                    acc_eval))
        
        if acc_eval >= best_acc:
            best_acc = acc_eval
            chk_path = os.path.join(args.checkpoint_dir, 'best.bin')
            print('Saving best checkpoint to', chk_path)
            torch.save(model.state_dict(), chk_path)
        
        # update params
        lr *= args.lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lrd
        
        if (epoch+1) % 20 == 0:
            chk_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.bin')
            print('Saving checkpoint to', chk_path)
            torch.save(model.state_dict(), chk_path)
        
        if args.export_training_curves and epoch > args.start_plot_idx:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(args.start_plot_idx, len(acc_history_train)) + 1
            plt.plot(epoch_x, acc_history_train[args.start_plot_idx:], '--', color='C0')
            plt.plot(epoch_x, acc_history_val[args.start_plot_idx:], '--', color='C1')
            plt.legend(['train', 'eval'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.xlim((args.start_plot_idx, epoch+1))
            plt.savefig(os.path.join(args.checkpoint_dir, 'acc.png'))
            plt.close('all')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./checkpoint/",
    )
    
    # model
    parser.add_argument("--in_dim", type=int, default=18)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrd", type=float, default=0.95)

    # data loader
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--start_plot_idx", type=int, default=1)
    parser.add_argument("--export_training_curves", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=5731)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main(args)
