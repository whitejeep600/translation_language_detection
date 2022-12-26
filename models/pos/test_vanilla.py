import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import TranslationDataset
from model import TranslationClassifier


def main(args):
    # read dataset
    test_path = args.data_dir / 'test_pos.json'
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    dataset = TranslationDataset(test_data)
    
    # TODO: crecate DataLoader for test dataset
    dataloader_test = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    model = TranslationClassifier(args.in_dim, args.embed_dim, args.num_layers,
                        args.dropout, args.num_classes).to(args.device)
    model.eval()

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    result = {'arabic': {}, 'chinese': {}, 'indonesian': {}, 'japanese': {}}
    idx2lang = {0: 'arabic', 1: 'chinese', 2: 'indonesian', 3: 'japanese'}
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            batch['input'] = batch['input'].to(args.device)
            batch['gt'] = batch['gt'].to(args.device)
            
            # eval
            pred, loss = model(batch)
            
            # result
            pred = pred.detach().cpu()
            gt = batch['gt'].detach().cpu().view_as(pred)
            
            for i in range(len(pred)):
                lang = idx2lang[gt[i].item()]
                if batch['translator'][i] not in result[lang]:
                    result[lang][batch['translator'][i]] = {'correct': 0, 'cnt': 0}
                result[lang][batch['translator'][i]]['correct'] += int(pred[i] == gt[i])
                result[lang][batch['translator'][i]]['cnt'] += 1
    # result = {
        # 'arabic': {'mbart': {'correct': 220, 'cnt': 347}, 'helsinki': {'correct': 235, 'cnt': 353}},
        # 'chinese': {'mbart': {'correct': 369, 'cnt': 540}, 'helsinki': {'correct': 334, 'cnt': 556}},
        # 'indonesian': {'mbart': {'correct': 210, 'cnt': 348}, 'helsinki': {'correct': 227, 'cnt': 347}},
        # 'japanese': {'mbart': {'correct': 102, 'cnt': 350}, 'helsinki': {'correct': 135, 'cnt': 350}}
    # }
    C = 0
    N = 0
    mbart = [0, 0]
    helsinki = [0, 0]
    for k in result.keys():
        print(f'---{k}---')
        for k2 in result[k].keys():
            print(f'   -{k2}:')
            result[k][k2]['acc'] = result[k][k2]['correct'] / result[k][k2]['cnt']
            C += result[k][k2]['correct']
            N += result[k][k2]['cnt']
            
            if k2 == 'mbart':
                mbart[0] += result[k][k2]['correct']
                mbart[1] += result[k][k2]['cnt']
            else:
                helsinki[0] += result[k][k2]['correct']
                helsinki[1] += result[k][k2]['cnt']
            print(f'   {result[k][k2]}\n')
    
    print('\nmbart:', mbart[0]/mbart[1])
    print('helsinki:', helsinki[0]/helsinki[1])
    print('total:', C/N)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    # model
    parser.add_argument("--in_dim", type=int, default=18)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)

    # data loader
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
