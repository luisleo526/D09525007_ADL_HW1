import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from model import SeqClassifier

from sklearn.model_selection import KFold

import torch.optim as optim

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):

    device = torch.device(args.device)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    data=[y for x in data.keys() for y in data[x] ]

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    datasets = SeqClsDataset(data, vocab, intent2idx, args.max_len)

    with open(f"./{args.name}_result","w") as f:
        f.write(f">> Learning Rate: {args.lr}, max_len: {args.max_len}\n")

    if args.split > 1:

        for i in range(1,5):

            hidden_size = args.hidden_size
            num_layers = args.num_layers
            batch_size = args.batch_size * 2**i

            torch.manual_seed(12)
            kf = KFold(n_splits=args.split)
            fold=0
            f_acc=0
            for train_ind, test_ind in kf.split(datasets):

                fold+=1
                model = SeqClassifier(embeddings=embeddings,hidden_size=hidden_size,
                            num_layers=num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(intent2idx))

                model.to(device)
                optimizer = [ optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) 
                             ,optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) ]
                #criterion = torch.nn.CrossEntropyLoss()
                criterion = torch.nn.NLLLoss()
                criterion.to(device)

                train_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets.collate_fn(x)),sampler=SubsetRandomSampler(train_ind))
                test_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets.collate_fn(x)),sampler=SubsetRandomSampler(test_ind))

                epoch_pbar = trange(args.num_epoch, desc="Epoch")
                for epoch in epoch_pbar:

                    if len(optimizer) > 1:
                        for g in optimizer[-1].param_groups:
                            g['lr'] = args.lr * 0.2 ** ( epoch // 4 )

                    tacc,acc = train(model,[train_loader,test_loader],optimizer,criterion)

                    epoch_pbar.set_postfix(fold=f"{fold:d}/{kf.get_n_splits():d}",Acc=f"{tacc:.4f}% / {acc:.4f}%")

                f_acc += acc / args.split

            info=f"Dropout:{args.dropout:.4f}, hidden_size:{hidden_size:d}, layers:{num_layers:d}, batch_size:{batch_size:d}\n"
            info+=f"Accuracy: {f_acc:.2f}%"
            print("="*40)
            print(info)
            with open(f"./{args.name}_result","a") as f:
                f.write(f"{info}\n")
            print("="*40)

    else:

        model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size,
                        num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(intent2idx))

        model.to(device)
        optimizer = [ optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) 
                     ,optim.SGD(model.parameters(), lr=args.lr) ]
        criterion = torch.nn.NLLLoss()
        criterion.to(device)

        train_loader=DataLoader(datasets,batch_size=args.batch_size,shuffle=False,collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets.collate_fn(x)))

        epoch_pbar = trange(args.num_epoch, desc="Epoch")
        for epoch in epoch_pbar:

            if len(optimizer) > 1:
                for g in optimizer[-1].param_groups:
                    g['lr'] = args.lr * 0.2 ** ( epoch // 4 )

            tacc = train(model,[train_loader],optimizer,criterion)
            epoch_pbar.set_postfix(Acc=f"{tacc:.4f}%")

        torch.save(model.state_dict(),args.ckpt_dir / "intent_best_model.pth")


def train(model,dataloader,optimizer,criterion):

    f = torch.nn.LogSoftmax(dim=1)

    model.train()
    tacc=0;n=0
    for labels, texts in dataloader[0]:

        out = model(texts)
        out = f(out)
        p_labels = torch.argmax(out, dim=1)
        
        tacc+=torch.sum(p_labels==labels)
        n+=len(labels)

        loss = criterion(out,labels)
        for opt in optimizer: opt.zero_grad()
        loss.backward()
        for opt in optimizer: opt.step()

    tacc=tacc.item()/n*100
        
    if len(dataloader) > 1:

        model.eval()         
        acc=0;n=0
        for labels, texts in dataloader[1]:
            out = model(texts)
            out = f(out)
            p_labels = torch.argmax(out, dim=1)
            acc=acc+torch.sum(p_labels == labels)
            n = n + len(labels)
        acc=acc.item()/n*100

    if len(dataloader) > 1:
        return tacc,acc
    else:
        return tacc
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)  #1024
    parser.add_argument("--num_layers", type=int, default=3)     #3
    parser.add_argument("--dropout", type=float, default=0.01)   #0.01
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)   #128

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=18)
    parser.add_argument("--split",type=int,default=5)

    # output
    parser.add_argument("--name", type=str, default="default_name")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
