import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import Dataset, DataLoader
from model import SeqClassifier

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

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # Parameters tuning

    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    evals=DataLoader(datasets['eval'],batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets['eval'].collate_fn(x)))
    train=DataLoader(datasets['train'],batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets['train'].collate_fn(x)))

    with open(f"./{args.name}_result","w") as f:
        f.write(f">> Learning Rate: {args.lr}, max_len: {args.max_len}\n")
    
    best_acc=0
    for j in range(5):
        model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size,
                            num_layers=args.num_layers,dropout=args.dropout/2**j,bidirectional=args.bidirectional,num_class=len(intent2idx))
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        criterion.to(device)

        # Train loop
        _acc=0
        epoch_pbar = trange(args.num_epoch, desc="Epoch")
        for epoch in epoch_pbar:

            model.train()
            for labels, texts in train:

                out = model(texts)
                p_labels = torch.argmax(out, dim=1)

                loss = criterion(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            acc=0
            n=0
            for labels, texts in evals:
                out = model(texts)
                p_labels = torch.argmax(out, dim=1)
                acc=acc+torch.sum(p_labels == labels)
                n = n + len(labels)
            acc=acc.item()/n*100

            if acc > 80:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.5   

            # if acc > _acc :
            #     _acc = acc
            # elif _acc > 60:
            #     scheduler.step()

            if epoch % 100 == 0:
                # print(f"\nEpoch: {epoch:5d}, Accuracy {acc:.4f}%, LR={scheduler.get_last_lr()[0]}")
                print(f"\nEpoch: {epoch:5d}, Accuracy {acc:.4f}%")

        print("="*40)
        print(f"Dropout: {args.dropout/2**j:.4f}, Accuracy: {acc:.4f}%")
        with open(f"./{args.name}_result","a") as f:
            f.write(f"Dropout: {args.dropout/2**j:.4f}, Accuracy: {acc:.4f}%.\n")
        print("="*40)
        if acc > best_acc :
            best_acc = acc
            best_paras = model.state_dict()

        torch.save(best_paras,args.ckpt_dir / f"./{args.name}_best_model.pth")
    

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
    parser.add_argument("--max_len", type=int, default=18)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=3000)

    # output
    parser.add_argument("--name", type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
