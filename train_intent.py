import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

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
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets

    train=DataLoader(datasets['train'],batch_size=128, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets['train'].collate_fn(x)))
    evals=DataLoader(datasets['eval'],batch_size=128, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in datasets['eval'].collate_fn(x)))

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size,num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(intent2idx))
    model.to(device)

    # TODO: init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:

        model.train()
        acc=0
        n=0
        for labels, texts in train:

            out = model(texts)
            p_labels = torch.argmax(out, dim=1)

            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc=acc+torch.sum(p_labels == labels)
            n = n + len(labels)

        acc=acc.item()/n*100

        if epoch % 50 == 0:
            print(f"Epoch: {epoch:5d}, Accuracy {acc:.4f}")
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        pass

    # TODO: Inference on test set


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
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
