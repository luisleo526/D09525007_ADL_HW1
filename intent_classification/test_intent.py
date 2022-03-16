import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import csv  

def main(args):

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.to(torch.device(args.device))
    model.eval()
    
    x=[]
    seq_len=[]
    for data in dataset.data:
        text = data['text'].split()
        x.append(text)
        seq_len.append(len(text))
        
    x = dataset.vocab.encode_batch(batch_tokens=x)
    x = torch.tensor(x,dtype=torch.int64).to(args.device)
    seq_len = torch.tensor(seq_len,dtype=torch.int64).to(args.device)

    p_label = model(x,seq_len)
    p_label=torch.argmax(p_label, dim=1)

    with open(args.pred_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','intent'])
        n=0
        for label in p_label:
            writer.writerow([f"test-{n}",f"{dataset.idx2label(label.item())}"])
            n=n+1


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/intent_best_model.pth"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
