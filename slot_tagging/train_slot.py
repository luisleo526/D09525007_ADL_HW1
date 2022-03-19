import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from dataset import SeqClsDataset
from utils import Vocab, Acc_counter
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from model import SeqClassifier

import torch
from tqdm import trange
import torch.optim as optim

from sklearn.model_selection import KFold

import torch.nn.functional as F

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):

    device = torch.device(args.device)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in ['train','eval']}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    data=[y for x in data.keys() for y in data[x] ]

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    datasets = SeqClsDataset(data, vocab, slot2idx, 36)

    data=[]
    for k in range(3):
        for j in range(4):
            for i in range(4):

                hidden_size = 128 * 2**i
                num_layers  = 2 + j
                batch_size  = 256 * 2**k
                dropout     = 0.1
                lr          = 0.1

                torch.manual_seed(24)

                if args.split > 1:
                    kf = KFold(n_splits=args.split)
                else:
                    kf = KFold(n_splits=100)

                fold=0
                f_acc={'token':0,'sentence':0}
                for train_ind, test_ind in kf.split(datasets):

                    fold+=1
                    
                    model = SeqClassifier(embeddings=embeddings,hidden_size=hidden_size,
                                num_layers=num_layers,dropout=dropout,bidirectional=args.bidirectional,num_class=len(slot2idx))

                    model.to(device)

                    # optimizer = [ optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) 
                    #              ,optim.SGD(model.parameters(), lr=lr, momentum=0.9) ]
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
                    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                    criterion = torch.nn.CrossEntropyLoss()
                    # criterion = torch.nn.NLLLoss()
                    criterion.to(device)

                    train_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=datasets.collate_fn,sampler=SubsetRandomSampler(train_ind))
                    test_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=datasets.collate_fn,sampler=SubsetRandomSampler(test_ind))

                    epoch_pbar = trange(args.num_epoch, desc="Epoch")

                    for epoch in epoch_pbar:

                        msg = train(model,[train_loader,test_loader],optimizer,criterion,device)

                        epoch_pbar.set_postfix(fold=f"{fold:d}/{kf.get_n_splits():d}",token=f"{msg['train']['token']:.4f}% / {msg['val']['token']:.4f}%",
                            sentence=f"{msg['train']['sentence']:.4f}% / {msg['val']['sentence']:.4f}%")

                    f_acc['token'] += msg['val']['token'] / kf.get_n_splits()
                    f_acc['sentence'] += msg['val']['sentence'] / kf.get_n_splits()

                    if args.split < 1: 
                        torch.save(model.state_dict(),args.ckpt_dir / "intent_best_model.pth")
                        quit()

                data.append( { 'params':{'hidden_size':hidden_size,'num_layers':num_layers,'batch_size':batch_size,
                    'dropout':dropout,'lr':lr},'token_acc':f_acc['token'],'sentence_acc':f_acc['sentence'] } )

                info=f"Dropout:{dropout:.4f}, hidden_size:{hidden_size:d}, layers:{num_layers:d}, batch_size:{batch_size:d}, LR={lr:.4E}\n"
                info+=f"Accuracy: {f_acc['token']:.4f}% / {f_acc['sentence']:.4f}%"

                print("="*40)
                print(info)
                print("="*40)

    with open(f"{args.name}.json", 'w') as f:
        json.dump(data, f)

def train(model,dataloader,optimizer,criterion,device):

    msg={}

    token_acc = Acc_counter()
    sent_acc = Acc_counter()

    model.train()
    for y, x, length in dataloader[0]:

        _y = model(x,length,device)
        loss = model.loss_and_acc(_y,y,criterion,device,sent_acc,token_acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msg['train'] = {'token':token_acc.out(),'sentence':sent_acc.out()}

    if len(dataloader) > 1:

        token_acc = Acc_counter()
        sent_acc = Acc_counter()

        model.eval()         
        for y, x, length in dataloader[1]:
            _y = model(x,length,device)
            loss = model.loss_and_acc(_y,y,criterion,device,sent_acc,token_acc)

    msg['val'] = {'token':token_acc.out(),'sentence':sent_acc.out()}

    return msg
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)  #1024
    parser.add_argument("--num_layers", type=int, default=2)     #3
    parser.add_argument("--dropout", type=float, default=0.25)   #0.01
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)   #128

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--split",type=int,default=5)

    # output
    parser.add_argument("--name", type=str, default="default_name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
