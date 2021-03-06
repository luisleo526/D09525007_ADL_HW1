import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from dataset2 import SeqClsDataset
from utils import Vocab, Acc_counter
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from model2 import SeqClassifier

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import torch
from tqdm import trange
import torch.optim as optim

from sklearn.model_selection import KFold

import torch.nn.functional as F
import csv  
import os
import sys

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
    data=[x for x in data if len(x['tokens']) <= args.max_len ]
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    datasets = SeqClsDataset(data, vocab, slot2idx, args.max_len)

    if not args.predict:
        data=[]
        history=[]
        for k in range(1):
            for j in range(1):
                for i in range(1):

                    hidden_size = args.hidden_size
                    num_layers  = args.num_layers
                    batch_size  = args.batch_size
                    dropout     = args.dropout
                    lr          = args.lr

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
                        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=0.05)
                        # optimizer = optim.SGD(model.parameters(), lr=lr)

                        criterion = torch.nn.CrossEntropyLoss()
                        # criterion = torch.nn.NLLLoss()
                        criterion.to(device)

                        train_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=datasets.collate_fn,sampler=SubsetRandomSampler(train_ind))
                        test_loader=DataLoader(datasets,batch_size=batch_size,shuffle=False,collate_fn=datasets.collate_fn,sampler=SubsetRandomSampler(test_ind))

                        epoch_pbar = trange(args.num_epoch, desc="Epoch")

                        for epoch in epoch_pbar:

                            msg = train(model,[train_loader,test_loader],optimizer,criterion,device)

                            epoch_pbar.set_postfix(fold=f"{fold:d}/{kf.get_n_splits():d}",token=f"{msg['train']['token']:.4f}% / {msg['test']['token']:.4f}%",
                                sentence=f"{msg['train']['sentence']:.4f}% / {msg['test']['sentence']:.4f}%")

                            for s in ['train','test']:
                                msg[s]['epoch'] = epoch
                                msg[s]['fold'] = fold
                                msg[s]['type'] = s
                                history.append(msg[s])

                        # ### seqeval evaluation
                        # model.eval()
                        # true_y=[]
                        # p_y=[]
                        # for y, x, length in test_loader:

                        #     _y = model(x,length,args.device)
                        #     py=[]
                        #     for i in range(len(y)):
                        #         yy = _y[i][:len(y[i])]
                        #         ppy = torch.argmax(yy,dim=1).tolist()
                        #         py.append(ppy)

                        #     tmp=[]
                        #     for yy in y:
                        #         t=[]
                        #         for s in yy:
                        #             t.append(datasets.idx2label(s))
                        #         tmp.append(t)

                        #     true_y += tmp

                        #     tmp=[]
                        #     for yy in py:
                        #         t=[]
                        #         for s in yy:
                        #             t.append(datasets.idx2label(s))
                        #         tmp.append(t)
                        #     p_y += tmp

                        # print( classification_report(true_y, p_y, mode='strict', scheme=IOB2) )

                        f_acc['token'] += msg['test']['token'] / kf.get_n_splits()
                        f_acc['sentence'] += msg['test']['sentence'] / kf.get_n_splits()

                        if args.split < 1: 
                            torch.save(model.state_dict(),args.ckpt_dir / "slot_best_model.pth")
                            sys.exit()

                    data.append( { 'params':{'hidden_size':hidden_size,'num_layers':num_layers,'batch_size':batch_size,
                        'dropout':dropout,'lr':lr},'token_acc':f_acc['token'],'sentence_acc':f_acc['sentence'] } )

                    info=f"Dropout:{dropout:.4f}, hidden_size:{hidden_size:d}, layers:{num_layers:d}, batch_size:{batch_size:d}, LR={lr:.4E}\n"
                    info+=f"Accuracy: {f_acc['token']:.4f}% / {f_acc['sentence']:.4f}%"

                    print("="*40)
                    print(info)
                    print("="*40)

        with open(f"{args.name}.json", 'w') as f:
            json.dump(history, f)

    else:
        model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size,
                num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,num_class=len(slot2idx))
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt)
        model.to(device)
        predict(args,model,vocab,slot2idx)

def predict(args,model,vocab,slot2idx):

    test_file = args.data_dir / "test.json"
    test_datasets = SeqClsDataset(json.loads(test_file.read_text()), vocab, slot2idx, args.max_len)

    model.eval()
    with open(args.pred_file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','tags'])
        for sample in test_datasets.data:
            x = torch.tensor( [test_datasets.vocab.encode(sample['tokens'])] )
            prediction = model.predict_label(x,args.device)
            msg = " ".join([test_datasets.idx2label(i) for i in prediction])
            writer.writerow([sample['id'],msg])

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

    msg['test'] = {'token':token_acc.out(),'sentence':sent_acc.out()}

    return msg
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path",type=Path,default="./ckpt/slot/slot_best_model.pth")
    parser.add_argument("--predict",type=bool,default=False)

    parser.add_argument("--test_file",type=Path, default="pred.tags.csv")
    parser.add_argument("--pred_file",type=Path, default="pred.tags.csv")
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
    parser.add_argument("--max_len", type=int, default=36)

    # model
    parser.add_argument("--hidden_size", type=int, default=384)  #1024
    parser.add_argument("--num_layers", type=int, default=2)     #3
    parser.add_argument("--dropout", type=float, default=0.2)   #0.01
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-1)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)   #128

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=25)
    parser.add_argument("--split",type=int,default=5)

    # output
    parser.add_argument("--name", type=str, default="default_name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
