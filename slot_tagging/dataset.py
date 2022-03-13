from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        texts=[]
        labels=[]
        data={}
        for sample in samples:
            texts.append(sample['tokens'])
            y=[self.label2idx(x) for x in sample['tags']]
            for i in range(self.max_len-len(y)): y.append(len(self.label_mapping))
            labels.append(y)
        texts=self.vocab.encode_batch(batch_tokens=texts)

        labels=torch.tensor(labels,dtype=torch.int64)
        texts=torch.tensor(texts,dtype=torch.int64)

        return labels, texts

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
