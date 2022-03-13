from typing import Dict

import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn as nn
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        self.rnn = nn.GRU( input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,bidirectional=bidirectional)
        
        self.num_class = num_class
        
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2,num_class)
        else:
            self.fc = nn.Linear(hidden_size,num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:

        x = self.embed(batch)

        out, _ = self.rnn(x)
    
        out = self.fc(out)

        return F.log_softmax(torch.squeeze(out, 0),dim=2)
