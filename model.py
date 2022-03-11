from typing import Dict

import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence
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

        # TODO: model architecture
        #self.rnn = torch.nn.LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,bidirectional=bidirectional)
        self.rnn = torch.nn.GRU( input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,bidirectional=bidirectional)
        
        self.fc = torch.nn.Linear(hidden_size,num_class)

        # self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size)
        # self.fc2 = torch.nn.Linear(hidden_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:

        x = self.embed(batch)
        x = pack_padded_sequence(x, [len(data) for data in x], batch_first=True)

        #out_pack, (ht, ct) = self.rnn(x)
        out_pack, ht = self.rnn(x)

        # hidden = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)
        # hidden = self.fc1(hidden)
        # hidden = self.fc2(hidden)
  
        hidden = ht[-1,:,:]
        hidden = self.fc(hidden)

        return hidden
