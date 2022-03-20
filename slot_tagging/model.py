from typing import Dict

import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

        self.rnn = nn.LSTM( input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,bidirectional=bidirectional,batch_first=True)
        
        self.num_class = num_class
        
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2,num_class)
        else:
            self.fc = nn.Linear(hidden_size,num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch, length, device) -> Dict[str, torch.Tensor]:

        x = self.embed(batch.to(device))
        x = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)

        out, _ = self.rnn(x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        out = F.log_softmax(out,dim=2)
        
        return out

    def loss_and_acc(self,_y,y,criterion,device,acc1,acc2):

        loss=None

        for i in range(len(y)):

            yy = _y[i][:len(y[i])]

            if loss is None :
                loss = criterion(yy,torch.tensor(y[i]).to(device))
            else:
                loss += criterion(yy,torch.tensor(y[i]).to(device))

            py = torch.argmax(yy,dim=1).tolist()
            
            acc1.add(py==y[i])

            for j in range(len(py)):
                acc2.add(py[j]==y[i][j])

        return loss

    def predict_label(self,x,device):

        x = self.embed(x.to(device))
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = F.log_softmax(out,dim=2)

        prediction = torch.argmax(out,dim=2).view(-1).tolist()

        return prediction
