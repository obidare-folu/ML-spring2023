
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.title_conv = nn.Conv1d(
            in_channels=hid_size,
            out_channels=hid_size,
            kernel_size=2
        )
        self.title_avgpool = nn.AdaptiveAvgPool1d(1)
        # self.title_lin = nn.Linear(in_features=hid_size, out_features=1)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.full_conv = nn.Conv1d(
            in_channels=hid_size,
            out_channels=hid_size,
            kernel_size=2
        )
        self.full_avgpool = nn.AdaptiveAvgPool1d(1)
        # self.full_lin = nn.Linear(in_features=hid_size, out_features=1)
        
        # self.category_out = # <YOUR CODE HERE>
        self.category_out = torch.nn.Linear(n_cat_features, hid_size)


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        # title = # <YOUR CODE HERE>
        title = F.relu(self.title_conv(title_beg))
        title = self.title_avgpool(title)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        # full = # <YOUR CODE HERE>
        full = F.relu(self.full_conv(full_beg))
        full = self.full_avgpool(full)

        category = F.relu(self.category_out(input3))
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        # out = # <YOUR CODE HERE>
        out = F.relu(self.inter_dense(concatenated))
        out = F.relu(self.final_dense(out))
        
        return out
  