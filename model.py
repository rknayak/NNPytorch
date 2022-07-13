
import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in=69*69, n_hidden=512, n_out=10):
        super().__init__()
        
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.layer1 = nn.Linear(n_in,n_hidden)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden,n_hidden)
        self.activation2 = nn.ReLU()
#         self.layer3 = nn.Linear(n_hidden,n_hidden)
#         self.activation3 = nn.ReLU()
        self.layer4 = nn.Linear(n_hidden,n_out)
        
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)
        
    
    def forward(self,x):
        
        # add hidden layer, with relu activation function
        out = self.layer1(x)
        out = self.activation1(out)
        out = self.dropout(out)
        
        # add hidden layer, with relu activation function
        out = self.layer2(out)
        out = self.activation2(out)
        
#         # add hidden layer, with relu activation function
#         out = self.layer3(out)
#         out = self.activation3(out)
#         out = self.dropout(out)

        # add output layer
        out = self.layer4(out)
        
        return out