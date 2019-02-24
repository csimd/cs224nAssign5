import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(torch.nn.Module):
    def __init__(self, e_word):

        super(Highway, self).__init__()

        self.e_word = e_word
        self.weights_proj = torch.nn.Linear(e_word, e_word, bias=True)
        self.weights_gate = torch.nn.Linear(e_word, e_word, bias=True)

    def forward(self, x_convout):
    	x_proj = F.relu(self.weights_proj(x_convout))
    	x_gate = self.weights_gate(x_convout).sigmoid()
    	x_highway = torch.mul(x_proj, x_gate) + torch.mul(torch.add(torch.mul(x_gate, -1), 1), x_convout)
    	return x_highway
