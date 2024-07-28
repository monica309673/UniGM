import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

class FuseLayer(nn.Module):
    r"""Applies an fuse layer to the incoming data:.`
    """
    __constants__ = ['in_dim', 'out_dim', 'experts']

    def __init__(self, in_dim, out_dim, experts, freeze, type_layer, bias=False, align=True, lite=False):
        super(FuseLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.experts = experts
        self.align = align
        self.lite = lite
        self.type_layer = type_layer

        if type_layer == 'linear':
            if freeze == 1:
                self.weight = nn.Parameter(torch.randn(experts, out_dim, in_dim), requires_grad=False)
            elif freeze == 0:
                self.weight = nn.Parameter(torch.randn(experts, out_dim, in_dim), requires_grad=True)       

        elif type_layer == 'embedding':
            self.weight = nn.Parameter(torch.randn(experts, in_dim, out_dim), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(experts, out_dim))
        else:
            self.bias = None
        self.align = True 
        if self.align:
            align_conv = torch.zeros(self.experts, self.experts, 1, 1)
            for i in range(self.experts):
                align_conv[i, i, 0, 0] = 1.0
            self.align_conv = nn.Parameter(align_conv, requires_grad=True) 
        else:
            self.align = False

        # attention layer
        self.attention_up = nn.Sequential(
            nn.AdaptiveAvgPool1d(1)
            )
        self.attention_down = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, experts),
            nn.Flatten(),
            nn.Softmax()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.attention_up:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.attention_up:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.attention_down:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for i in range(self.experts):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, rnn):
        x_ori = x
        x = torch.unsqueeze(x, 1) # B * 1 * E
        x = self.attention_up(x) # B * 1 * 1
        x, _ = rnn(x) #  x: B * 1 * H
        sigmoid_attention = self.attention_down(x)  
        if self.align:
            weight = F.conv2d(self.weight.unsqueeze(0), weight=self.align_conv, bias=None, stride=1, padding=0, dilation=1)
            weight = weight.squeeze()
        else:
            weight = self.weight

        fuse_weight = torch.einsum('be,eij->bij', sigmoid_attention, weight)

        if self.type_layer == 'embedding':
            y = F.embedding(x_ori, weight = fuse_weight)
        elif self.type_layer == 'linear':
            if self.bias is not None:
                fuse_bias = torch.einsum('be,eo->bo', sigmoid_attention, self.bias)
                y = torch.einsum('be,boe->bo', x_ori, fuse_weight) + fuse_bias
            else:
                y = torch.einsum('be,boe->bo', x_ori, fuse_weight)
        return y
