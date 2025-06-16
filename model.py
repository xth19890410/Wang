# -*- coding: utf-8 -*-
# @Time : 2022/11/1 20:07
# @Author : Teng Qing

from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from statistics import mean
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
device = torch.device('cuda')

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def ortho_norm(weight):
    wtw = torch.mm(weight.t(), weight) + 1e-4 * torch.eye(weight.shape[1]).to(weight.device)
    L = torch.linalg.cholesky(wtw)
    weight_ortho = torch.mm(weight, L.inverse().t())
    return weight_ortho
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, activation=torch.tanh):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

        self.weight = glorot_init(in_features, out_features)
        self.ortho_weight = torch.zeros_like(self.weight)
        self.activation = activation

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        self.ortho_weight = ortho_norm(self.weight)
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        # output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = theta * torch.mm(support, self.ortho_weight) + (1 - theta) * r
        if self.residual:
            output = output+input
        return self.activation(output)


class GCN(nn.Module):
    #初始化操作
    def __init__(self, nfeat, nhidden, nclass, dropout):
        super(GCN, self).__init__()
        for ij in range(len(nfeat)):
            nfeat_j = nfeat[ij]
        self.gc1 = GraphConvolution(nfeat_j, nhidden)
        self.gc2 = GraphConvolution(nhidden, nclass)
        self.dropout = dropout

    #前向传播
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MAUGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nhidden1, nclass, dropout, lamda, alpha, lamda1, variant, m=0.5):
        super(MAUGCN, self).__init__()

        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant))
        self.layer = nlayers
        self.fcs = nn.ModuleList()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        for ij in range(len(nfeat)):
            nfeat_j = nfeat[ij]
            self.fcs.append(nn.Linear(nfeat_j, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        # self.catt= SelfAttention(nhidden)


        # self.att1 = GraphAttentionLayer(nhidden, nhidden,dropout,alpha,concat=True)
        # self.att2 = SelfAttentionWide(nhidden, heads=8, mask=False)

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        # self.params3 = list(self.catt.parameters())

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.lamda1 = lamda1
    def forward(self, x, adj):
        layer_fcs = []
        outputtsum = 0
        outputs = []
        outputts_out = []
        output_allcons = []
        for k in range(len(adj)):
            oth = len(adj)+1
            adjj = adj[k]
            input = x[k]

            # x = torch.log(torch.from_numpy(np.array(x.cpu(),np.float)))
            input = F.dropout(input, self.dropout, training=self.training)
            # x = x.to(device)
            # x = x.to(torch.float32)
            layer_inner = self.act_fn(self.fcs[k](input))
            layer_fcs.append(layer_inner)
            att_nid = []
            att_nid_sum = 0

            output_cons = []

            for i, con in enumerate(self.convs):
                # content = self.att1(layer_inner,adjj)
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                #1

                if k >=1 :
                    layer_inner = self.w * layer_inner + (1-self.w) * output_allcons[self.layer*(k-1)+i]

                layer_inner = self.act_fn(con(layer_inner, adjj, layer_fcs[k], self.lamda, self.alpha, i + 1))#GCNII
                # layer_inner = self.act_fn(con(layer_inner, adjj))#GCN
                # layer_inner = self.lamda1*layer_inner+(1-self.lamda1)*content
                # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                # layer_inner = self.act_fn(self.att2(layer_inner))

                output_cons.append(layer_inner)
                output_allcons.append(layer_inner)

                att_nid.append((torch.spmm(adjj, layer_inner).detach().cpu().numpy().mean()))
                att_nid_sum = att_nid_sum+att_nid[i]
            outputts = 0
            for ii in range(len(att_nid)):
                # nid = output_cons[ii]*(att_nid[ii]/att_nid_sum)
                nid = output_cons[ii]
                outputts = outputts+nid
            outputts= F.dropout(outputts, self.dropout, training=self.training)
            outputs.append(outputts)
        for kj in range(len(adj)):
            layer_output = self.fcs[-1](outputs[kj])
            outputtsum = outputtsum +layer_output
            outputts_out.append(F.log_softmax(layer_output,dim=1))

        outputsmean = torch.mean(torch.stack(outputts_out[0:len(adj)]), dim=0, keepdim=True)
        return F.log_softmax(outputtsum,dim=1),outputsmean.squeeze(0),outputts_out,self.w