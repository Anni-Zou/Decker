import torch
import torch.nn as nn
import numpy as np


class RGCNLayer(nn.Module):

    def __init__(self, n_head, input_size, output_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.output_size = output_size

        assert input_size == output_size

        self.w_vs = nn.Parameter(torch.zeros(input_size, output_size * n_head))
        nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (input_size + output_size)))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, normalized_adj_t):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        normalized_adj_t: tensor of shape (b_sz*n_head, n_node, n_node)
            normalized_adj_t[:, j, i] ==  1/n indicates a directed edge i --> j and in_degree(j) == n
        """

        o_size, n_head = self.output_size, self.n_head
        bs, n_node, _ = inputs.size()

        w_vs = self.w_vs
        output = inputs.matmul(w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size
        output = output.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, o_size)  # (b_sz*n_head) x n_node x o_size
        normalized_adj_t = normalized_adj_t.to(output.device)
        output = normalized_adj_t.bmm(output).view(bs, n_head, n_node, o_size).sum(1)  # b_sz x n_node x dv
        output = self.activation(output)
        output = self.dropout(output)
        return output


class RGCN(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([RGCNLayer(num_heads, input_size, input_size,
                                               dropout) for l in range(num_layers + 1)])

    def forward(self, inputs, adj):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        adj: tensor of shape (b_sz, n_head, n_node, n_node)
            we assume the identity matrix representating self loops are already added to adj
        returns: 
            tensor of shape (b_sz, n_node, dv)
        """
        bs, n_head, n_node, _ = adj.size()

        in_degree = torch.max(adj.sum(2), adj.new_ones(()))
        adj_t = adj.transpose(2, 3)
        normalized_adj_t = (adj_t / in_degree.unsqueeze(3)).view(bs * n_head, n_node, n_node)
        assert ((torch.abs(normalized_adj_t.sum(2) - 1) < 1e-5) | (torch.abs(normalized_adj_t.sum(2)) < 1e-5)).all()
        output = inputs
        for layer in self.layers:
            output = layer(output, normalized_adj_t)
        return output