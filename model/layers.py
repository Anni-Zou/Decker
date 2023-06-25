import torch
import torch.nn as nn
import numpy as np

#============================================================Freeze============================================================
def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze_params(params):
    for p in params:
        p.requires_grad = False

def unfreeze_params(params):
    for p in params:
        p.requires_grad = True

#============================================================MLP============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, layer_norm=True):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, h_dim)
        self.layerNorm = nn.LayerNorm(h_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(h_dim, out_dim)
        self.activate = nn.Tanh()
    
    def forward(self, input):
        output = self.layer1(self.dropout(input))
        output = self.activate(output)
        if self.layerNorm:
            output = self.layerNorm(output)
        return self.layer2(output)

#============================================================Embeddings============================================================
class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim, padding_idx=0)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.fill_(0)
                self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = nn.GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale

#============================================================Pooling Methods============================================================
class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled

class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


class MatrixVectorScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)
        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, 1, d_k)*(n*b, l, d_k) -> (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)   #(n*b, l)
        output = (attn.unsqueeze(2) * v).sum(1) #(n*b, l, 1)*(n*b, l, d_v) -> (n*b, d_v)
        return output, attn


class AttPoolLayer(nn.Module):
    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the output dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v  #n_head must be divided by d_k -> n_head = 2or4

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, d_q_ori) -> (b, n_head*d_k) -> (b, n_head, d_k)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, d_k_ori) -> (b, l, n_head*d_k) -> (b, l, n_head, d_k)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n_head, d_v)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)    #(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)  #(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)  #(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.bool()  ##
            mask = ~mask        ##
            mask = mask.repeat(n_head, 1)   #(n_head*bs, len_k)
        output, attn = self.attention(qs, ks, vs, mask=mask)# (n*b, d_v), (n*b, l)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn

class BilinearAttentionLayer(nn.Module):
    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)
    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))  #(bs, seq_len, query_dim) * (bs, query_dim, 1) -> (bs, seq_len, 1)
        attn = self.softmax(attn.squeeze(-1))   #(bs, seq_len)
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)    #(bs, 1, seq_len) * (bs, seq_len, value_dim) -> (bs,1,value_dim) -> (bs, value_dim)
        return pooled, attn