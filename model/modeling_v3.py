import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout
from transformers import DebertaV2PreTrainedModel, DebertaV2Model, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput
from transformers.utils import logging
from model import modeling_graph
from model import layers
logger = logging.get_logger(__name__)

class MyConFact_Roberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        self.LMEncoder = RobertaModel(config)
        self.cxtpooler = ContextPooler(config)
        output_dim = self.cxtpooler.output_dim
        self.cptEmbedding = layers.CustomizedEmbedding(799273, output_dim, output_dim, use_contextualized=False, pretrained_concept_emb=None, freeze_ent_emb=True, scale=1, init_range=0.02)
        #self.cptEmbedding = nn.Embedding(799273, output_dim, padding_idx=0)
        #self.cptPooler = nn.Linear(1024, output_dim)    #transform cp_embs from ConceptNet to model hidden size
        self.GraphEncoder = modeling_graph.RGCN(output_dim, 2, 2, 0.1)
        self.finalPooler = layers.BilinearAttentionLayer(output_dim, output_dim)
        #self.finalPooler = layers.MultiheadAttPoolLayer(2, output_dim, output_dim, 0.1)
        self.mlp = layers.MLP(2*output_dim, output_dim, num_labels, dropout=0.1, layer_norm=True)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
  
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, fact_ids, fact_mask, fact_type,         
                        concept_ids, adj_cpt2cpt, adj_cpt2fct, graph_mask, labels=None):
        '''
        input_ids, attention_mask, token_type_ids:  (bs, seq)
        fact_ids, fact_mask, fact_type:             (bs, n_facts, fct_length)
        concept_ids:                (bs, cpt_nodes) 
        adj_cpt2cpt, adj_cpt2fct, graph_mask: (bs, n_nodes, n_nodes), (bs, n_nodes, n_nodes), (bs, n_nodes) 
        '''
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## Question Encoding
        question_output = self.LMEncoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            #output_attentions = True,
            #output_hidden_states = False,
        )   #(last_hidden_states, hidden_states, attentions): ((bs,seq,dim), (), ())
        question_output = question_output[0]    #(bs,seq,dim)
        question_output = self.cxtpooler(question_output)  #(bs, dim)

        ## Facts Encoding
        bsz, n_facts, _ = fact_ids.size()
        _, max_nodes = concept_ids.size()
        flat_fact_ids = fact_ids.view(-1, fact_ids.size(-1)) if fact_ids is not None else None  #(bs*n_fact, fct_length)
        flat_fact_mask = fact_mask.view(-1, fact_mask.size(-1)) if fact_mask is not None else None  #(bs*n_fact, fct_length)
        flat_fact_type = fact_type.view(-1, fact_type.size(-1)) if fact_type is not None else None  #(bs*n_fact, fct_length)
        facts_outputs = self.LMEncoder(flat_fact_ids, token_type_ids=flat_fact_mask, attention_mask=flat_fact_type)
        facts_output = facts_outputs[0]  #(bs*n_fact, fct_length, dim)
        facts_output = self.cxtpooler(facts_output)    #(bs*n_fact, dim)
        facts_output = facts_output.view(bsz, n_facts, -1)  #(bs, n_fact, dim)


        ## Graph Encoding
        inputs, adj = self.create_graph(concept_ids, adj_cpt2cpt, adj_cpt2fct, facts_output)

        #inputs, adj, mask = self.create_rgcn(concept_ids, cpt2cpt_info, cpt2fct_info, num_actual, facts_output)   
        graph_output = self.GraphEncoder(inputs, adj)   #(bs, n_node, dim)
        pooled_output, _ = self.finalPooler(question_output, graph_output, graph_mask) #(bs, dim) & (bs, n_node, dim) -> (bs, dim)


        concat = torch.cat((pooled_output, question_output),1)   #(bs, 2*dim)
        logits = self.mlp(concat)


        loss = None
        print(labels)
        print(logits)
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = torch.nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    print(labeled_logits)
                    print(labels)
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = torch.nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        output = (logits,)
        return ((loss,) + output) if loss is not None else output

    def create_graph(self, concept_ids, adj_cpt2cpt, adj_cpt2fct, facts_output):
        '''
        returns:
            inputs: tensor of shape (bs, n_nodes, d)
            adj: tensor of shape (bs, n_head, n_nodes, n_nodes)
        '''
        adj = torch.stack((adj_cpt2cpt,adj_cpt2fct), dim=1) #(bs, n_head, n_node, n_node)
        cp_embs = self.extract_cpt_embed(concept_ids)   #(bs, cpt_nodes, dim)
        inputs = torch.concat((cp_embs,facts_output),dim=1) #(bs, n_nodes, dim)
        return inputs, adj

    def extract_cpt_embed(self, concept_ids):
        bs, max_nodes = concept_ids.size()
        cpt_index = concept_ids.flatten()   #(bs*max_nodes)
        cpt_index[cpt_index==-1] = 0
        extracted_emb = self.cptEmbedding(cpt_index, None)
        extracted_emb = extracted_emb.view(bs, max_nodes, -1)
        return extracted_emb

    def load_cpt_embs(self):
        ent_emb_path = './data/cpnet/tzw.ent.npy'
        cp_emb = [np.load(ent_emb_path)]
        cp_emb = np.concatenate(cp_emb, 1)
        cp_emb = torch.tensor(cp_emb, dtype=torch.float)
        return cp_emb

class MyConFact_Deberta(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.init_range = getattr(config, "initializer_range", 0.02)
        self.num_labels = num_labels
        self.deberta = DebertaV2Model(config)
        #layers.freeze_net(self.deberta)
        self.cxtpooler = ContextPooler(config)
        output_dim = self.cxtpooler.output_dim
        self.cptEmbedding = layers.CustomizedEmbedding(799273, output_dim, output_dim, use_contextualized=False, pretrained_concept_emb=None, freeze_ent_emb=True, scale=1, init_range=0.02)
        #self.cptEmbedding = nn.Embedding(799273, output_dim, padding_idx=0)
        #self.cptPooler = nn.Linear(1024, output_dim)    #transform cp_embs from ConceptNet to model hidden size
        self.GraphEncoder = modeling_graph.RGCN(output_dim, 3, 2, 0.1)
        #self.finalPooler = layers.BilinearAttentionLayer(output_dim, output_dim)
        self.finalPooler = layers.MultiheadAttPoolLayer(2, output_dim, output_dim, 0.1)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        #self.classifier = nn.Linear(2*output_dim, num_labels)
        self.mlp = layers.MLP(3*output_dim, output_dim, num_labels, dropout=drop_out, layer_norm=True)

        # Initialize weights and apply final processing
        self.post_init()
        
    

    def forward(self, input_ids, attention_mask, fact_input_ids, fact_attention_mask,         
                        concept_ids, adj_cpt2cpt, adj_cpt2fct, adj_q2others, graph_mask, labels=None):
        '''
        input_ids, attention_mask:  (bs, seq_len)
        fact_input_ids, fact_attention_mask: (bs, n_facts, seq_len)
        concept_ids:                (bs, cpt_nodes) 
        adj_cpt2cpt, adj_cpt2fct, graph_mask: (bs, n_nodes, n_nodes), (bs, n_nodes, n_nodes), (bs, n_nodes) 
        '''
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## Question Encoding
        question_output = self.deberta(input_ids, attention_mask=attention_mask)  #(bs, seq, dim), (), ()
        #(last_hidden_states, hidden_states, attentions): ((bs,seq,dim), (), ())
        question_output = question_output[0]    #(bs,seq,dim)
        question_output = self.cxtpooler(question_output)  #(bs, dim)

        ## Facts Encoding
        bsz, n_facts, _ = fact_input_ids.size()
        for i in range(n_facts):
            current_fact_output = self.deberta(fact_input_ids[i], attention_mask=fact_attention_mask[i])   #
        flat_fact_ids = fact_input_ids.view(-1, fact_input_ids.size(-1)) if fact_input_ids is not None else None  #(bs*n_fact, fct_length)
        flat_fact_mask = fact_attention_mask.view(-1, fact_attention_mask.size(-1)) if fact_attention_mask is not None else None  #(bs*n_fact, fct_length)
        facts_outputs = self.deberta(flat_fact_ids, attention_mask=flat_fact_mask)
        facts_output = facts_outputs[0]  #(bs*n_fact, fct_length, dim)
        facts_output = self.cxtpooler(facts_output)    #(bs*n_fact, dim)
        facts_output = facts_output.view(bsz, n_facts, -1)  #(bs, n_fact, dim)

        ## Graph Encoding
        inputs, adj = self.create_graph(concept_ids, adj_cpt2cpt, adj_cpt2fct, adj_q2others, question_output, facts_output)

        graph_output = self.GraphEncoder(inputs, adj)   #(bs, n_node, dim)
        interation_output = graph_output[:,0]   #(bs, dim)
        pooled_output, _ = self.finalPooler(question_output, graph_output, graph_mask) #(bs, dim) & (bs, n_node, dim) -> (bs, dim)

        concat = torch.cat((question_output, pooled_output, interation_output),1)   #(bs, 3*dim)
        #logits = self.dropout(concat)
        logits = self.mlp(concat)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = torch.nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = torch.nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()
        output = (logits,)
        return ((loss,) + output) if loss is not None else output

    def create_graph(self, concept_ids, adj_cpt2cpt, adj_cpt2fct, adj_q2others, question_output, facts_output):
        '''
        returns:
            inputs: tensor of shape (bs, n_nodes, d)
            adj: tensor of shape (bs, n_head, n_nodes, n_nodes)
        '''
        adj = torch.stack((adj_q2others, adj_cpt2cpt,adj_cpt2fct), dim=1) #(bs, n_head, n_node, n_node) <- n_head=3
        cp_embs = self.extract_cpt_embed(concept_ids)   #(bs, cpt_nodes, dim)
        question_output = question_output.unsqueeze(1)  #(bs, 1, dim)
        inputs = torch.concat((question_output,cp_embs,facts_output),dim=1) #(bs, n_nodes, dim)
        return inputs, adj

    def extract_cpt_embed(self, concept_ids):
        #concept_ids: (bs, max_nodes)
        #num_actual: (bs ,3) <- (num_cpt, num_cptrel, num_fctrel)
        bs, max_nodes = concept_ids.size()
        cpt_index = concept_ids.flatten()   #(bs*max_nodes)
        cpt_index[cpt_index==-1] = 0
        extracted_emb = self.cptEmbedding(cpt_index, None)
        extracted_emb = extracted_emb.view(bs, max_nodes, -1)
        #cp_emb = self.load_cpt_embs()   #(799273, 1024)
        #cpt_index = cpt_index.to(cp_emb.device) #move to cpu
        #cp_emb = cp_emb.to(cpt_index.device)
        #print(cpt_index)
        #extracted_emb = cp_emb.index_select(0, cpt_index)   #(bs*max_nodes, hdim=1024)
        #extracted_emb = extracted_emb.to(concept_ids.device)
        #extracted_emb = extracted_emb.view(bs, max_nodes, -1)   #(bs, max_nodes, hdim=1024)
        #extracted_emb = self.cptPooler(extracted_emb)   #(bs, max_nodes, hdim)
        return extracted_emb

    def load_cpt_embs(self):
        ent_emb_path = './data/cpnet/tzw.ent.npy'
        cp_emb = [np.load(ent_emb_path)]
        cp_emb = np.concatenate(cp_emb, 1)
        cp_emb = torch.tensor(cp_emb, dtype=torch.float)
        return cp_emb

