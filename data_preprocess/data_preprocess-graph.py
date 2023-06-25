import networkx as nx
import json
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import string
import nltk
import os
from collections import Counter
translator = str.maketrans('', '', string.punctuation)
nltk_stopwords = nltk.corpus.stopwords.words('english')

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]



concept2id = None       # {word0:0, word1:1, ...}
id2concept = None       # [word0, word1, ...]
relation2id = None      # {rel0:0, rel1:1, ...}
id2relation = None      # [rel0, rel1, ...]

cpnet = None            # MultiDiGraph with 784110 nodes and 4302606 edges
cpnet_simple = None     # with 784110 nodes and 2004395 edges (merge cpnet with weights of edges)
max_cpt_nodes = 20

def load_resources(cpnet_vocab_path):
    '''
    Input: 
        cpnet_vocab_path: str
    Output: 
        concept2id, id2concept, relation2id, id2relation: global variables
    '''
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

def load_cpnet(cpnet_graph_path):
    '''
    Input: 
        cpnet_graph_path: str
    Output:
        cpnet, cpnet_simple: global variables
    '''
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

def concepts_to_graph_info_Part1(data):
    concept_ids = set(data['concept_ids'])
    extra_ids = set()
    for nodei in concept_ids:
        for nodej in concept_ids:
            if nodei != nodej and nodei in cpnet_simple.nodes and nodej in cpnet_simple.nodes:
                extra_ids |= set(cpnet_simple[nodei]) & set(cpnet_simple[nodej])
    extra_ids = extra_ids - concept_ids
    data['extra_ids'] = sorted(extra_ids)
    data['concept_ids'] = sorted(concept_ids)
    return data

def concepts_to_graph_info_Part2(data):
    global id2relation
    global max_cpt_nodes
    global id2concept
    cids = data['concept_ids'] + data['extra_ids']
    del data['extra_ids']
    n_rel = len(id2relation)
    n_node = len(cids)
    # prune for maximum concept nodes
    if n_node >= max_cpt_nodes:
        cids = cids[:max_cpt_nodes]
    elif n_node < max_cpt_nodes:
        paddings = [0] * (max_cpt_nodes-n_node)
        cids += paddings
    n_node = min(n_node, max_cpt_nodes)
    assert max_cpt_nodes == len(cids)
    data['concept_ids'] = cids
    #retrieve relations from ConceptNet
    head, tail, redge = [], [], []
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        head.append(s)
                        tail.append(t)
                        redge.append(e_attr['rel'])
    assert len(head) == len(tail) == len(redge)
    data['cpt_to_cpt'] = dict([('src', head),('dst', tail)])
    #data['cpt_to_cpt'] = dict([('src', head),('dst', tail),('rel', redge)])
    data['concepts'] = [id2concept[id] for id in data['concept_ids']]
    return data

def concepts_to_graph_info_Part3(data):
    ## Todo: concept中会有横线，需要处理
    concepts = data['concepts']
    facts = data['facts']
    head, tail = [], []
    for i, concept in enumerate(concepts):
        for j, fact in enumerate(facts):
            fact_tokens = set(fact.translate(translator).lower().split())
            filtered_fact_tokens = [word for word in fact_tokens if word not in nltk_stopwords]
            if concept in filtered_fact_tokens:
                head.append(i)
                tail.append(j)
    assert len(head) == len(tail)
    data['cpt_wiz_fct'] = dict([('src', head), ('dst', tail)])
    del data['concepts']
    return data

#{ 'question': ,'ans':, 'facts': ,'concept_ids': ,'cpt_to_cpt': ,'cpt_wiz_fct': }
def concepts_to_graph_info_Part4(data):
    filtered_fact_index = Counter(data['cpt_wiz_fct']['dst']).most_common(5)
    filtered_fact_index = [item[0] for item in filtered_fact_index]
    filtered_len = len(filtered_fact_index)
    if filtered_len < 5:
        idx = 0
        while len(filtered_fact_index) < 5 and idx <= 19:
            if idx not in filtered_fact_index:
                filtered_fact_index = filtered_fact_index + [idx]
            idx += 1
    #print(filtered_fact_index)
    assert len(filtered_fact_index)==5
    data['filtered_facts'] = [data['facts'][idx] for idx in filtered_fact_index]
    del data['facts']
    
    head = data['cpt_wiz_fct']['src']
    tail = data['cpt_wiz_fct']['dst']
    new_head = []
    new_tail = []
    for idx in range(len(tail)):
        if tail[idx] in filtered_fact_index:
            new_head.append(head[idx])
            new_tail.append(tail[idx])
    map = {k: v for v, k in enumerate(filtered_fact_index)}
    new_tail = [map[tail] for tail in new_tail]
    data['cpt_wiz_fct'] = dict([('src', new_head), ('dst', new_tail)])

    return data

#{ 'question': ,'ans':, 'filtered_facts': ,'concept_ids': ,'cpt_to_cpt': ,'cpt_wiz_fct': }
def concepts_to_graph_info_Part5(data):
    n_cpt = len(data['concept_ids'])
    n_fct = len(data['filtered_facts'])
    n_total_nodes = 1 + n_cpt + n_fct
    # add self loop
    adj_q2others = np.eye(n_total_nodes, dtype=int)
    adj_cpt2cpt = np.eye(n_total_nodes, dtype=int)
    adj_cpt2fct = np.eye(n_total_nodes, dtype=int)
    # mask unused nodes
    graph_mask = [0]*n_total_nodes

    cpt2cpt_rel = len(data['cpt_to_cpt']['src'])
    cpt2fct_rel = len(data['cpt_wiz_fct']['src'])
    # add link between question and other nodes <-- 0 represents the question node
    for idx in range(n_total_nodes):
        adj_q2others[0][idx], adj_q2others[idx][0] = 1, 1
    # 
    for idx in range(cpt2cpt_rel):
        src, dst = data['cpt_to_cpt']['src'][idx]+1, data['cpt_to_cpt']['dst'][idx]+1   # existion of question node 0
        adj_cpt2cpt[src][dst] = 1
    for idx in range(cpt2fct_rel):
        src, dst = data['cpt_wiz_fct']['src'][idx]+1, data['cpt_wiz_fct']['dst'][idx]+n_cpt+1
        adj_cpt2fct[src][dst], adj_cpt2fct[dst][src] = 1, 1
    adj_cpt2cpt = adj_cpt2cpt.tolist()
    adj_cpt2fct = adj_cpt2fct.tolist()
    adj_q2others = adj_q2others.tolist()

    concept_ids = data['concept_ids']
    cpts = concept_ids.copy()
    while -1 in cpts: cpts.remove(-1)
    num_actual = len(cpts)
    graph_mask[:num_actual+n_fct+1] = [1]*(num_actual+n_fct+1)

    data['adj_cpt2cpt'] = adj_cpt2cpt
    data['adj_cpt2fct'] = adj_cpt2fct
    data['adj_q2others'] = adj_q2others
    data['graph_mask'] = graph_mask

    #del data['cpt_to_cpt']
    #del data['cpt_wiz_fct']
    return data

def generate_adj_data_from_grounded_concepts(retrieval_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'Generating graph data for {retrieval_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    ## Load concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    print(f'Successfully loading ConceptNet Vocabulary!')
    ## Load cpnet and cpnet_simple
    if cpnet is None or cpnet_simple is None:  
        load_cpnet(cpnet_graph_path)
    print(f'Successfully loading ConceptNet Graph!')


    print(f'Cleaning data from {retrieval_path}...')
    qa_data = []
    with open(retrieval_path, 'r', encoding='utf-8') as fin_ground:
        lines_ground = fin_ground.readlines()
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            dic["question"] = dic.pop("sent")
            dic["facts"] = dic.pop("ctxs")
            dic["facts"] = [fact["title"]+ ". " + fact["text"] for fact in dic["facts"]]
            dic["concept_ids"] = set(concept2id[c] for c in dic['qc'])
            del dic["qc"]
            qa_data.append(dic)
    #{ 'question': , 'ans': , 'facts': ,'concept_ids':}

    
    print(f'Retrieving extra nodes...')
    with Pool(num_processes) as p:
        res1 = list(tqdm(p.imap(concepts_to_graph_info_Part1, qa_data), total=len(qa_data)))
    #{ 'question': ,'ans': ,'facts': ,'concept_ids': ,'extra_ids': }

    print(f'Retreving concept-level relations...')
    with Pool(num_processes) as p:
        res2 = list(tqdm(p.imap(concepts_to_graph_info_Part2, res1), total=len(res1)))
    #{ 'question': ,'ans': ,'facts': ,'concept_ids': ,'cpt_to_cpt': , 'concepts': }
    
    print(f'Processing multi-level relations...')
    with Pool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_graph_info_Part3, res2), total=len(res2)))
    #{ 'question': ,'ans':, 'facts': ,'concept_ids': ,'cpt_to_cpt': ,'cpt_wiz_fct': }

    print(f'Filtering facts...')
    with Pool(num_processes) as p:
        res4 = list(tqdm(p.imap(concepts_to_graph_info_Part4, res3), total=len(res3)))
    #{ 'question': ,'ans':, 'filtered_facts': ,'concept_ids': ,'cpt_to_cpt': ,'cpt_wiz_fct': }

    print(f'Creating adjacency matrix...')
    with Pool(num_processes) as p:
        res5 = list(tqdm(p.imap(concepts_to_graph_info_Part5, res4), total=len(res4)))
    #{ 'question': ,'ans':, 'filtered_facts': ,'concept_ids': ,'adj_cpt2cpt': ,'adj_cpt2fct': , 'graph_mask': }

    with open(output_path, 'w') as fout:
        for dic in res5:
            fout.write(json.dumps(dic) + '\n')

    print(f'Graph data successfully saved to {output_path}!')
    print()


if __name__ == "__main__":
    cpnet_vocab_path = '../data/cpnet/concept.txt'
    cpnet_graph_path = '../data/cpnet/conceptnet.en.pruned.graph'
   
    num_processes = 20
    file_pairs = {
        'train_csqa2': ('../data/csqa2/retrieval/train.retrieval.jsonl', '../data/csqa2/graph/train_graph_numnode20.jsonl'),
        'dev_csqa2': ('../data/csqa2/retrieval/dev.retrieval.jsonl', '../data/csqa2/graph/dev_graph_numnode20.jsonl'),
        'test_csqa2': ('../data/csqa2/retrieval/test.retrieval.jsonl', '../data/csqa2/graph/test_graph_numnode20.jsonl'),
        'train_creak': ('../data/creak/retrieval/train.retrieval.jsonl', '../data/creak/graph/train_graph_numnode20.jsonl'),
        'dev_creak': ('../data/creak/retrieval/dev.retrieval.jsonl', '../data/creak/graph/dev_graph_numnode20.jsonl'),
        'test_creak': ('../data/creak/retrieval/test.retrieval.jsonl', '../data/creak/graph/test_graph_numnode20.jsonl'),
        'contrast_creak': ('../data/creak/retrieval/contrast.retrieval.jsonl', '../data/creak/graph/contrast_graph_numnode20.jsonl')
    }
    for task in file_pairs:
        retrieval_path = file_pairs[task][0]
        output_path = file_pairs[task][1]
        generate_adj_data_from_grounded_concepts(retrieval_path, cpnet_graph_path, cpnet_vocab_path, 
                                                output_path, num_processes)

    

