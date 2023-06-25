
## Part One: Ground concepts
'''
Input file: ../data/dataset_name/train|dev|test
Output file: ../data/dataset_name/grounded/train|dev|test.grounded.jsonl
            {"sent": , "ans": , "qc":}
'''
python data_preprocess-grounded.py



## Part Two: Retrieve facts
'''
Input file: ../data/dataset_name/train|dev|test.grounded.jsonl
Output file: ../data/dataset_name/retrieval/train|dev|test.retrieval.jsonl
            {"sent": ,"ans": ,"qc": ,ctxs: }
'''
python data_preprocess-retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ../data/wiki/psgs_w100.tsv \
    --passages_embeddings "../data/wiki/wikipedia_embeddings/*" \
    --data_path ../data/ \
    --dataset creak \
    --output_dir ../data/creak/retrieval \

python data_preprocess-retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ../data/wiki/psgs_w100.tsv \
    --passages_embeddings "../data/wiki/wikipedia_embeddings/*" \
    --data_path ../data/ \
    --dataset csqa2 \
    --output_dir ../data/csqa2/retrieval \




## Part Three: Process and Load graph information
'''
Input file: ../data/dataset_name/retrieval/train|dev|test.retrieval.jsonl
Output file: ../data/dataset_name/graph/train|dev|test_graph_numnode50.jsonl
'''
python data_preprocess-graph.py
