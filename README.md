# Decker: Double Check with Heterogeneous Knowledge for Commonsense Fact Verification

**DECKER**, consists of three major modules: (i) Knowledge Retrieval Module which retrieves heterogeneous knowledge based on the input question; (ii) Double Check Module which merges information from structured and unstructured knowledge and makes a double check between them; (iii) Knowledge Fusion Module which combines heterogeneous knowledge together to obtain a final representation.

![](pics/overview.png)


## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the required knowledge bases:

```sh
sh download_rawdata_resource.sh
```

Download the datasets from the following repository and put them under `data/creak` and `data/csqa2`:

```
https://github.com/yasumasaonoe/creak/tree/main/data/creak
https://github.com/allenai/csqa2/tree/master/dataset
```

## Implementations

### Data preprocessing

Data preprocessing consists of three stages: (i) Ground concepts; (ii) Retrieve facts; (iii) Process and load graph information.

```
cd data_preprocess
sh data_preprocess.sh
```

### Training

```sh
sh confact_train.sh
```
