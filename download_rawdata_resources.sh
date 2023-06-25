# Download ConceptNet
mkdir -p data/
mkdir -p data/cpnet/
echo "Downloading ConceptNet..."
wget -nc -P data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
# download ConceptNet entity embedding
echo "Downloading ConceptNet entity embedding..."
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../


mkdir -p data/wiki
cd data/wiki
# Download Wikipedia passages
echo "Downloading Wikipedia passages"
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
echo "Decompressing Wikipedia passages"
gzip -d psgs_w100.tsz.gz

# Generate passages embeddings
## Here we directly download passages embeddings pre-computed with Contriever or Contriever-msmarco
### wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
tar -xvf wikipedia_embeddings.tar
cd ../../


# download CREAK dataset
mkdir -p data/creak/

# create output folders for CREAK
mkdir -p data/creak/grounded/
mkdir -p data/creak/retrieval/
mkdir -p data/creak/graph/


# download CSQA2 dataset
mkdir -p data/csqa2/

# create output folders for CSQA2
mkdir -p data/csqa2/grounded/
mkdir -p data/csqa2/retrieval/
mkdir -p data/csqa2/graph/


