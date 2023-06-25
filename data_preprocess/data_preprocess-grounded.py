import json
import spacy
from tqdm import tqdm
from multiprocessing import Pool
from spacy.matcher import Matcher
import nltk


blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])

nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

## Set global variables
CPNET_VOCAB = '../data/cpnet/concept.txt'
PATTERN_PATH = '../data/cpnet/matcher_patterns.json'
nlp = None
matcher = None



def read_qa_file_cs2(qa_file: str):
    print(f'Reading from CSQA2 file...')
    with open(qa_file, 'r') as fin:
        lines = [line for line in fin]
    sents, answers = [], []
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        sents.append(j["question"])
        answers.append(j["answer"] if "answer" in j else None)
    assert len(sents) == len(answers)
    return sents, answers

def read_qa_file_creak(qa_file: str):
    print(f'Reading from CREAK file...')
    with open(qa_file, 'r') as fin:
        lines = [line for line in fin]
    sents, answers = [], []
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        sents.append(j["sentence"])
        answers.append(j["label"] if "label" in j else None)
    assert len(sents) == len(answers)
    return sents, answers


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs

def ground_mentioned_concepts(nlp, matcher, s, ans=None):

    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    if ans is not None:
        ans_matcher = Matcher(nlp.vocab)
        ans_words = nlp(ans)
        # print(ans_words)
        ans_matcher.add(ans, [[{'TEXT': token.text.lower()} for token in ans_words]])

        ans_match = ans_matcher(doc)
        ans_mentions = set()
        for _, ans_start, ans_end in ans_match:
            ans_mentions.add((ans_start, ans_end))

    for match_id, start, end in matches:
        if ans is not None:
            if (start, end) in ans_mentions:
                continue

        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)

        # print("span", span)
        # print("concept", original_concept)
        # print("Matched '" + span + "' to the rule '" + string_id)

        # why do you lemmatize a mention whose len == 1?

        if len(original_concept.split("_")) == 1:
            # tag = doc[start].tag_
            # if tag in ['VBN', 'VBG']:

            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        # print("span:")
        # print(span)
        # print("concept_sorted:")
        # print(concepts_sorted)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        # print("exact match:")
        # print(exact_match)
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)

    return mentioned_concepts

def hard_ground(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res

def ground_question(qa_pair):
    s, a = qa_pair
    question_concepts = ground_mentioned_concepts(nlp, matcher, s)
    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    question_concepts = sorted(list(question_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts}



def load_match(PATTERN_PATH):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    with open(PATTERN_PATH, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)
    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, [pattern])
    return nlp, matcher

def prune(data):
    # reload cpnet_vocab
    global CPNET_VOCAB
    with open(CPNET_VOCAB, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    print(f'Pruning concepts now...')
    prune_data = []
    for item in tqdm(data):
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)


        try:
            assert len(prune_qc) > 0
        except Exception as e:
            pass
        item["qc"] = prune_qc

        prune_data.append(item)
    return prune_data

def extract_concepts(sents, answers, num_processes):
    print(f'Extracting concepts now...')
    global nlp, matcher, PATTERN_PATH
    if nlp is None or matcher is None:
        nlp, matcher = load_match(PATTERN_PATH)
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_question, zip(sents, answers)), total=len(sents)))
    return res

def write_grounded(res, output_path):
    print(f'Writing grounded concepts...')
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')
    print(f'Grounded concepts saved to {output_path}')
    print()


def ground(qa_file, output_path, task):
    if task.endswith('csqa2'):
        sents, answers = read_qa_file_cs2(qa_file)
    if task.endswith('creak'):
        sents, answers = read_qa_file_creak(qa_file)
    #sents, answers = read_qa_file(qa_file)
    res = extract_concepts(sents, answers, num_processes=6)
    res = prune(res)
    write_grounded(res, output_path)


if __name__ == "__main__":
    file_pairs = {
        'train_csqa2': ('../data/csqa2/CSQA2_train.json', '../data/csqa2/grounded/train.grounded.jsonl'),
        'dev_csqa2': ('../data/csqa2/CSQA2_dev.json', '../data/csqa2/grounded/dev.grounded.jsonl'),
        'test_csqa2': ('../data/csqa2/CSQA2_test_no_answers.json', '../data/csqa2/grounded/test.grounded.jsonl'),
        'train_creak': ('../data/creak/train.json', '../data/creak/grounded/train.grounded.jsonl'),
        'dev_creak': ('../data/creak/dev.json', '../data/creak/grounded/dev.grounded.jsonl'),
        'test_creak': ('../data/creak/test_without_labels.json', '../data/creak/grounded/test.grounded.jsonl'),
        'contrast_creak': ('../data/creak/contrast_set.json', '../data/creak/grounded/contrast.grounded.jsonl')
    }
    for task in file_pairs:
        ground(file_pairs[task][0], file_pairs[task][1], task)
