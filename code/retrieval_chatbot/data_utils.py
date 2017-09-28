import string
import numpy as np
import csv
import pickle
from nltk import *
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from nltk.stem.porter import PorterStemmer
from random import sample




vocab_size = 8000
msg_len = 30

def tokenize(prompt):
    prompt = prompt.lower()
    translator = prompt.maketrans('', '', string.punctuation)
    promptNoPuncs = prompt.translate(translator)
    tokens = word_tokenize(promptNoPuncs)
    tokens = [token for token in tokens if not token in corpus.stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = stemTokens(tokens, stemmer)
    return tokens


def stemTokens(tokens, stemmer):
    stemmedTokens = []
    for token in tokens:
        stemmedTokens.append(stemmer.stem(token))
    return stemmedTokens



def create_vocab(prompts, responses):
    vocab = ['PAD', 'UNK']
    vocab_ctr = Counter()
    for prompt, response in zip(prompts, responses):
        tokens_prompt = tokenize(prompt)
        tokens_response = tokenize(prompt)
        for token in tokens_prompt:
            vocab_ctr[token] += 1
        for token in tokens_response:
            vocab_ctr[token] += 1

    common_vocab = vocab_ctr.most_common(vocab_size-2)
    for tuple in common_vocab:
        vocab.append(tuple[0])
    print ("Vocab size {}".format(len(vocab)))
    return vocab


def vectorize(msg, vocab):
    vector = np.zeros(msg_len)
    msg_words = tokenize(msg)

    for i in range(min(len(msg_words), msg_len)):
        if msg_words[i] in vocab:
            vector[i] = vocab.index(msg_words[i])
        else:
            vector[i] = vocab.index("UNK")
    return list(vector)


def gen_training_data():
    prompt_vectors = []
    response_vectors = []
    labels = []

    prompts = []
    responses = []
    with open('train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row['Context'])
            responses.append(row['Utterance'])
            labels.append(row['Label'])
    vocab = create_vocab(prompts, responses)
    print ("Generating vectors")
    for prompt, response in zip(prompts, responses):
        prompt_vectors.append(vectorize(prompt, vocab))
        response_vectors.append(vectorize(response, vocab))

    np.save('vocab.npy', np.array(vocab))
    np.save('prompts.npy', np.array(prompt_vectors))
    np.save('responses.npy', np.array(response_vectors))
    np.save('labels.npy', np.array(labels))

    print ("Done generating training data")


if __name__ == "__main__":
    gen_training_data()



