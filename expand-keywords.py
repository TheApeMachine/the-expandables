import numpy as np
import gensim
import spacy
import nltk
from nltk.corpus import wordnet as wn
from stemming.porter2 import stem
from itertools import repeat

model       = gensim.models.KeyedVectors.load_word2vec_format('/home/theapemachine/data/word2vec/text8-vector.bin', binary=True)
nlp         = spacy.load('en')
w2v_words   = []
wn_words    = []
spacy_words = []
words       = []
categories  = [
    'advice',
    'hygiene',
    'equipment',
    'activities',
    'technology',
    'info',
    'administrative',
    'job',
    'education',
    'home',
    'health',
    'food'
]

def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:10]

def iterate(origin):
    try:
        for synset in wn.synsets(origin):
            for lemma in synset.lemma_names():
                if lemma.find('_') == -1:
                    w2v_words.append(lemma)

        for word in model.most_similar(origin):
            if word[0].find('_') == -1:
                wn_words.append(word[0])

        for w in most_similar(nlp.vocab[u''.join([origin])]):
            spacy_words.append(w.lower_)

    except:
        pass

def clean(w2v_words, wn_words, spacy_words, words):
    for w in w2v_words:
        if w in wn_words or w in spacy_words:
            words.append(stem(w))

    for w in wn_words:
        if w in w2v_words or w in spacy_words:
            words.append(stem(w))

    for w in spacy_words:
        if w in w2v_words or w in wn_words:
            words.append(stem(w))

    return [], [], []

for category in categories:
    print "------------------"
    print category
    print "------------------"
    iterate(category)

    w2v_words, wn_words, spacy_words = clean(
        w2v_words, wn_words, spacy_words, words
    )

    for i in repeat(None, 2):
        for w in np.unique(words):
            iterate(w)

        w2v_words, wn_words, spacy_words = clean(
            w2v_words, wn_words, spacy_words, words
        )

    for w in np.unique(words):
        print w

    print
    words = []
