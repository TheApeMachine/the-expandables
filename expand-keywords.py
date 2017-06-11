#!/usr/bin/env python

import sys
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
w2v_stems   = []
wn_stems    = []
spacy_stems = []
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
                    w2v_stems.append(stem(lemma))

        for word in model.most_similar(origin):
            if word[0].find('_') == -1:
                wn_words.append(word[0])
                wn_stems.append(stem(word[0]))

        for w in most_similar(nlp.vocab[u''.join([origin])]):
            spacy_words.append(w.lower_)
            spacy_stems.append(stem(w.lower_))

    except:
        pass
        
def clean(w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems, words):
    for w in w2v_words:
        if stem(w) in wn_stems or stem(w) in spacy_stems:
            words.append(w)

    for w in wn_words:
        if stem(w) in w2v_stems or stem(w) in spacy_stems:
            words.append(w)

    for w in spacy_words:
        if stem(w) in w2v_stems or stem(w) in wn_stems:
            words.append(w)

    return [], [], [], [], [], []

if len(sys.argv) > 1:
    iterate(sys.argv[1])

    w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems = clean(
        w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems, words
    )

    for i in repeat(None, 2):
        for w in np.unique(words):
            iterate(w)

        w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems = clean(
            w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems, words
        )

    for w in np.unique(words):
        print w

    words = []
else:
    for category in categories:
        print "------------------"
        print category
        print "------------------"
        iterate(category)

        w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems = clean(
            w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems, words
        )

        for i in repeat(None, 2):
            for w in np.unique(words):
                iterate(w)

            w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems = clean(
                w2v_words, wn_words, spacy_words, w2v_stems, wn_stems, spacy_stems, words
            )

        for w in np.unique(words):
            print w

        print
        words = []
