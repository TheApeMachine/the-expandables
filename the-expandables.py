#!/usr/bin/env python

import os
import sys
import numpy as np
import gensim
import spacy
import nltk
import torch
from nltk.corpus      import wordnet as wn
from torchtext.vocab  import load_word_vectors
from stemming.porter2 import stem
from itertools        import repeat
from operator         import itemgetter

def load_data():
    print "LOADING WORD2VEC MODEL..."
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/theapemachine/data/word2vec/text8-vector.bin', binary=True)

    print "LOADING SPACY MODEL..."
    spacy_model = spacy.load('en')

    print "LOADING GLOVE..."
    glove_dict, glove_arr, glove_size = load_word_vectors('.', 'glove.6B', 100)

    print "LOADING RESNIK..."
    resnik = nltk.corpus.wordnet_ic.ic('ic-bnc-resnik-add1.dat')

    print
    print "DATA LOADED!"
    print

    return w2v_model, spacy_model, glove_dict, glove_arr, glove_size, resnik

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:20]

def compute_score(word, match, resnik, score=0.0):
    try:
        word   = wn.synsets(word)[0]
        match  = wn.synsets(match)[0]
        score += sigmoid(word.path_similarity(match))
        score += sigmoid(word.lch_similarity(match))
        score += sigmoid(word.wup_similarity(match))
        score += sigmoid(word.jcn_similarity(match, resnik))
    except:
        pass

    return score

def expand(model, word, resnik):
    word        = word.lower()
    score_total = 0.0
    expansions  = []
    results     = []

    # TRY WORDNET FIRST
    try:
        for synset in model.synsets(word):
            for lemma in synset.lemma_names():
                score        = compute_score(word, lemma.lower(), resnik)
                score_total += score

                expansions.append({
                    'word':  lemma.lower(),
                    'score': score
                })
    except AttributeError:
        # TRY WORD2VEC
        try:
            words = model.most_similar(word)

            for w in words:
                score        = compute_score(word, w[0].lower(), resnik, w[1])
                score_total += score

                expansions.append({
                    'word':  w[0].lower(),
                    'score': score
                })
        except AttributeError:
            # TRY SPACY
            try:
                for w in most_similar(model.vocab[u''.join([word])]):
                    score        = compute_score(word, w.lower_, resnik)
                    score_total += score

                    expansions.append({
                        'word':  w.lower_,
                        'score': score
                    })
            except AttributeError:
                # TRY GLOVE
                for w in closest(model[0], model[1], get_word(word, model[0], model[1])):
                    score        = compute_score(word, w[0], resnik, w[1])
                    score_total += score

                    expansions.append({
                        'word':  w[0].lower(),
                        'score': score
                    })


    average = score_total / len(expansions)

    for e in expansions:
        if e['score'] > 0.0 and e['score'] >= average:
            results.append(e)

    return results

def get_word(word, glove_dict, glove_arr):
    return glove_arr[glove_dict[word]]

def closest(glove_dict, glove_arr, d, n=10):
    all_dists = [(w, torch.dist(d, get_word(w, glove_dict, glove_arr))) for w in glove_dict]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    w2v, spacy, glove_dict, glove_arr, glove_size, resnik = load_data()

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

    for c in categories:
        wn_words    = np.unique(expand(wn, c.lower(), resnik))
        w2v_words   = np.unique(expand(w2v, c.lower(), resnik))
        spacy_words = np.unique(expand(spacy, c.lower(), resnik))
        glove_words = np.unique(expand([glove_dict, glove_arr], c.lower(), resnik))
        results     = []

        for w in wn_words:
            results.append(w)

        for w in w2v_words:
            results.append(w)

        for w in spacy_words:
            results.append(w)

        for w in glove_words:
            results.append(w)

        print '--------------'
        print c
        print '--------------'

        for w in results:
            print "{} ({})".format(w['word'], w['score'])

        print


if __name__ == "__main__":
    main()
