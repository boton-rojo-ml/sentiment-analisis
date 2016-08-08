#!/usr/bin/env python
# -*- coding: utf-8 -*-

from string import punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Tienes que descargarte las stopwords primero via nltk.download()


espanol_stopwords = stopwords.words('spanish')

el_stemmer = SnowballStemmer('spanish')

non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str, range(10)))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens = word_tokenize(text)

    # stem
    try:
        stems = stem_tokens(tokens, el_stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    stop_words=espanol_stopwords)

