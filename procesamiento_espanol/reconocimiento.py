#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Las clásicas
import numpy as np
import pandas as pd
import os
import cPickle as pickle

# Para el tratamiento del lenguaje
from string import punctuation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Para el aprendizaje
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import sklearn.metrics as metricas

# Creamos el tokenizador en español
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
    text = ' '.join([c for c in text if c not in non_words])
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
    #  tokenizer=tokenize,
    lowercase=True,
    stop_words=espanol_stopwords)


# Cargamos los datos
arch_prepro = "privado/prepro_ascii_data.pck"
if not os.path.isfile(arch_prepro):
    df = pd.read_csv("privado/data_train.csv")

    X = list(df["comment"])
    X = map(lambda x: str(x).decode('unicode_escape').encode('utf-8', 'ignore').strip(), X)
    X = vectorizer.fit_transform(X)

    y = np.array(df['flag'])

    temp_dir = {'X': X, 'y': y, 'vectorizer': vectorizer}
    pickle.dump(temp_dir, open(arch_prepro, "wb"))

else:
    temp_dir = pickle.load(open(arch_prepro, "rb"))
    y = temp_dir['y']
    X = temp_dir['X']
    vectorizer = temp_dir['vectorizer']

print X.shape

print vectorizer.vocabulary_.keys()

# Separamos 20% para validación
indices = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
for tr_i, va_i in indices:
    X_train, X_valid = X[tr_i], X[va_i]
    y_train, y_valid = y[tr_i], y[va_i]
    break



# Pasos del procesamiento
# pipeline = Pipeline([
#     ('vect', preproc.vectorizer),
#     ('cls', LinearSVC()),
# ])
#
# # Aqui definimos el espacio de parámetros a explorar
# parameters = {
# #    'vect__max_df': (0.9, 0.95, 1.0),
# #    'vect__min_df': (10, 20, 50),
# #    'vect__max_features': (1000, 5000),
# #    'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas o bigramas
#     'cls__C': (0.2, 0.5, 0.7),
# #    'cls__loss': ('hinge', 'squared_hinge'),
# #    'cls__max_iter': (500, 1000)
# }
#
# # Aprende la mejor manera
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='roc_auc')
# grid_search.fit(datos_train, polaridad_train)
#
# # Muestra los parametros seleccionados
# print grid_search.best_params_
#
# # Muestra los resultados con datos de validacion
# polaridad_estimada = grid_search.predict(datos_valid)
# metricas.confusion_matrix(polaridad_valid, polaridad_estimada)
