#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Las clásicas
import numpy as np
import pandas as pd
import os
import cPickle as pickle

# Para el tratamiento del lenguaje
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Para el aprendizaje
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import sklearn.metrics as metricas

# Cargamos los datos
df = pd.read_csv("privado/data_train.csv")

X = df["comment"]
X = X.apply(lambda x: str(x).decode('unicode_escape').encode('utf-8', 'ignore').strip())

y = np.array(df['flag'])

# Problema binario 1 = Comentario negativo
y = np.where(y > 0, 0, 1)

# Separamos 20% para validación
indices = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
for tr_i, va_i in indices:
    X_train, X_valid = X.loc[tr_i], X.loc[va_i]
    y_train, y_valid = y[tr_i], y[va_i]
    break


espanol_stopwords = stopwords.words('spanish')
vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=True,
    stop_words=espanol_stopwords)

# Pasos del procesamiento
pipeline = Pipeline([
     ('vect', vectorizer),
     ('cls', SVC()),
    ])

# Aqui definimos el espacio de parámetros a explorar
parameters = {
    # 'vect__max_df': (0.9, 0.95, 1.0),
    # 'vect__min_df': (1, 0.05, 0.1),
    'vect__max_features': (20000, 40000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas o bigramas
    'cls__C': (0.001, 0.01, 0.1),
    'cls__class_weight': ('balanced', None)
    # 'cls__loss': ('hinge', 'squared_hinge'),
    # 'cls__max_iter': (500, 1000)
}

# # Aprende la mejor manera
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)
#
# # Muestra los parametros seleccionados
print grid_search.best_params_
#
# # Muestra los resultados con datos de validacion
y_est = grid_search.predict(X_valid)
print metricas.confusion_matrix(y_valid, y_est)
