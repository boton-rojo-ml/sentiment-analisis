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

# Creamos el tokenizador en español
espanol_stopwords = stopwords.words('spanish')

vectorizer = CountVectorizer(
    analyzer='word',
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


# Separamos 20% para validación
y = np.where(y > 0, 0, 1)

indices = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=0)
for tr_i, va_i in indices:
    X_train, X_valid = X[tr_i], X[va_i]
    y_train, y_valid = y[tr_i], y[va_i]
    break


cls = LinearSVC()

cls.fit(X_train, y_train)
y_est = cls.predict(X_valid)

print metricas.confusion_matrix(y_valid, y_est)

# Pasos del procesamiento
pipeline = Pipeline([
     ('vect', preproc.vectorizer),
     ('cls', LinearSVC()),
    ])

# Aqui definimos el espacio de parámetros a explorar
parameters = {
    'vect__max_df': (0.9, 0.95, 1.0),
    'vect__min_df': (1, 0.05, 0.1),
    'vect__max_features': (10000, 20000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas o bigramas
    'cls__C': (0.01, 0.1, 1, 10),
    'cls__class_weight': ('Balanced', None)
    #'cls__loss': ('hinge', 'squared_hinge'),
    #'cls__max_iter': (500, 1000)
}

# # Aprende la mejor manera
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='roc_auc')
grid_search.fit(datos_train, polaridad_train)
#
# # Muestra los parametros seleccionados
print grid_search.best_params_
#
# # Muestra los resultados con datos de validacion
polaridad_estimada = grid_search.predict(datos_valid)
metricas.confusion_matrix(polaridad_valid, polaridad_estimada)
