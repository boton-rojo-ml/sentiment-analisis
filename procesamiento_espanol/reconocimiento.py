#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metricas
import preproc


pipeline = Pipeline([
    ('vect', preproc.vectorizer),
    ('cls', LinearSVC()),
])


# Aqui definimos el espacio de parámetros a explorar
parameters = {
    'vect__max_df': (0.9, 0.95, 1.0),
    'vect__min_df': (10, 20, 50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas o bigramas
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}


grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='roc_auc')

grid_search.fit(datos_train, polaridad_train)
polaridad_estimada = grid_search.predict(datos_valid)
metricas.confusion_matrix(polaridad_valid, polaridad_estimada)
