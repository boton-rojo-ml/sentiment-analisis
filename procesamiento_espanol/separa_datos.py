#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import StratifiedShuffleSplit


train_index, test_index = StratifiedShuffleSplit(polaridad, 1,
                                                 test_size=0.2,
                                                 random_state=0)

datos_train, datos_test = datos[train_index], datos[test_index]
polaridad_train, polaridad_test = polaridad[train_index], polaridad[test_index]


train_index, valid_index = StratifiedShuffleSplit(polaridad_train, 1,
                                                  test_size=0.2,
                                                  random_state=0)

datos_train, datos_valid = datos_train[train_index], datos_valid[valid_index]
polaridad_train, polaridad_valid = polaridad[train_index], polaridad[valid_index]

