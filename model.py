#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 21:21:37 2026

@author: hounsousamuel
"""

import os, sys
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 111)
pd.set_option("display.max_column", 111)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt.space import Integer, Categorical, Real
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings("ignore")


class Model:
    def __init__(self, random_state=42, lr=1e-3, verbose=0, cv=2):
        self.random_state = random_state
        self.lr = lr
        self.v = verbose
        self.cv = cv or StratifiedKFold(n_splits=3, random_state=self.random_state, shuffle=True)
    
    def _build_models(self):
        dic = {}
        xgb = XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            max_leaves=41,
            random_state=self.random_state,
            learning_rate=self.lr,
            objective='binary:logistic',
            tree_method="hist",
            n_jobs=-1,
            verbose=self.v
            )
        dic['xgb'] = ("xgb", xgb)
        
        hist = HistGradientBoostingClassifier(
            max_iter=1000,
            max_leaf_nodes=41,
            learning_rate=self.lr,
            loss='log_loss',
            n_iter_no_change=20,
            early_stopping=True,
            validation_fraction=0.1,
            scoring="f1_macro",
            max_depth=None,
            tol=1e-3,
            random_state=self.random_state,
            class_weight="balanced",
            verbose=self.v
            )
        dic['hist'] = ('hist', hist)
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=self.random_state,
            verbose=self.v,
            )
        dic['rf'] = ('rf', rf)
        
        log_reg = LogisticRegression(max_iter=3000, tol=1e-6, class_weight="balanced", C=10)
        pip_log = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ('log_reg', log_reg)
             ]
            )
        dic['log_reg'] = ("log_reg", pip_log)
        return dic
    
    def _get_dict_opt(self, name:str):
        name = str(name).lower()
        PARAMS = {
            "rf": {
                "n_estimators": Integer(200, 700),
                "max_depth": Categorical([4, 6, 8, 10, 12, None]),
                "max_features": Categorical(['sqrt', "log2"])
                },
            "xgb": {
                "n_estimators": Integer(1000, 3000),
                "max_depth": Categorical([4, 6, 8, 10, 12, 16]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform')
                },
            "hist": {
                "max_iter": Integer(1000, 3000),
                "max_depth": Categorical([4, 6, 8, 10, 12, None]),
                "learning_rate": Real(1e-4, 1e-1, prior='log-uniform'),
                "tol": Real(1e-6, 1e-1, prior='log-uniform')
                }
            }
        return PARAMS.get(name, {})
    
    def _optimize(self, dict_of_models:dict, X, y, max_iter=10):
        exclude = ['log_reg']
        models = [dict_of_models[k] for k in dict_of_models.keys() if k not in exclude]
        best_models = {}
        
        def _opt():
            for name, model in models:
                s = time.time()
                print('Optimisation de ', str(name).upper())
                search_spaces = self._get_dict_opt(name)
                bayes = BayesSearchCV(
                    model, 
                    search_spaces,
                    scoring="f1_macro",
                    n_iter=max_iter,
                    n_jobs=-1,
                    cv=self.cv,
                    return_train_score=True
                    )
                bayes.fit(X, y)
                best = bayes.best_estimator_
                best_models[name] = (name, best)
                print("Fit de ", str(name).upper(), " fini : \n")
                print('Meilleur score cv : ', bayes.best_score_)
                print('Meillsur params : ', dict(bayes.best_params_))
                print('Infos d\'optimisations : \n', pd.DataFrame(bayes.cv_results_))
                print(f"Fit fini en : {time.time() - s} secondes !")
                print()
        _opt()
                
        for name in exclude:
            if name in dict_of_models:
                _, model = dict_of_models[name]
                best_models[name] = (name, model)
                
        return best_models
    
    def run(self, opt=False, X=None, y=None, max_iter=10):
        models = self._build_models()
        log_reg = LogisticRegression(max_iter=3000, tol=1e-6, class_weight="balanced", C=10)
        meta = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ('log_reg', log_reg)
             ]
            )
        if not opt:
            stack = StackingClassifier(
                list(models.values()),
                final_estimator=meta,
                passthrough=True,
                n_jobs=-1,
                cv=self.cv,
                stack_method="predict_proba"
                )
        else:
            if not X is None and not y is None:
                X = np.asarray(X)
                y = np.asarray(y)
                best_models = self._optimize(models, X, y, max_iter)
                stack = StackingClassifier(
                    list(best_models.values()),
                    final_estimator=meta,
                    passthrough=True,
                    n_jobs=-1,
                    cv=self.cv,
                    stack_method="predict_proba"
                )
        return stack
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    np.random.seed(0)
    X, y = make_classification(10000, 30)
    model = Model()
    stack = model.run(opt=True, X=X, y=y)
    print(stack)