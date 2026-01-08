#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 03:24:40 2026

@author: hounsousamuel
"""
import os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import HistGradientBoostingRegressor
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score
from model import Model

_dir_ = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
os.makedirs(_dir_, exist_ok=True)

class AntiPhishingIA:
    def __init__(
            self, optimize:bool = False, 
            dataset_file:str = "dataset.pkl", 
            model_file:str = "model.pkl", 
            random_state:int = 42, 
            features_name:list = [],
            verbose:int = 0,
            lr:float = 1e-3
        ):
        self.optimize = optimize
        self.rs = random_state
        self.v = verbose
        self.lr = lr
        self.feature_names = features_name
        self.dataset_file = os.path.join(_dir_, dataset_file or "dataset.pkl")
        self.model_file = os.path.join(_dir_, model_file or "model.pkl")
        self.data = pd.DataFrame()
        self.model = None
        self.Model = Model(random_state=self.rs, lr=self.lr, verbose=self.v)
        self.load_dataset()
       
        if not self.data.empty:
            self.feature_names = features_name or [p for p in list(self.data.columns) if p not in ('url', "label")]
            self._validate_data(self.data)
        else:
            self.feature_names = features_name
        self.le = LabelEncoder()
        self.imputer = IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                max_iter=1000,
                max_depth=6,
                n_iter_no_change=20,
                random_state=self.rs,
                ),
            max_iter=20,
            tol=1e-5
            )
        self.load_model()
        self.SMOTE = SMOTE(k_neighbors=20, random_state=self.rs)
        
    
    def load_dataset(self):
        if os.path.exists(self.dataset_file):
            data = None
            try:
                data = joblib.load(self.dataset_file)
                print('Dataset chargé avec succès !')
            except Exception as e:
                print('Erreur lors du chargement du dataset dans : ', self.dataset_file, " \nErreur : ", str(e))
            if data:
                self._validate_data(data)
                self.data = pd.DataFrame(data)
            else:
                self.data = pd.DataFrame()
        else:
            print('Dataset inexistant, démarrage avec un dataset vide !')
            self.data = pd.DataFrame()
    
    def save_dataset(self):
        try:
            to_save = pd.DataFrame(self.data).to_dict(orient='records')
            joblib.dump(to_save, self.dataset_file)
            print("Savegarde du dataset réussi !")
        except Exception as e:
            print('Erreur lors de la sauvegarde du dataset dans : ', self.dataset_file, " \nErreur : ", str(e))
    
    def load_model(self):
        if os.path.exists(self.model_file):
            data = None
            try:
                data = joblib.load(self.model_file)
                print('Modèle bien chargé, demarrage de la restauration !')
            except Exception as e:
                print('Erreur lors du chargement du dataset dans : ', self.model_file, " \nErreur : ", str(e))
            if data:
                self.model = data['model']
                self.le = data['le']
                self.feature_names = data['features_name']
                self.imputer = data['imputer']
            print('Modèle restauré correctement !')
            
        else:
            print('Model inexistant, démarrage sans modèle !')
            self.model = None
        
    
    def save_model(self):
        try:
            to_save = {
                "model": self.model,
                "le": self.le,
                "imputer": self.imputer,
                "features_name": self.feature_names
                }
            joblib.dump(to_save, self.model_file)
            print("Savegarde du model réussi !")
        except Exception as e:
            print('Erreur lors de la sauvegarde du modèle dans : ', self.model_file, " \nErreur : ", str(e))
    
    def prepa_data(self, data:list, mode:str = "fit"):
        if isinstance(data, dict):
            data = [data]
        if any(not isinstance(x, dict) for x in data):
            for x in data:
                if not isinstance(x, dict):
                    print(x)
            raise ValueError('Les données doivent être une liste de dictionnaire !')
        
        data = self._validate_data(data)
        data = pd.DataFrame(data)
        if data.empty:
            raise ValueError('Data vide')
        mode = mode.strip().lower()
        if 'label' not in data.columns and mode == 'fit':
            raise ValueError("La colonne 'label' est requise en mode 'fit'")
        if 'label' in data.columns:
            data['label'] = data['label'].apply(lambda x: x if isinstance(x, (str, int)) else str(x[0]))
            
        columns = list(data.columns)
        missing = [p for p in self.feature_names if p not in columns]
        filtered = pd.DataFrame()
        if missing:
            for miss in missing:
                data[miss] = 0
        for col in self.feature_names:
            filtered[col] = data[col]
        if "label" in data:
            filtered['label'] = data['label']
        if "url" in data:
            filtered['url'] = data["url"]
            
        if mode == "fit":
            if self.data.empty:
                self.data = filtered
            if not self.data.equals(filtered):
                self.data = pd.concat((self.data, filtered), axis=0)
            else:
                self.data = filtered
            print("Avant, taille de self.data : ", len(self.data))
            self.data = self.data.drop_duplicates(subset=['url'], inplace=False, ignore_index=True)
            print("Après, taille de self.data : ", len(self.data))
            # input()
            self.save_dataset()
            to_drop = ['url']
            for drop in to_drop:
                self.data = self.data.drop(drop, axis=1, inplace=False)
                
            X = self.data.drop('label', axis=1, inplace=False)
            y = self.data["label"]
            y = self.le.fit_transform(y)
            if self.data.isna().sum().sum() != 0:
                print('NaN dans les données : ', self.data.isna().sum().sum())
                X = self.imputer.fit_transform(X.to_numpy())
            X, y = self.SMOTE.fit_resample(X, y)
            return X, y
            
        else:
            if 'label' in data.columns:
                y = data['label']
                X = data.drop("label", axis=1, inplace=False)
            else:
                y = None
                X = data
          
            to_drop = ['url']
            for drop in to_drop:
                X.drop(drop, axis=1, inplace=True)
            
            return X, y
        
    def _validate_data(self, data):
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, type(pd.DataFrame())):
            data = data.to_dict(orient='records')
        if any(not isinstance(x, dict) for x in data):
            for x in data:
                if not isinstance(x, dict):
                    print(x)
            raise ValueError('Les données doivent être une liste de dictionnaire !')
            
        data = pd.DataFrame(data)
        if data.empty:
            return data
        
        required = ['url']
        for r in required:
            if not r in data.columns:
                raise ValueError(f'Column manquant, {r}')
        # print(data.head(2))
        return data
    
    def fit(self, data:list, max_iter=10):
        if isinstance(data, dict):
            data = [data]
        if any(not isinstance(x, dict) for x in data):
            raise ValueError('Les données doivent être une liste de dictionnaire !')
    
        X, y = self.prepa_data(data, mode="fit")
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=self.rs)
        self.model = self.Model.run(opt=self.optimize, X=X_train, y=y_train, max_iter=max_iter)
        self.model.fit(X, y)
        self.evaluate(X_test, y_test)
        self.save_model()
        return self
    
    def evaluate(self, X, y):
        predict =  self.model.predict(X)
        cr = classification_report(y, predict)
        cm = confusion_matrix(y, predict)
        score = self.model.score(X, y)
        hml, js = hamming_loss(y, predict), jaccard_score(y, predict)
        print('Score : ', score)
        print('hamming_loss(plus c\'est petit mieux c\'est) : ', hml)
        print('Jaccard score(plus c\'est grand mieux c\'est) : ', js)
        print('Confusion matrix : \n', cm)
        print('Classification report : \n', cr)
    
    def predict(self, data:list, y_is_str:bool = False):
        if not self.model:
            raise ValueError('Veuillez d\abord entrainé le model avec la méthode fit !')
        if isinstance(data, dict):
            data = [data]
        if not all(isinstance(x, dict) for x in data):
            raise ValueError('Les données doivent être une liste de dictionnaire !')
        self._validate_data(data)
        X, y = self.prepa_data(data, mode="predict")
        if y_is_str:
            y = self.le.inverse_transform(y) if not y is None else None
        y_pred = np.asarray(self.model.predict(X)).astype(int)
        y_pred_proba = np.asarray(self.model.predict_proba(X)).astype(float)
        # print(y_pred, y_pred_proba)
        to_return = {
            "predict": {i:pred for i, pred in enumerate(self.le.inverse_transform(y_pred))},
            "predict_proba": {i:dict(zip(self.le.classes_, [float(x) for x in line])) for i, line in enumerate(y_pred_proba)},
            "true_label": {i:label for i, label in enumerate(y)} if not y is None else {}
            }
        return to_return

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import json
    def fit():
        file = "data/dataset.pkl"
        mod = AntiPhishingIA(optimize=True)
        data = joblib.load(file)
        frame = pd.DataFrame(data)
        print()
        print(frame.describe())
        print()
        print(frame['label'].value_counts())
        print()
        train, test = tts(frame, test_size=0.2)
        test_list = test.to_dict(orient="records")
        train_list = train.to_dict(orient='records')
        print("NaN: ", train.isna().sum().sum())
        input()
        joblib.dump(test_list, "test.pkl")
        
        # mod.fit(max_iter=10, data=train_list)
        for t in test_list:
            pred = mod.predict(t)
            try:
                print(json.dumps(pred, indent=2, ensure_ascii=False))
            except:
                print(pred)
            
        
    fit()