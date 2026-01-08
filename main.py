#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 21:56:25 2026

@author: hounsousamuel
"""

import os, sys
import json
import time
import signal
import config
import joblib
import pandas as pd
from config import MODEL_NAME, DATA_NAME, WHITELIST
from pprint import pprint
from train import AntiPhishingIA
from features import clean_url, features_extractor, verify_ip_in_url, _get_domain
from diskcache import Cache

_dir_ = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache")
_dir_1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
MAX_CACHE_SIZE = 1024 * 1024  #
CACHE = Cache(_dir_, size_limit=MAX_CACHE_SIZE)
CACHE_TIMEOUT = 24 * 3600
cmds = []

def _help():
    print('Bienvenue !')
    print('Voici les commandes essentielles : ')
    print("    -Entrez: 'quit' ou 'q' pour quitter")
    print("    -Entrez: 'help' pour obtenir des infos")
    print("    -Entrez: 'change_seuil' suivie de la nouvelle valeur pour modifier le sueil(Ex: 'change_seuil 0.7')")
    print("    -Entrez: 'train' suivi du chemin vers vos données pour fitter un nouveau modèle")
    print("    -Entrez: 'printia' pour voir le modèle")
    print("    -Entrez: 'ls' ou 'history' pour voir l'historique des commandes")
    print("    -Entrez: 'clear'pour nettoyer le cache")
    print("    -Entrez: une url pour l'analyser(Ex: https:exemple.com ou exemple.com).\nSi il commence par change_seuil ou un des mots clées précédent veuillez le précéder de https ou http comme une vraie url.")
    print("Bonne expérience!")
    

def _validate(path:str):
    reasons = []
    is_valide = False
    if not os.path.exists(path):
        return is_valide, ['Chemin inexistant!']
    if not path.endswith((".pkl", ".joblib")):
        return is_valide, ['Chemin invalide, doit terminer par .pkl ou .joblib']
    try:
        data = joblib.load(path)
        if not isinstance(data, list):
            return is_valide, ['Les données doivent être une liste']
        for idx, line in enumerate(data):
            if not isinstance(line, dict):
                return is_valide, [f'Les données doivent contenir uniquement des dictionnaires, {idx} est {type(line).__name__}']
            keys = list(line.keys())
            for k in keys:
                if k not in ('url', 'label'):
                    return is_valide, ['Les clés des dictionnaires sont invalide']
                label = line['label']
                if isinstance(label, str):
                    if not label in ('safe', "phishing"):
                        return is_valide, ['Les valeurs de \'label\' doit être 0 ou 1']
                else:    
                    return is_valide, ['Les labels doivent être des string (safe ou phishing)']
        is_valide = True
        
    except Exception as e:
        reasons.append(f"{type(e).__name__}: {str(e)}")
        reasons.append('Erreur de lecture des données')
        return is_valide, reasons
    
    return is_valide, reasons

def predict(IA:AntiPhishingIA, url:str, seuil:float):
    dic = {"phishing": False, "prob": 0}
    if not 0 <= seuil <= 1:
        return dic
    value = CACHE.get(url, None)
    if value:
        return value
    domain = _get_domain(url)
    if domain in WHITELIST:
        dic['prob'] = 0
        CACHE.set(url, dic, expire=CACHE_TIMEOUT)
        return dic
    if verify_ip_in_url(url):
        dic['phishing'] = True
        dic['prob'] = 0.99
        CACHE.set(url, dic, expire=CACHE_TIMEOUT)
        return dic
    data = [features_extractor(url)]
    pred = IA.predict(data)
    proba = pred['predict_proba']
    try:
        first = proba[0]
    except:
        first = proba['0']
    proba_ph = first.get('phishing', 0)
    dic['phishing'] = ((1 - proba_ph) <= seuil)
    dic['prob'] = float(proba_ph)
    CACHE.set(url, dic, expire=CACHE_TIMEOUT)
    return dic
    
def signal_handler():
    def _sig(sig, frame):
        try:
            CACHE.close()
        except:
            pass
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGQUIT, _sig)
    
def fit_new(path):
    data = joblib.load(path)
    li = []
    try:
        for line in data:
            try:
                r = features_extractor(line['url'])
                r['label'] = line['label']
                li.append(r)
            except:
                pass
        print(len(li), 'echantillons extraits !')
        frame = pd.DataFrame(li)
        try:
            print(frame['label'].value_counts())
        except:
            pass
        t = time.time()
        dataset_name = f"dataset_{t}.pkl"
        model_name = f"model_{t}.pkl"
        save_path = os.path.join(_dir_1, dataset_name)
        print('Dataset file : ', save_path)
        joblib.dump(li, save_path)
        mod = AntiPhishingIA(dataset_file=dataset_name, optimize=True, model_file=model_name)
        mod.fit(li)
        return mod
    except Exception as e:
        print('Erreur : ', str(e))
        return
    

def main(mode:str = 'cli'):
    mode = mode.strip().lower()
    IA = AntiPhishingIA(model_file=MODEL_NAME, dataset_file=DATA_NAME)
    if mode == 'cli':
        while True:
            try:
                cmd = input(">>>(l'url ou la commande) : ").strip() or ""
                cmds.append({
                    "id": len(cmds),
                    "timestamp": time.time(),
                    "cmd": cmd
                })
                if not cmd:
                    continue
                if cmd.lower() == "help":
                    print(_help())
                elif cmd.lower() in("quit", "q"):
                    print('Bye')
                    break
                elif cmd.lower() == "printia":
                    print(IA)
                    print(IA.model)
                    continue
                elif cmd.lower() == "clear":
                    CACHE.clear()
                    continue
                elif cmd.lower() in ('ls', "history"):
                    print(json.dumps(cmds, indent=2, ensure_ascii=False))
                    continue
                elif cmd.startswith('change_seuil'):
                    try:
                        new = cmd.split()[-1] or -1
                        try:
                            new = float(new)
                        except:
                            print("La nouvelle valeur doir être un nombre !")
                            continue
                        
                        if not 0 <= new <= 1:
                            print('Nouveau seuil invalide, doit être entre 0 et 1')
                            continue
                        else:
                            config.SEUIL = new
                            print('Nouveau seuil : ', config.SEUIL)
                    except:
                        continue
                        
                elif cmd.startswith('train'):
                    path = cmd.split()[-1] or ""
                    if not os.path.exists(path):
                        print('Chemain inexistant !')
                        continue
                    else:
                        is_valide, reasons = _validate(path)
                        if not is_valide:
                            print('Struture invailde, raisons : ', '\n'.join(reasons))
                            continue
                        else:
                            mod = fit_new(path)
                            if mod:
                                IA = mod
                            
                else:
                    url = clean_url(cmd)
                    pred = predict(IA, url, config.SEUIL)
                    try:
                        print('Analyse pour : ', url)
                        print(json.dumps(pred, indent=2, ensure_ascii=False))
                    except Exception as e:
                        print('Erreur : ', str(e))
                        print(pprint(pred))
            except Exception as e:
                print('Erreur globale : ', str(e))
                import traceback
                traceback.print_exc()
                continue
    else:
        main('cli')
        raise NotImplementedError("Mode non implémenter")
    return 0

if __name__ == "__main__":
    signal_handler()
    main("cli")