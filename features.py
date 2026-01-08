#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 21:21:24 2026

@author: hounsousamuel
"""
import math, os, sys, ipaddress
from urllib.parse import urlparse, parse_qs
from tldextract import extract
from collections import Counter

SUSPICIOUS_WORDS = [
    'login', 'signin', 'verify-account', 'secure', 'account', 'update', 'banking',
    'payment', 'confirm', 'ebay', 'paypal', 'support', 'security',
    'validate', 'authenticate', 'password', 'credit', 'card', 'click'
]

SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.gq', '.cf', '.icu', '.top', '.xyz', '.club', '.download', '.win']

def _calculate_entropy(word:str):
    if not word:
        return 0
    
    counter = Counter(word)
    length = len(word)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())

    return entropy

def _get_domain(url:str):
    url = clean_url(url)
    extracted = extract(url)
    return extracted.domain + '.' + extracted.suffix

def clean_url(url:str):
    url = url.strip().strip(",;'\"").strip()
    if not url.startswith('http'):
        if url.startswith(":/"):
            url = "https" + url
        else:
            url = "https://" + url
    return url

def verify_ip_in_url(url):
    host = urlparse(url).hostname or ''
    if '.' in host :
        host_list = list(host.split('.'))
    else :
        host_list = [host]
    res = []
    for host in host_list:
        try:
            ipaddress.ip_address(host)
            res.append(True)
        except:
           pass
        if str(host).startswith('0x'):
           host1 = host.lower().removeprefix('0x')
           if all(c in '0123456789abcdef' for c in host1) and len(host1) <= 8:
                try:
                    num = int(host1,16)
                    res.append(True)
                except:
                   pass
        if host.isdigit():
            try:
                num = int(host)
                return 0 <= num <= 0xFFFFFFFF
            except:
                pass
        res.append(False)
    return any(r for r in res)    
    
def features_extractor(url:str):
    url = clean_url(url)
    domain = _get_domain(url)
    parse = urlparse(url)
    features = {
        "url": url,
        "is_ip": int(verify_ip_in_url(url)),
        'length': len(url),
        "too_length": int(len(url) > 78),
        "domain_length": len(domain),
        "domain_entropy": _calculate_entropy(domain),
        "has_at_sign": int("@" in url),
        'num_dash': url.count('-'),
        'dash_in_domain':int('-' in domain),
        'has_https': int(url.startswith('https://')),
        "has_punycode": int("xn--" in url),
        "num_query_params": len(parse_qs(parse.query)),
        'pos_slash': url.find('//'),
        'path_length': len(parse.path),
        'has_port': int(parse.port is not None),
        'digits_ratio_domain':sum(c.isdigit() for c in domain),
        "num_suspicious_words": sum(url.lower().count(k) for k in SUSPICIOUS_WORDS) ,
        "num_subdomain": len(extract(url).subdomain.split('.')) if extract(url).subdomain else 0,
        "suspicious_tld": int(extract(url).suffix in SUSPICIOUS_TLDS),
        'is_popular_tld': int(domain.endswith(('.com', '.org', '.fr', '.de', '.uk'))),
        }
    return features

if __name__ == "__main__":
    import pandas as pd
    from random import shuffle
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option("display.max_row", 111)
    pd.set_option('display.max_columns', 111)

    df = pd.read_csv('dataset4.csv')
    df1 = pd.read_csv('dataset3.csv')
    print(df1["status"].value_counts(), df['label'].value_counts())
    # input()
    print(df1.shape)
    print(df.shape,df.columns)
    urls1, labels = df['URL'], df['label']
    urls, labels1 = df1['url'], df1["status"].map({"legitimate":"safe",'phishing':'phishing'})
    f = lambda x: 'safe' if x>=1 else 'phishing'
    labels = [f(i) for i in labels]
    # print(json.dumps(dict(zip(urls1[:50],labels[:50])),indent=2))
    # print(json.dumps(dict(zip(urls[:50],labels1[:50])),indent=2))
    # input()
    li = []
    li_safe, li_phish = [], []
    for urls_, labels_ in [(urls1,labels),(urls,labels1)]:
        for i, url in enumerate(urls_):
            try:
                re = features_extractor(url)
                
                re['label'] = labels_[i]
                li.append(re)
            except Exception as e:
                print(f"Erreur pour l'URL {url}: {e}")
        print('Terminé')
        shuffle(li)
        print(pd.DataFrame(li),pd.DataFrame(li).columns,pd.DataFrame(li)['label'].value_counts())
    print(len(li))
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(li, f)
    url = "https://example.com"
    f = features_extractor(url)
    print(f)