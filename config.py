#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 17:02:43 2026

@author: hounsousamuel
"""

MODEL_NAME = 'model.pkl'
DATA_NAME = 'dataset.pkl'
SEUIL = 0.6
WHITELIST = [
    'google.com', 'youtube.com', 'gmail.com', 'wikipedia.org', 
    'amazon.com', 'amazonaws.com', 'apple.com', 'icloud.com', 
    'microsoft.com', 'live.com', 'outlook.com', 'office.com', 
    'github.com', 'gitlab.com', 'bitbucket.org', 'stackoverflow.com', 
    'python.org', 'pypi.org', 'docs.python.org', 'mozilla.org', 
    'firefox.com', 'cloudflare.com', 'openai.com', 'huggingface.co', 
    'kaggle.com', 'linkedin.com', 'twitter.com', 'x.com', 'facebook.com', 
    'instagram.com', 'whatsapp.com', 'telegram.org', 'discord.com', 'slack.com',
    'zoom.us', 'dropbox.com', 'drive.google.com', 'paypal.com', 'stripe.com', 
    'visa.com', 'mastercard.com', 'binance.com', 'coinbase.com', 'kraken.com', 
    'yahoo.com', 'bing.com', 'duckduckgo.com', 'netflix.com', 'spotify.com', 
    'soundcloud.com', 'reddit.com', 'medium.com', 'dev.to', 'hashnode.com', 
    'notion.so', 'trello.com', 'asana.com', 'atlassian.com', 'jira.com', 
    'confluence.com', 'digitalocean.com', 'heroku.com', 'vercel.com', 
    'netlify.com', 'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com', 
    'ibm.com', 'oracle.com', 'sap.com', 'intel.com', 'amd.com', 'nvidia.com', 
    'coursera.org', 'edx.org', 'udemy.com', 'khanacademy.org', 'openstreetmap.org',
    'bbc.com', 'cnn.com', 'reuters.com', 'who.int', 'un.org', "localhost", "127.0.0.1"
]