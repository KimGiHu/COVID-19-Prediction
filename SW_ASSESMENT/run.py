import pandas as pd
import numpy as np
import pickle
import argparse
import os
import datetime
import spacy
import re
from models import LSTM
import torch
import json
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.json')
parser.add_argument('--output', type=str, default='output.json')
args = parser.parse_args()

nlps = {
        'korean': 'ko_core_news_lg',
        'english': 'en_core_web_lg',
        'japanese': 'ja_core_news_lg',
        'french': 'fr_core_news_lg',
        'italiano': 'it_core_news_lg',
        'spanish': 'es_core_news_lg',
        'russian': 'ru_core_news_lg',
        'portuguese': 'pt_core_news_lg',
}

# languages
languages = ['korean', 'english', 'japanese', 'french', 'italiano', 'spanish', 'russian', 'portuguese']

# language code
codes = {
    'korean': 'ko',
    'english': 'en',
    'japanese': 'ja',
    'french': 'fr',
    'italiano': 'it',
    'spanish': 'es',
    'russian': 'ru',
    'portuguese': 'pt',
}

lang_filter = {
    'korean' : re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]'),
    'english' : re.compile('[^a-zA-Z]'),
    # 'japanese' : re.compile('[^ぁ-んァ-ン一-龯]')
    'japanese' : re.compile('[^一-龯]'),
    'french' : re.compile(r'[^a-zA-Zàâäçéèêëîïôöùûüÿ]'),
    'italiano' : re.compile(r'[^a-zA-Zàèéìòóù]'),
    'spanish' : re.compile(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]'),
    'russian' : re.compile(r'[^а-яА-ЯёЁ]'),
    'portuguese' : re.compile(r'[^a-zA-ZáâãàéêíóôõúçÁÂÃÀÉÊÍÓÔÕÚÇ]')
}

Ns = {
    'United States': 15,
    'Republic of Korea': 110,
    'France': 25,
    'Japan': 110,
    'Russia': 5,
    'Portugal':30,
    'Italy':150
}


def tokenizer(text, language, keywords, nlp):
    if language in languages:
        doc = nlp(text)
        words = [token.lemma_ for token in doc if token.lemma_ in keywords]
        return words
    else:
        return text.split()


def run_task(data, lang, keywords, N):
    nlp = spacy.load(nlps[lang])
    v = np.zeros(N)
    for title, full_text in data:
        text = title + ' ' + full_text
        text = lang_filter[lang].sub(' ', text)
        words = tokenizer(text, lang, keywords, nlp)
        for word in words:
            v[keywords.index(word)] += 1
    return v


def preprocess(datas, lang, keywords, N):
    # Pool
    pool = mp.Pool(processes=mp.cpu_count()+1)

    # Run
    results = [pool.apply_async(run_task, args=(data, lang, keywords, N)) for data in datas]
    vectors = [p.get() for p in results]
    return np.concatenate(vectors)



def run(input_dict):
    news = input_dict["News"]
    lang = input_dict["Language"]
    country = input_dict["Country"]
    confirmed = input_dict["Confirmed"]

    N = Ns[country]

    keywords = pd.read_csv(f'./keywords/{country}_keyword.csv', index_col=0).index[:N].tolist()

    # load model
    with open(f'./keywords/{country}_model/model.pkl', 'rb') as f:
        clf = pickle.load(f)

    datas = preprocess(news, lang, keywords, N)
    
    pred = clf.predict_proba(datas[np.newaxis,:])[:,1]

    # load model
    lstm = torch.load(f'./lstm.pt').cpu()
    lstm.eval()
    ncov = torch.tensor(confirmed).float().unsqueeze(0)
    pred2 = lstm(ncov)
    pred2 = torch.softmax(pred2, dim=-1).detach().numpy()[:,1]
    pred = np.where(( 0.9 * pred + 0.1 * pred2 ) > 0.5, 1, 0)
    output = {
        "RiskIndex": int(pred[0])
    }
    return output


if __name__ == '__main__':
    # read input
    with open(args.input, "r") as f:
        input_dict = json.load(f)

    output = run(input_dict)

    # write output
    with open(args.output, "w") as f:
        json.dump(output, f)
    
