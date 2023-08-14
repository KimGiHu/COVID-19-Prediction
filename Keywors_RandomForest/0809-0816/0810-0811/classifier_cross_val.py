import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,precision_recall_fscore_support
from sklearn.preprocessing import KBinsDiscretizer 
import numpy as np
import argparse
import os 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
parser.add_argument('--count', type=int,
                    default=200, help=' ')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

names = ['ctfidf'] + ['rev_metric1', 'rev_metric2', 'rev_metric3']

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# train : test = 8 : 2
train_dates = pd.date_range(start='2019-08-01', end='2022-05-31', freq='D').tolist()
test_dates = pd.date_range(start='2022-06-01', end='2023-02-28', freq='D').tolist()

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

dates_split = {"train":train_dates, "test":test_dates}

datas = {}
Ns = range(5, 201, 5)
for name in names:
    datas[name] = {}
    mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
    for split in ['train', 'test']:
        datas[name][split] = dict()
        for N in Ns:
            datas[name][split][N] = []
            for i in range(13, len(dates_split[split])):
                tmp = []
                for date in dates_split[split][i-13:i+1]:
                    tmp.append(mat.loc[date].values[:N])
                tmp = np.concatenate(tmp) # list tmp -> numpy tmp
                datas[name][split][N].append(tmp)
                   
            datas[name][split][N] = np.array(datas[name][split][N])

t = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/key_dates_%s.csv'%args.lang, index_col='date')['label']

labels = {}
for split in ['train', 'test']:
    labels[split] = []
    for date in dates_split[split][13:]:
        if date.replace('_','-') not in t.index:
            labels[split].append(0)
        else:
            tmp = date.replace('_','-')
            labels[split].append(t[tmp])

    labels[split] = np.array(labels[split])
print(labels['train'].shape, labels['test'].shape)

datasets = {}

for name in names:
    datasets[name] = {}
    for split in ['train', 'test']:
        datasets[name][split] = {}
        for N in Ns:
            datasets[name][split][N] = (datas[name][split][N], labels[split])
            
k = 5
max = -1
index = 0
for N in Ns:
    for name in names:
        train_dataset = datasets[name]['train'][N]
        test_dataset = datasets[name]['test'][N]
        
        # K-fold(K=5) Cross Validation model evaluation
        kfold = KFold(n_splits=k, shuffle=True, random_state=0)
        scores = 0
        for (train_ids, test_ids) in kfold.split(train_dataset[0]):
            train_X, test_X_cv = train_dataset[0][train_ids], train_dataset[0][test_ids]
            train_y, test_y_cv = train_dataset[1][train_ids], train_dataset[1][test_ids]

            pre = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform', subsample=200000)
            train_X = pre.fit_transform(train_X)
            test_X_cv = pre.transform(test_X_cv)

            clf = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

            clf.fit(train_X, train_y)
            test_pred_cv = clf.predict_proba(test_X_cv)[:,1]
            scores += roc_auc_score(test_y_cv, test_pred_cv)

    scores_mean = scores / k
    if max < scores_mean :
        max = scores_mean
        index = N
        print(f'AUC for the {index} keywords  = {max}')

auc_results={}


for name in names:
    train_dataset = datasets[name]['train'][index]
    test_dataset = datasets[name]['test'][index]

    train_X, train_y = train_dataset
    test_X, test_y = test_dataset

    pre = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform', subsample=200000)
    train_X = pre.fit_transform(train_X)
    test_X = pre.transform(test_X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)
    
    clf.fit(train_X, train_y)
    test_pred = clf.predict_proba(test_X)[:,1]
    
    test_auc = roc_auc_score(test_y, test_pred)
    auc_results[name] = test_auc
        
print("######################## AUC Results ########################")
# the AUC of test dataset
print("Test AUC")
for name in names:
    print(name, end=' ')
    print("%.2f"%auc_results[name], end=' ')
    print()
