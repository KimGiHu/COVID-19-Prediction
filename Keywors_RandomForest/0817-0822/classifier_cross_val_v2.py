import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.preprocessing import KBinsDiscretizer 
import numpy as np
import argparse
import os 
from sklearn.model_selection import KFold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
parser.add_argument('--count', type=int,
                    default=200, help=' ')
parser.add_argument('--paths', type = str,
                    default = './results/proposed_method1', help='Enter the stored path')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

names = ['ctfidf'] + ['rev_metric1', 'rev_metric2', 'rev_metric3']

createDirectory(args.paths)

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# train : test = 8 : 2
train_dates = pd.date_range(start='2019-08-01', end='2022-06-30', freq='D').tolist()
test_dates = pd.date_range(start='2022-07-01', end='2023-02-28', freq='D').tolist()

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

dates_split = {"train":train_dates, "test":test_dates}

datas = {}
Ns = range(5, 31, 5)
for name in names:
    datas[name] = {}
    # mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
    
    mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
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

index = {}
for name in names:
    max = -1
    index[name] = -1
    for N in Ns:
        train_dataset = datasets[name]['train'][N]
        test_dataset = datasets[name]['test'][N]
        
        # K-fold(K=5) Cross Validation model evaluation
        kfold = KFold(n_splits=k, shuffle=False)
        sum = 0
        for (train_ids, test_ids) in kfold.split(train_dataset[0]):
            train_X, test_X_cv = train_dataset[0][train_ids], train_dataset[0][test_ids]
            train_y, test_y_cv = train_dataset[1][train_ids], train_dataset[1][test_ids]

            # pre = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform', subsample=200000)
            # train_X = pre.fit_transform(train_X)
            # test_X_cv = pre.transform(test_X_cv)

            if name=='ctfidf' : 
                clf0 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

                clf0.fit(train_X, train_y)
                test_pred_cv = clf0.predict_proba(test_X_cv)[:,1]
            if name=='rev_metric1':
                clf1 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

                clf1.fit(train_X, train_y)
                test_pred_cv = clf1.predict_proba(test_X_cv)[:,1]
            if name=='rev_metric2':
                clf2 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

                clf2.fit(train_X, train_y)
                test_pred_cv = clf2.predict_proba(test_X_cv)[:,1]
            if name=='rev_metric3':
                clf3 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

                clf3.fit(train_X, train_y)
                test_pred_cv = clf3.predict_proba(test_X_cv)[:,1]
            # f1-score
            # tmp1 = test_y_cv.flatten()
            # tmp1 = np.where(tmp1>0.5, 0, 1)
            # tmp2 = test_pred_cv.flatten()
            # tmp2 = np.where(tmp2 > 0.5, 0, 1)
            # sum += f1_score(tmp1, tmp2)

            # AUC
            sum += roc_auc_score(test_y_cv, test_pred_cv)
        
        sum = sum/k    
        
        if max < sum :
            max = sum
            index[name] = N
            print(name)
            print(f'AUC for the {name} {index[name]} keywords  = {max}')

index_dot ={}
index_dot['ctfidf'] = index['ctfidf']
index_dot['rev_metric1'] = index['rev_metric1']
index_dot['rev_metric2'] = index['rev_metric2']
index_dot['rev_metric3'] = index['rev_metric3']

auc_results={}

for name in names:
    train_dataset = datasets[name]['train'][index[name]]
    test_dataset = datasets[name]['test'][index[name]]

    train_X, train_y = train_dataset
    test_X, test_y = test_dataset

    # pre = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform', subsample=200000)
    # train_X = pre.fit_transform(train_X)
    # test_X = pre.transform(test_X)

    if name=='ctfidf' :
        clf0 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

        clf0.fit(train_X, train_y)
        test_pred = clf0.predict_proba(test_X)[:,1]
    if name=='rev_metric1' :
        clf1 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

        clf1.fit(train_X, train_y)
        test_pred = clf1.predict_proba(test_X)[:,1]
    if name=='rev_metric2' :
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

        clf2.fit(train_X, train_y)
        test_pred = clf2.predict_proba(test_X)[:,1]
    if name=='rev_metric3' :
        clf3 = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

        clf3.fit(train_X, train_y)
        test_pred = clf3.predict_proba(test_X)[:,1]
    
    # AUC metric
    test_auc = roc_auc_score(test_y, test_pred)
    auc_results[name] = test_auc

for name in names:
    print("######################## %s AUC Results ########################"%name)
    print(name, end=' ')
    print("AUC : %.2f"%auc_results[name]+"\tindex : %.2f"%index[name], end=' ')
    print()
