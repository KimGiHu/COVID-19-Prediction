import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer 
import numpy as np
import argparse
import os 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
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

# set the train/validation/test dates
train_dates = pd.date_range(start='2019-08-01', end='2022-08-31', freq='D').tolist()
val_dates = pd.date_range(start='2022-09-01', end='2022-11-30', freq='D').tolist() 
test_dates = pd.date_range(start='2022-12-01', end='2023-02-28', freq='D').tolist() 

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(val_dates)):
    val_dates[i] = val_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

dates_split = {"train":train_dates, "val":val_dates, "test":test_dates}

datas = {}
Ns = range(5, 201, 5)
for name in names:
    datas[name] = {}
    mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_constant2/frequency_matrix_200_{}.csv'.format(name), index_col=0)
    for split in ['train', 'val', 'test']:
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
for split in ['train', 'val', 'test']:
    labels[split] = []
    for date in dates_split[split][13:]:
        if date.replace('_','-') not in t.index:
            labels[split].append(0)
        else:
            tmp = date.replace('_','-')
            labels[split].append(t[tmp])

    labels[split] = np.array(labels[split])
print(labels['train'].shape, labels['val'].shape, labels['test'].shape)

datasets = {}
for name in names:
    datasets[name] = {}
    for split in ['train', 'val', 'test']:
        datasets[name][split] = {}
        for N in Ns:
            datasets[name][split][N] = (datas[name][split][N], labels[split])

auc_results = {}
for N in Ns:
    auc_results[N] = {}
    for name in names:
        train_dataset = datasets[name]['train'][N]
        val_dataset = datasets[name]['val'][N]
        test_dataset = datasets[name]['test'][N]
        
        train_X, train_y = train_dataset
        val_X, val_y = val_dataset
        test_X, test_y = test_dataset

        pre = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform')
        train_X = pre.fit_transform(train_X)
        val_X = pre.transform(val_X)
        test_X = pre.transform(test_X)

        # clf = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)
        clf = Pipeline([
                    # ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
                            ('classification',RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0))
                        ])

        clf.fit(train_X, train_y)
        val_pred = clf.predict_proba(val_X)[:,1]
        test_pred = clf.predict_proba(test_X)[:,1]

        val_auc = roc_auc_score(val_y, val_pred)
        test_auc = roc_auc_score(test_y, test_pred)
        auc_results[N][name] = (val_auc, test_auc)

# the AUC of validation dataset
print("Validation Results")
for name in names:
    print(name, end=' ')
    for N in Ns:
        print("%.2f"%auc_results[N][name][0], end=' ')
    print()

# the AUC of test dataset
print("Test Results")
for name in names:
    print(name, end=' ')
    for N in Ns:
        print("%.2f"%auc_results[N][name][1], end=' ')
    print()
