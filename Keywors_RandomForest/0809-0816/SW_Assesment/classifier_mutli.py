import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,precision_recall_fscore_support
import numpy as np
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
parser.add_argument('--count', type=int,
                    default=200, help=' ')
parser.add_argument('--paths', type = str,
                    default = './results', help='Enter the stored path')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

createDirectory(args.paths)

names = ['ctfidf'] + ['rev_metric1', 'rev_metric2', 'rev_metric3']

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# train : val : test = 8 : 1 : 1
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
    # mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
    
    mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)    
    
    # mat = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
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

precision_results ={}
recall_results ={}
f1_results = {}
auc_results = {}

for N in Ns:
    precision_results[N] ={}
    recall_results[N] ={}
    f1_results[N] = {}
    auc_results[N] = {}
    

    for name in names:
        train_dataset = datasets[name]['train'][N]
        val_dataset = datasets[name]['val'][N]
        test_dataset = datasets[name]['test'][N]
        
        train_X, train_y = train_dataset
        val_X, val_y = val_dataset
        test_X, test_y = test_dataset

        clf = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)

        clf.fit(train_X, train_y)
        val_pred = clf.predict_proba(val_X)[:,1]
        test_pred = clf.predict_proba(test_X)[:,1]
        
        tmp1 = val_y.flatten()
        tmp1 = np.where(tmp1 > 0.5, 0, 1)

        tmp2 = val_pred.flatten()
        tmp2 = np.where(tmp2 > 0.5, 0, 1)

        tmp3 = test_y.flatten()
        tmp3 = np.where(tmp3 > 0.5, 0, 1)

        tmp4 = test_pred.flatten()
        tmp4 = np.where(tmp4 > 0.5, 0, 1)
        
        val_precision = precision_score(tmp1, tmp2)
        test_precision = precision_score(tmp3, tmp4)
        
        val_recall = recall_score(tmp1, tmp2)
        test_recall = recall_score(tmp3, tmp4)
        
        
        val_f1 = f1_score(tmp1, tmp2)
        test_f1 = f1_score(tmp3, tmp4)
        
        precision_results[N][name] = (val_precision, test_precision)
        recall_results[N][name] = (val_recall, test_recall)
        f1_results[N][name] = (val_f1, test_f1)

        try : 
            val_auc = roc_auc_score(val_y, val_pred)
        except:
            val_auc = 0.1

        try : 
            test_auc = roc_auc_score(test_y, test_pred)
        except:
            test_auc = 0.1
        auc_results[N][name] = (val_auc, test_auc)


# print("######################## Precision Results ########################")
# # the precision of validation dataset
# print("Validation Precision")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%precision_results[N][name][0], end=' ')
#     print()
# # the precision of test dataset
# print("Test Precision")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%precision_results[N][name][1], end=' ')
#     print()

# print("######################## Recall Results ########################")
# # the recall of validation dataset
# print("Validation Recall")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%recall_results[N][name][0], end=' ')
#     print()
# # the recall of test dataset
# print("Test Recall")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%recall_results[N][name][1], end=' ')
#     print()

# print("######################## F1-SCORE Results ########################")
# # the f1-score of validation dataset
# print("Validation f1-score")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%f1_results[N][name][0], end=' ')
#     print()
# # the f1-score of test dataset
# print("Test f1-score")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%f1_results[N][name][1], end=' ')
#     print()

# print("######################## AUC Results ########################")
# # the AUC of validation dataset
# print("Validation AUC")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%auc_results[N][name][0], end=' ')
#     print()
# # the AUC of test dataset
# print("Test AUC")
# for name in names:
#     print(name, end=' ')
#     for N in Ns:
#         print("%.2f"%auc_results[N][name][1], end=' ')
#     print()

print("%s Validation & Test Results stored in the input directory"%args.lang)
with open(args.paths+"/%s_metrics.txt"%args.lang,"w") as text_file :
    print("Validation AUC", file=text_file)
    for name in names:
        print(name, end=' ', file=text_file)
        for N in Ns:
            print("%.2f"%auc_results[N][name][0], end=' ', file=text_file)
        print(end='\n', file=text_file)
    # the AUC of test dataset
    print("Test AUC", file=text_file)
    for name in names:
        print(name, end=' ', file=text_file)
        for N in Ns:
            print("%.2f"%auc_results[N][name][1], end=' ', file=text_file)
        print(end='\n', file=text_file)
