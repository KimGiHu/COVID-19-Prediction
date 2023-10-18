# 2023-09-12, version4
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,f1_score,classification_report,confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer 
import numpy as np
import argparse
import os 
from sklearn.model_selection import KFold
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='root dir for data')
parser.add_argument('--count', type=int,
                    default=200, help=' ')
parser.add_argument('--pumsa', type=str,
                    default='NOUN', help=' ')
parser.add_argument('--paths', type = str,
                    default = './results', help='Enter the stored path')
parser.add_argument('--file', type = str,
                    default = 'log.txt', help='Input the log files')
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
names = ['ctfidf1', 'ctfidf2', 'ctfidf3']+['new_metric1', 'new_metric2', 'new_metric3','new_metric4', 'new_metric5', 'new_metric6']

createDirectory(args.paths+'/%s'%args.lang)
file = open(args.paths+'/%s'%args.lang+'/%s'%args.file)
file.close()

dates = pd.date_range(start='2020-02-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# train : test = 8 : 2
train_dates = pd.date_range(start='2020-02-01', end='2022-06-30', freq='D').tolist()
test_dates = pd.date_range(start='2022-07-01', end='2023-02-28', freq='D').tolist()

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')


current_path = os.getcwd()
path = os.path.join(current_path, '%s'%args.lang)

dates_split = {"train":train_dates, "test":test_dates}

datas = {}
# Ns = range(5, 31, 5)
Ns = range(20, 201, 20)

for name in names:
    datas[name] = {}
    mat = pd.read_csv(path+'/new_keywords_%s'%args.pumsa+'/frequency_matrix_%s'%args.count+'_{}.csv'.format(name), index_col=0)
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

# for split in ['train', 'test']:
#     for N in Ns :
#         tmp = [] 
#         for name in names :
#             tmp.append(datas[name][split][N])
#         # tmp = np.concatenate(tmp,axis=0)
#         tmp = np.concatenate(tmp,axis=1)
#         for name in names : 
#             datas[name][split][N]=tmp

t = pd.read_csv(path+'/Re_mean_%s.csv'%args.lang, index_col='date')['label']

labels = {}
for split in ['train', 'test']:
    labels[split] = []
    
    # not appended keyword matrix
    for date in dates_split[split][13:]:
        if date.replace('_','-') not in t.index:
            labels[split].append(0)
        else:
            tmp = date.replace('_','-')
            labels[split].append(t[tmp])

    labels[split] = np.array(labels[split])

    # # appended keyword matrix 
    # for i in range(0,len(names)):
    #     for date in dates_split[split][13:]:
    #         if date.replace('_','-') not in t.index:
    #             labels[split].append(0)
    #         else:
    #             tmp = date.replace('_','-')
    #             labels[split].append(t[tmp])
    

print(labels['train'].shape, labels['test'].shape)

datasets = {}

for name in names : 
    datasets[name] = {}
    for split in ['train','test']:
        datasets[name][split] = {}
        for N in Ns :
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
        i = 0
        for (train_ids, test_ids) in kfold.split(train_dataset[0]):
            print(str(i+1)+' th')
            train_X, test_X_cv = train_dataset[0][train_ids], train_dataset[0][test_ids]
            train_y, test_y_cv = train_dataset[1][train_ids], train_dataset[1][test_ids]
            # print(np.sum(train_y==0), np.sum(train_y==1), np.sum(train_y==2))
            # continue

            # clf = RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced_subsample', random_state=0)
            # clf = BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all')
            # clf = make_pipeline(StandardScaler(),BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all'))
            # clf = SVC()
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

            clf.fit(train_X,train_y)
            test_pred_cv = clf.predict(test_X_cv)


            # class 0
            TP_0 = np.logical_and(test_pred_cv==0, test_y_cv==0).sum()
            FP_0 = np.logical_and(test_pred_cv==0, test_y_cv!=0).sum()
            FN_0 = np.logical_and(test_pred_cv!=0, test_y_cv==0).sum()
            TN_0 = np.logical_and(test_pred_cv!=0, test_y_cv!=0).sum()
            preicsion_0 = TP_0 / (TP_0+FP_0+1e-8)
            recall_0 = TP_0 / (TP_0+FN_0+1e-8)
            f1_0 = 2 * (preicsion_0 * recall_0) / (preicsion_0 + recall_0 + 1e-8)
            # print(preicsion_0)
            # print(recall_0)
            # print(f1_0)


            # class 2
            TP_2 = np.logical_and(test_pred_cv==2, test_y_cv==2).sum()
            FP_2 = np.logical_and(test_pred_cv==2, test_y_cv!=2).sum()
            FN_2 = np.logical_and(test_pred_cv!=2, test_y_cv==2).sum()
            TN_2 = np.logical_and(test_pred_cv!=2, test_y_cv!=2).sum()
            preicsion_2 = TP_2 / (TP_2+FP_2+1e-8)
            recall_2 = TP_2 / (TP_2+FN_2+1e-8)
            f1_2 = 2 * (preicsion_2 * recall_2) / (preicsion_2 + recall_2 + 1e-8)
            # print(preicsion_2)
            # print(recall_2)
            # print(f1_2)            

            # f1-score
            sum += (f1_2 + f1_0) /2
            # sum += recall_0
            # sum += recall_2
            print(confusion_matrix(test_y_cv, test_pred_cv,labels=[0, 1, 2])) 
            print(classification_report(test_y_cv, test_pred_cv, labels=[0, 1, 2]))
            i += 1

        sum = sum/k
        
        if max < sum :
            max = sum
            index[name] = N
            print(name)
            print(f'F1_score for the {name} {index[name]} keywords  = {max}')

index_dot ={}
index_dot['ctfidf1'] = index['ctfidf1']
index_dot['ctfidf2'] = index['ctfidf2']
index_dot['ctfidf3'] = index['ctfidf3']
index_dot['new_metric1'] = index['new_metric1']
index_dot['new_metric2'] = index['new_metric2']
index_dot['new_metric3'] = index['new_metric3']
index_dot['new_metric4'] = index['new_metric4']
index_dot['new_metric5'] = index['new_metric5']
index_dot['new_metric6'] = index['new_metric6']

f1_results = {}

for name in names:
    train_dataset = datasets[name]['train'][index[name]]
    test_dataset = datasets[name]['test'][index[name]]

    train_X, train_y = train_dataset
    test_X, test_y = test_dataset

    # clf = BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all')
    # clf = make_pipeline(StandardScaler(),BalancedRandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, class_weight='balanced', random_state=0, sampling_strategy = 'all'))
    # clf = SVC()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_X, train_y)
    test_pred = clf.predict(test_X)

    print(name)
    print("%s"%name+"confusion_matrix")
    print(confusion_matrix(test_y, test_pred,labels=[0, 1, 2])) 
    print(classification_report(test_y, test_pred, labels=[0, 1, 2]))

    # test_f1 = f1_score(test_y_cv, test_pred_cv,average='micro')
    # f1_results[name] = test_f1

# for name in names:
#     print("######################## %s F1-score Results ########################"%name)
#     print(name, end=' ')
#     print("F1-score : %.2f"%f1_results[name]+"\tindex : %.2f"%index[name], end=' ')
#     print()
