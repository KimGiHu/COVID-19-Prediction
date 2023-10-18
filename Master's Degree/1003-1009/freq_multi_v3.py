# import the necessary packages
import pandas as pd 
import numpy as np
import os
import re
import argparse

# python inputs : language, count, store_path
parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str,
                    default='United States', help='Select the language')
parser.add_argument('--count', type=int,
                    default=200, help='Choose your vocab size')
parser.add_argument('--pumsa', type=str,
                    default='NOUN', help=' ')
args = parser.parse_args()

# to make the directory which does not exist
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()

for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# train : test = 8 : 2 for K-fold cross Validation
train_dates = pd.date_range(start='2019-08-01', end='2022-06-30', freq='D').tolist()
test_dates = pd.date_range(start='2022-07-01', end='2023-02-28', freq='D').tolist()


# to seperate the train and test dates
for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

# set the stored data path
current_path = os.getcwd()

path = os.path.join(current_path, '%s'%args.lang)
createDirectory(path)


# to redefine the date times for stretch the training period
dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

# define the key_vector/non_key_vector to append the vocabs when confirmed cases is increased
key_vector1 = []
non_key_vector1 = []

key_vector2 = []
non_key_vector2 = []

key_vector3 = []
non_key_vector3 = []

re_factor_mean = pd.DataFrame(pd.read_csv(path+'/Re_mean_%s.csv'%args.lang), columns=['date','label'])

for date in train_dates :
    if (args.lang == 'Republic of Korea'):
        vector = np.loadtxt(path+'/vectors2_%s'%args.pumsa+'/%s.csv'%date, delimiter=',') # preprocessing only nouns
    else: 
        vector = np.loadtxt(path+'/vectors2_%s'%args.pumsa+'/%s.csv'%date, delimiter=',')
    # re_factor_mean < 0.8 
    if date.replace('_','-') in re_factor_mean[re_factor_mean['label']==0]['date'].tolist():
        key_vector1.append(vector)   
    else:
        non_key_vector1.append(vector)
    # re_factor_mean >= 0.8 and <1.2    
    if date.replace('_','-') in re_factor_mean[re_factor_mean['label']==1]['date'].tolist():
        key_vector2.append(vector)
    else:
        non_key_vector2.append(vector)
    # re_factor_mean > 1.2    
    if date.replace('_','-') in re_factor_mean[re_factor_mean['label']==2]['date'].tolist():
        key_vector3.append(vector)
    else:
        non_key_vector3.append(vector)

# # to show the length of key_vectors and non_key_vecotrs which is related to Re_factor_mean
# and the ratio is showed >> 43+708+131(882) : 1022+357+934(2313) >>> keyvector/nonkeyvector = 38%
# print(len(key_vector1)); print(len(non_key_vector1))
# print(len(key_vector2)); print(len(non_key_vector2))
# print(len(key_vector3)); print(len(non_key_vector3))

print("2-1. Keywords Extraction")
# relative tf_tc term
r_tf_tc1 = np.sum(key_vector1, axis=0)/43
r_tf_tc2 = np.sum(key_vector2, axis=0)/708
r_tf_tc3 = np.sum(key_vector3, axis=0)/131

# relative tf_total term
tf_t13 = r_tf_tc1 + r_tf_tc3
tf_t12 = r_tf_tc1 + r_tf_tc2
tf_t23 = r_tf_tc2 + r_tf_tc3

A13 = (np.sum(key_vector1)+np.sum(key_vector3)) / 2
A12 = (np.sum(key_vector1)+np.sum(key_vector2)) / 2
A23 = (np.sum(key_vector2)+np.sum(key_vector3)) / 2

rev_metric1 = (r_tf_tc1-r_tf_tc3) * np.log( 1 + (A13 / (tf_t13 + 1e-8)) )
rev_metric2 = -rev_metric1
rev_metric3 = (r_tf_tc1-r_tf_tc2) * np.log( 1 + (A12 / (tf_t12 + 1e-8)) ) 
rev_metric4 = -rev_metric3
rev_metric5 = (r_tf_tc2-r_tf_tc3) * np.log( 1 + (A23 / (tf_t23 + 1e-8)) )
rev_metric6 = -rev_metric5

createDirectory(path+'/new_keywords_%s'%args.pumsa)

words = pd.read_csv(path+'/keywords_%s'%args.pumsa+'_%s.csv'%args.count)['word'].tolist()
keywords = pd.read_csv(path +'/keywords_%s'%args.pumsa+'_%s.csv'%args.count)['word'].tolist()
for i in range (len(keywords)):
    if len(keywords[i]) <= 1 :
        rev_metric1[i] = -1
        rev_metric2[i] = -1
        rev_metric3[i] = -1

data_metric = {'keywords':words,
               'new_metric1':rev_metric1, 'new_metric2':rev_metric2, 'new_metric3':rev_metric3,
               'new_metric4':rev_metric4, 'new_metric5':rev_metric5, 'new_metric6':rev_metric6,
               }

# keywords number : 2000
pd.DataFrame(data_metric).to_csv(path+'/new_keywords_%s'%args.pumsa+'/metric_for_paper_%s.csv'%args.count, index=False)

# to form the total matrix of dates : keyword
complete_matrix = {}

for date in dates:
    if (args.lang == 'Republic of Korea'):
        vector = pd.read_csv(path+'/vectors2_%s'%args.pumsa+'/%s.csv'%date, header=None).values.flatten()
        news_len = pd.read_csv(path+'/dvectors_%s'%args.pumsa+'/%s.csv'%date, header=None)
    else: 
        vector = pd.read_csv(path+'/vectors2_%s'%args.pumsa+'/%s.csv'%date, header=None).values.flatten()
        news_len = pd.read_csv(path+'/dvectors_%s'%args.pumsa+'/%s.csv'%date, header=None)
    complete_matrix[date] = vector / len(news_len)

complete_table = pd.DataFrame(complete_matrix, index=keywords)
complete_table.to_csv(path+'/new_keywords_%s'%args.pumsa+'/complete_table_%s.csv'%args.count)

# to pick out the top keywords from the complete matrix
# to store the highest value of the 'named' metrics in dates
names = ['new_metric1', 'new_metric2', 'new_metric3','new_metric4', 'new_metric5', 'new_metric6']
for name in names:
    data_metric = pd.DataFrame(pd.read_csv(path+'/new_keywords_%s'%args.pumsa+'/metric_for_paper_%s.csv'%args.count))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = data_metric['keywords'].tolist()

    pd.DataFrame(keywords).to_csv(path+'/new_keywords_%s'%args.pumsa+'/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv(path+'/new_keywords_%s'%args.pumsa+'/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    total_data.to_csv(path+'/new_keywords_%s'%args.pumsa+'/frequency_matrix_%s'%args.count+'_%s.csv'%name)
    
    print(name, total_data.columns.tolist()[:15])

## 2-2. Class-Based TF-IDF Based Baselines
tf_tc1 = np.sum(key_vector1, axis=0)
tf_t1 = np.sum(key_vector1, axis=0) + np.sum(non_key_vector1, axis=0)
A1 = ( np.sum(key_vector1) + np.sum(non_key_vector1) ) / 2
ctfidf1 = tf_tc1 * np.log( 1 + (A1 / (tf_t1 + 1e-8)) )

tf_tc2 = np.sum(key_vector2, axis=0)
tf_t2 = np.sum(key_vector2, axis=0) + np.sum(non_key_vector2, axis=0)
A2 = ( np.sum(key_vector2) + np.sum(non_key_vector2) ) / 2
ctfidf2 = tf_tc2 * np.log( 1 + (A2 / (tf_t2 + 1e-8)) )

tf_tc3 = np.sum(key_vector3, axis=0)
tf_t3 = np.sum(key_vector3, axis=0) + np.sum(non_key_vector3, axis=0)
A3 = ( np.sum(key_vector3) + np.sum(non_key_vector3) ) / 2
ctfidf3 = tf_tc3 * np.log( 1 + (A3 / (tf_t3 + 1e-8)) )


data_metric = {'keywords':words,
               'tf_tc1':tf_tc1,
               'tf_t1':tf_t1,
               'ctfidf1':ctfidf1,
               'tf_tc2':tf_tc2,
               'tf_t2':tf_t2,
               'ctfidf2':ctfidf2,
               'tf_tc3':tf_tc3,
               'tf_t3':tf_t3,
               'ctfidf3':ctfidf3
               }

pd.DataFrame(data_metric).to_csv(path+'/new_keywords_%s'%args.pumsa+'/metric_for_paper_%s_ctfidf.csv'%args.count, index=False)

for name in ['ctfidf1','ctfidf2','ctfidf3']:
    data_metric = pd.DataFrame(pd.read_csv(path+'/new_keywords_%s'%args.pumsa+'/metric_for_paper_%s_ctfidf.csv'%args.count))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = data_metric['keywords'].tolist()

    pd.DataFrame(keywords).to_csv(path+'/new_keywords_%s'%args.pumsa+'/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv(path+'/new_keywords_%s'%args.pumsa+'/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    total_data.to_csv(path+'/new_keywords_%s'%args.pumsa+'/frequency_matrix_%s'%args.count+'_%s.csv'%name)
    
    print(name, total_data.columns.tolist()[:15])
