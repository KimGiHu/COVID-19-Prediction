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
args = parser.parse_args()

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()

for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

print(f'dates[0] = {dates[0]}')

# train : val : test = 8 : 1 : 1
train_dates = pd.date_range(start='2019-08-01', end='2022-07-31', freq='D').tolist()
test_dates = pd.date_range(start='2022-08-01', end='2023-02-28', freq='D').tolist()

# train : val : test = 6 : 2 : 2
# train_dates = pd.date_range(start='2019-08-01', end='2021-08-31', freq='D').tolist()
# val_dates = pd.date_range(start='2021-09-01', end='2022-05-31', freq='D').tolist() 
# test_dates = pd.date_range(start='2022-06-01', end='2023-02-28', freq='D').tolist()

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

print(len(train_dates), len(test_dates))

# set the stored data path
current_path = os.getcwd()

path = os.path.join(current_path, 'python_TF-IDF_%s'%args.lang)

# to redefine the date times for stretch the training period
dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

keywords = pd.read_csv(path + '/keywords_noun_%s.csv'%args.count)['word'].tolist()

""" 1. seperate into key and non-key groups """
""" 2. calculate the word importance metric from the vectors """
key_vector = []
non_key_vector = []
   
key_date = pd.DataFrame(pd.read_csv(path+'/key_dates_%s.csv'%args.lang), columns=['date','label'])

for date in train_dates:
    # vector = np.loadtxt(path+'/vectors2/%s.csv'%date, delimiter=',')
    if (args.lang == 'Republic of Korea'):
        vector = np.loadtxt(path+'/vectors2_noun/vectors2%s.csv'%date, delimiter=',')
    else: 
        vector = np.loadtxt(path+'/vectors2_noun/%s.csv'%date, delimiter=',')
    if date.replace('_','-') in key_date[key_date['label']==1]['date'].tolist():
        key_vector.append(vector)
    else:
        non_key_vector.append(vector)

print("2-1. Proposed Method")
average_key = np.average(key_vector, axis=0)
average_non_key = np.average(non_key_vector, axis=0)

appearance_key = np.sum(np.array(key_vector) > 0, axis=0)
appearance_non_key = np.sum(np.array(non_key_vector) > 0, axis=0)

# probability expression
proportion_key = (np.sum(key_vector, axis=0)) / (np.sum(np.array(key_vector)))
proportion_non_key = (np.sum(non_key_vector, axis=0)) / (np.sum(np.array(non_key_vector)))


### soykeyword variation description
# metric1: proportion-based, divided the number of times a keyword appears in a document set by the total number of words in the document set
# metric2: appearance-based, counted once per document, no additional calculation
# metric3: average-based, divided the number of times a keyword appears in a document set by the total number of documents in the document set
soykeyword_metric1 = proportion_key/(proportion_key+proportion_non_key+1e-8)
soykeyword_metric2 = appearance_key/(appearance_key+appearance_non_key+1e-8)
soykeyword_metric3 = average_key / (average_key + average_non_key+1e-8)

key_freq = np.sum(key_vector, axis=0)
non_key_freq = np.sum(non_key_vector, axis=0)
total_freq = key_freq + non_key_freq

freq_metric = np.tanh(0.0001486977747764619 * key_freq )

# freq_metric = np.tanh(0.00015 * key_freq )

rev_metric1 = soykeyword_metric1 * freq_metric
rev_metric2 = soykeyword_metric2 * freq_metric
rev_metric3 = soykeyword_metric3 * freq_metric

if (args.lang == 'Republic of Korea'):
    for i in range (len(keywords)):
        if len(keywords[i]) <= 1 :
            rev_metric1[i] = -1
            rev_metric2[i] = -1
            rev_metric3[i] = -1

words = pd.read_csv(path+'/keywords_noun_%s.csv'%args.count)['word'].tolist()

data_metric = {'keywords':words,
               'rev_metric1':rev_metric1, 'rev_metric2':rev_metric2, 'rev_metric3':rev_metric3,
               }

# keywords number : 2000
# createDirectory(path+'/keywords2000_noun_month')
createDirectory(path+'/keywords2000_noun_cross_val')
pd.DataFrame(data_metric).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/metric_for_paper_%s.csv'%args.count, index=False)

# to form the total matrix of dates : keyword
complete_matrix = {}

for date in dates:
    if (args.lang == 'Republic of Korea'):
        vector = pd.read_csv(path+'/vectors2_noun/vectors2%s.csv'%date, header=None).values.flatten()
        news_len = pd.read_csv(path+'/dvectors_noun/dvectors%s.csv'%date, header=None)
    else: 
        vector = pd.read_csv(path+'/vectors2_noun/%s.csv'%date, header=None).values.flatten()
        news_len = pd.read_csv(path+'/dvectors_noun/%s.csv'%date, header=None)
    complete_matrix[date] = vector / len(news_len)

complete_table = pd.DataFrame(complete_matrix, index=keywords)
# complete_table.to_csv(path+'/keywords2000_noun_month/complete_table_%s.csv'%args.count)
complete_table.to_csv(path+'/keywords2000_noun_cross_val/complete_table_%s.csv'%args.count)


# to pick out the top keywords from the complete matrix
# to store the highest value of the 'named' metrics in dates
names = ['rev_metric1', 'rev_metric2', 'rev_metric3']
for name in names:
    data_metric = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/metric_for_paper_%s.csv'%args.count))
    
    # data_metric = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/metric_for_paper_%s.csv'%args.count))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = data_metric['keywords'].tolist()

    pd.DataFrame(keywords).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    total_data.to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/frequency_matrix_%s'%args.count+'_%s.csv'%name)
    
    print(name, total_data.columns.tolist()[:15])
    # pd.DataFrame(keywords).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/%s.csv'%name, index=False, header=False)
    # total_data = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    # total_data.to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/frequency_matrix_%s'%args.count+'_%s.csv'%name)

print("2-2. Class-Based TF-IDF Baselines")
tf_tc = np.sum(key_vector, axis=0)
tf_t = np.sum(key_vector, axis=0) + np.sum(non_key_vector, axis=0)

A = ( np.sum(key_vector) + np.sum(non_key_vector) ) / 2

ctfidf = tf_tc * np.log( 1 + (A / (tf_t + 1e-8)) )

tf_tc = np.sum(key_vector, axis=0)
tf_t = np.sum(key_vector, axis=0) + np.sum(non_key_vector, axis=0)

A = ( np.sum(key_vector) + np.sum(non_key_vector) ) / 2

ctfidf = tf_tc * np.log( 1 + (A / (tf_t + 1e-8)) )

words = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords_noun_%s.csv'%args.count)['word'].tolist()

if (args.lang == 'Republic of Korea'):
    for i in range (len(words)):
        if len(words[i]) <= 1 :
            ctfidf[i] = -1
        
data_metric = {'keywords':words,
               'tf_tc':tf_tc,
               'tf_t':tf_t,
               'ctfidf':ctfidf
               }

# pd.DataFrame(data_metric).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/metric_for_paper_ctfidf_%s.csv'%args.count, index=False)
pd.DataFrame(data_metric).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/metric_for_paper_ctfidf_%s.csv'%args.count, index=False)


for name in ['ctfidf']:
    # data_metric = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/metric_for_paper_ctfidf_%s.csv'%args.count))
    data_metric = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/metric_for_paper_ctfidf_%s.csv'%args.count))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords_noun_%s.csv'%args.count)['word'].tolist()
    keywords = [keywords[i] for i in data_metric.index.tolist()]

    # pd.DataFrame(keywords).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/%s.csv'%name, index=False, header=False)
    # total_data = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    # total_data.to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_month/frequency_matrix_%s'%args.count+'_%s.csv' %name)
    
    pd.DataFrame(keywords).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/complete_table_%s.csv'%args.count, index_col='Unnamed: 0').T[keywords[:2000]]
    total_data.to_csv('./python_TF-IDF_%s'%args.lang+'/keywords2000_noun_cross_val/frequency_matrix_%s'%args.count+'_%s.csv' %name)
    
    print(name, total_data.columns.tolist()[:15])


    
