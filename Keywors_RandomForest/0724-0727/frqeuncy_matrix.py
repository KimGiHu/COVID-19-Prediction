# import the necessary packages
import pandas as pd 
import numpy as np
import os
import argparse

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


""" collect a list of dates """
# to return a fixed frequency DatetimeIndex. 'D' meaning : calendar day frequency
dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()

# reference : https://www.programiz.com/python-programming/datetime/strftime
# strftime :  method returns a string representing date and time using date, time or datetime object.
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

print(f'dates[0] = {dates[0]}')


train_dates = pd.date_range(start='2019-08-01', end='2022-08-31', freq='D').tolist()
val_dates = pd.date_range(start='2022-09-01', end='2022-11-30', freq='D').tolist() 
test_dates = pd.date_range(start='2022-12-01', end='2023-02-28', freq='D').tolist() 

for i in range(len(train_dates)):
    train_dates[i] = train_dates[i].strftime('%Y_%m_%d')

for i in range(len(val_dates)):
    val_dates[i] = val_dates[i].strftime('%Y_%m_%d')

for i in range(len(test_dates)):
    test_dates[i] = test_dates[i].strftime('%Y_%m_%d')

# substract a list of 'USA' type dataset from the multi-language datasets
path = []

temp = os.listdir('./data_covid')
for item in temp :
    if args.lang in item :
        path.append(item)
path.sort()


# to redefine the date times for stretch the training period
dates = pd.date_range(start='2019-08-01', end='2023-02-28', freq='D').tolist()
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%Y_%m_%d')

keywords = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords_200.csv')['word'].tolist()

""" 1. seperate into key and non-key groups """
""" 2. calculate the word importance metric from the vectors """
key_vector = []
non_key_vector = []



key_date = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/key_dates_%s.csv'%args.lang), columns=['date','label'])

for date in train_dates:
    vector = np.loadtxt('./python_TF-IDF_%s'%args.lang+'/vectors2/%s.csv'%date, delimiter=',')
    if date.replace('_','-') in key_date[key_date['label']==1]['date'].tolist():
        key_vector.append(vector)
    else:
        non_key_vector.append(vector)

# print(key_vector)
average_key = np.average(key_vector, axis=0)
# print(average_key)
average_non_key = np.average(non_key_vector, axis=0)
# print(average_non_key)
appearance_key = np.sum(np.array(key_vector) > 0, axis=0)
appearance_non_key = np.sum(np.array(non_key_vector) > 0, axis=0)

# error message :
# TF_IDF.py:77: RuntimeWarning: invalid value encountered in scalar divide
#  proportion_key = np.sum(key_vector, axis=0) / np.sum(np.array(key_vector))
# print(key_vector)
# print(non_key_vector)

# tmp1 = np.sum(key_vector, axis=0)
# tmp2 = np.sum(np.array(key_vector))
# print("tmp1")
# print(tmp1)
# print("tmp2")
# print(tmp2)
proportion_key = (np.sum(key_vector, axis=0)) / (np.sum(np.array(key_vector)))
# proportion_key = tmp1 / tmp2
proportion_non_key = (np.sum(non_key_vector, axis=0)) / (np.sum(np.array(non_key_vector)))
# exit()

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
freq_metric = np.tanh(30000 * key_freq / np.sum(total_freq))
rev_metric1 = soykeyword_metric1 * freq_metric
rev_metric2 = soykeyword_metric2 * freq_metric
rev_metric3 = soykeyword_metric3 * freq_metric

words = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/keywords_200.csv')['word'].tolist()

data_metric = {'keywords':words,
               'average_key':average_key, 'average_non_key':average_non_key,
               'freq_key':np.sum(key_vector, axis=0), 'freq_non_key':np.sum(non_key_vector, axis=0),
               'total_freq': total_freq, 'freq_metric':freq_metric, 'rev_metric1':rev_metric1, 'rev_metric2':rev_metric2, 'rev_metric3':rev_metric3,
               'soykeyword_metric1':soykeyword_metric1, 'soykeyword_metric2':soykeyword_metric2, 'soykeyword_metric3':soykeyword_metric3}

pd.DataFrame(data_metric).to_csv('./python_TF-IDF_%s'%args.lang+'/metric_for_paper.csv', index=False)

# to form the total matrix of dates : keyword
complete_matrix = {}

for date in dates:
    vector = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/vectors2/%s.csv'%date, header=None).values.flatten()
    complete_matrix[date] = vector

complete_table = pd.DataFrame(complete_matrix, index=keywords)
complete_table.to_csv('./python_TF-IDF_%s'%args.lang+'/complete_table.csv')

createDirectory('./python_TF-IDF_%s'%args.lang+'/keywords_200')

# to pick out the top keywords from the complete matrix
# to store the highest value of the 'named' metrics in dates
names = ['soykeyword_metric1', 'soykeyword_metric2', 'soykeyword_metric3', 'average_key', 'average_non_key', 'freq_key', 'freq_non_key', 'total_freq', 'freq_metric', 'rev_metric1', 'rev_metric2', 'rev_metric3']
for name in names:
    data_metric = pd.DataFrame(pd.read_csv('./python_TF-IDF_%s'%args.lang+'/metric_for_paper.csv'))
    data_metric.sort_values(by=[name], ascending=False, inplace=True)
    keywords = data_metric['keywords'].tolist()
    pd.DataFrame(keywords).to_csv('./python_TF-IDF_%s'%args.lang+'/keywords_200/%s.csv'%name, index=False, header=False)
    total_data = pd.read_csv('./python_TF-IDF_%s'%args.lang+'/complete_table.csv', index_col='Unnamed: 0').T[keywords[:300]]
    total_data.to_csv('./python_TF-IDF_%s'%args.lang+'/frequency_matrix_%s.csv'%name)
