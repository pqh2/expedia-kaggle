import csv
import gzip
###
import numpy as np
###

from heapq import nlargest
from operator import itemgetter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
import sklearn
from sknn.mlp import *
from sklearn.feature_extraction import DictVectorizer
from dateutil.parser import parse
from nltk.stem.snowball import SnowballStemmer
import editdistance
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from nltk.corpus import words
import gensim
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn import decomposition
from sklearn import datasets
from collections import defaultdict

raw_train = pandas.read_csv("/Users/patrickhess/Documents/kaggle/expedia/train", encoding="ISO-8859-1")
#dest_raw = pandas.read_csv("/Users/patrickhess/Documents/kaggle/expedia/destinations", encoding="ISO-8859-1")
raw_test = pandas.read_csv("/Users/patrickhess/Documents/kaggle/expedia/test", encoding="ISO-8859-1")


for_aggreggating = pandas.read_csv('/Users/patrickhess/Documents/kaggle/expedia/train',
                    dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    chunksize=1000000)


best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))

total = 0
f = open("/Users/patrickhess/Documents/kaggle/expedia/train", "r")

# Calc counts
while 1:
    line = f.readline().strip()
    total += 1
    if total % 10000000 == 0:
        print('Read {} lines...'.format(total))
    if line == '':
        break
    arr = line.split(",")
    #book_year = int(arr[0][:4])
    user_location_city = arr[5]
    orig_destination_distance = arr[6]
    srch_destination_id = arr[16]
    #is_booking = int(arr[18])
    hotel_country = arr[21]
    hotel_market = arr[22]
    hotel_cluster = arr[23]
    if user_location_city != '' and orig_destination_distance != '':
        best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

f.close()

aggs = []
for chunk in for_aggreggating:
    agg = chunk.groupby(['srch_destination_id',
                         'hotel_cluster'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)

print('')
aggs = pandas.concat(aggs, axis=0)
aggs.head()

CLICK_WEIGHT = 0.05
agg = aggs.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
agg.head()

def most_popular(group, n_max=20):
    return group


group = agg.groupby(['srch_destination_id']).apply(most_popular)

dest_relevance_total = {}
for index, row in group.iterrows():
    srch_dest = str(int(row['srch_destination_id']))
    cluster = str(int(row['hotel_cluster']))
    relevance = row['relevance']
    if srch_dest in dest_relevance_total:
        dest_relevance_total[srch_dest] += relevance
    else:
        dest_relevance_total[srch_dest] = relevance

relative_relevance = {str(i) : {} for i in range(100000)}

for index, row in group.iterrows():
    srch_dest = str(int(row['srch_destination_id']))
    cluster = str(int(row['hotel_cluster']))
    relevance = row['relevance']
    relative_relevance[srch_dest][cluster] = relevance / dest_relevance_total[srch_dest]


distance_country_dict = {}

Xorg = []
Y = []





def generate_point(row, is_test=False):
    data_point = {}
    #data_point['site_name'] = str(row['site_name'])
    data_point['posa_continent'] =  str(row['posa_continent'])
    #data_point['user_location_country'] = str(row['user_location_country'])
    #if row['orig_destination_distance'] != 'nan':
    #    data_point['distance'] =  float(row['orig_destination_distance'])
    #else:
    #    data_point['distance'] = 1970
    #if numpy.isnan(data_point['distance']):
    #    data_point['distance'] = 1970
    data_point['is_mobile'] = row['is_mobile']
    data_point['is_package']  = row['is_package']
    data_point['srch_rm_cnt'] = int(row['srch_rm_cnt'])
    data_point['srch_children_cnt'] =  int(row['srch_children_cnt'])
    data_point['hotel_continent'] = str(row['hotel_continent'])
    #data_point['hotel_country'] =  str(row['hotel_country'])
    data_point['dest_type_id'] = str(row['srch_destination_type_id'])
    data_point['srch_adults_cnt'] = int(row['srch_adults_cnt'])
    ci_date = None
    co_date = None
    try:
        if str(row['srch_ci']) != 'nan':
            ci_date = datetime.strptime(str(row['srch_ci']), "%Y-%m-%d")
        if str(row['srch_co']) != 'nan':
            co_date = datetime.strptime(str(row['srch_co']), "%Y-%m-%d")
    except ValueError:
        print "error"
    if ci_date is not None and co_date is not None:
        data_point['days_stay'] = (co_date - ci_date).days
    else:
        data_point['days_stay'] = 0
    if ci_date is not None:
        data_point['week_day_ci'] = str(ci_date.weekday())
        data_point['month_ci'] = str(ci_date.month)
    else:
        data_point['week_day_ci'] = ""
        data_point['month_ci'] = ""
    if co_date is not None:
        data_point['week_day_co'] = str(co_date.weekday())
        data_point['month_co'] = str(co_date.month)
    else:
        data_point['week_day_co'] = ""
        data_point['month_co'] = ""
    data_point['is_booking'] = str(1 if is_test else row['is_booking'])
    probs = relative_relevance[str(row['srch_destination_id'])]
    dests = {str(i): probs[str(i)] if str(i) in probs else 0 for i in range(101)}
    data_point.update(dests)
    return data_point

for it, row in raw_train.sample(frac=0.1).iterrows():
    Xorg.append(generate_point(row))
    print len(Xorg)
    Y.append(numpy.array(numpy.array(row['hotel_cluster'])))
    if len(Xorg) > 10 * 10**5:
        break
i = 0

out = open('submission.csv', "w")
out.write("id,hotel_cluster\n")
for it, row in raw_test.iterrows():
    s1 = (str(row['user_location_city']), str(row['orig_destination_distance']))
    #print s1
    #print s1
    indexes = []
    if s1 in best_hotels_od_ulc:
        d = best_hotels_od_ulc[s1]
        indexes = [int(x[0]) for x in nlargest(5, sorted(d.items()), key=itemgetter(1))]
        #print len(indexes)
    point =generate_point(row, 1)
    transf = v.transform(point)
    #print transf
    #print transf
    ypred = bst.predict(xgb.DMatrix((transf)))
    #print ypred
    preds = numpy.argsort(ypred)[0][::-1][:15]
    #print preds
    for ind in preds:
        if len(indexes) == 5:
            break
        if ind not in indexes:
            indexes.append(ind)
    out.write(str(i) + ',')
    #print indexes
    for ind in indexes:
        out.write(' ' + str(ind))
    out.write("\n")
    i += 1



Y = numpy.array(Y)
v = DictVectorizer(sparse=False)
X = v.fit_transform(Xorg)
#Y = raw_train[:2000001]['hotel_cluster']
#rfc1 = RandomForestClassifier(n_estimators=200)
#rfc1.fit(X, Y)
param = {}
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.05
param['max_depth'] = 15
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 100
param['verbose'] = True

"""
params = {
        'learning_rate': [0.05],
        'max_depth': [5, 10, 15],
        'min_child_weight':  [1, 2, 4],
}
xgbmodel = xgb.XGBClassifier()
clf1 = GridSearchCV(xgbmodel, params, verbose=1, n_jobs = 4, cv=3, early_stopping_rounds=5, verbose_eval=True, num_boost_round=25)
clf1.fit(X, Y)
"""


def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', metric

"""pca = decomposition.PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
"""

num_round = 25
bst = xgb.train(param, xgb.DMatrix(X[:750000], Y[:750000]), 5)
Ypred = bst.predict(xgb.DMatrix(X[950000:]))
indexes = []
for y in Ypred:
    indexes.append(numpy.argsort(y)[::-1][:5])

Xtest = []

score = 0
for i, y in enumerate(Y[950000:]):
    if y in indexes[i]:
        score += 1




out = open('submission.csv', "w")
out.write("id,hotel_cluster\n")
for i in range(len(indexes)):
    out.write(str(i) + ',')
    for ind in indexes[i]:
        out.write(' ' + str(ind))
    out.write("\n")

