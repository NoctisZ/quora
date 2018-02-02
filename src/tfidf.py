import pandas as pd;
import numpy as np;
import xgboost as xgb;
from nltk.corpus import stopwords
from collections import Counter
from sklearn.cross_validation import train_test_split

print('Reading train.csv')
df_train = pd.read_csv('../data/train.csv')
print('Reading test.csv')
df_test = pd.read_csv('../data/test.csv')
print('Generating text corpus')
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

stops = set(stopwords.words("english"))

def cosine_similarity(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def get_idf(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_idf(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


x_train = pd.DataFrame()
x_test = pd.DataFrame()
print('Creating train features')
x_train['word_match'] = df_train.apply(cosine_similarity, axis=1, raw=True)
x_train['tfidf_word_match'] = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
print('Creating test features')
x_test['word_match'] = df_test.apply(cosine_similarity, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
y_train = df_train['is_duplicate']


print('Training with XGBoost')

params = {
	'objective': 'binary:logistic',
	'eval_metric': 'logloss',
	'eta': 0.02,
	'max_depth': 4,
}

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

print('Submitting')
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('submission.csv', mode='w+', index=False)