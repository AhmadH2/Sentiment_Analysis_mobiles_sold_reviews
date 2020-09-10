
"""
@author: ahmad
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

with open('Mobiles.data', 'rb') as file:
    df = pickle.load(file)
    
# Drop missing values
df.dropna(inplace=True)

# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positively Rated'], 
                                                    random_state=0)

# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('')
print('AUC: ', roc_auc_score(y_test, predictions))
print('')
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs (Tokens mostly used in bad reviews):\n{}\n'.format(feature_names[sorted_coef_index[:15]]))
print('Largest Coefs (Tokens mostly used in good reviews): \n{}'.format(feature_names[sorted_coef_index[:-15:-1]]))
print('')
# These reviews are correctly identified good or bad
#output [1,0]
review_pred = model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working']))

print('positively prediction of these reviews:\n"not an issue, phone is working": {}\n"an issue, phone is not working": {}'.format(review_pred[0], review_pred[1]))    