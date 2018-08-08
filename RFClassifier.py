import constants
import pandas as pd
import SentenceGetter as sg
import features as ft
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
import sklearn.model_selection
from sklearn.metrics import classification_report


def trainandTestRF():
	biGrams = pd.read_csv(os.path.join(str(2) + 'Grams', constants.NGRAM_TRAINING), index_col=0, engine='c').append(
		pd.read_csv(os.path.join(str(2) + 'Grams', constants.NGRAM_TESTING), index_col=0, engine='c'),
		ignore_index=True, sort=True).fillna(int(0))
	triGrams = pd.read_csv(os.path.join(str(3) + 'Grams', constants.NGRAM_TRAINING), index_col=0, engine='c').append(
		pd.read_csv(os.path.join(str(3) + 'Grams', constants.NGRAM_TESTING), index_col=0, engine='c'),
		ignore_index=True, sort=True).fillna(int(0))
	tetraGrams = pd.read_csv(os.path.join(str(4) + 'Grams', constants.NGRAM_TRAINING), index_col=0, engine='c').append(
		pd.read_csv(os.path.join(str(4) + 'Grams', constants.NGRAM_TESTING), index_col=0, engine='c'),
		ignore_index=True, sort=True).fillna(int(0))
	pentaGrams = pd.read_csv(os.path.join(str(5) + 'Grams', constants.NGRAM_TRAINING), index_col=0, engine='c').append(
		pd.read_csv(os.path.join(str(5) + 'Grams', constants.NGRAM_TESTING), index_col=0, engine='c'),
		ignore_index=True, sort=True).fillna(int(0))
	combinedGrams = biGrams.join(triGrams.drop(['SENTENCE_NUMBER', 'TAG', 'WORD'], axis=1)).join(tetraGrams.drop(['SENTENCE_NUMBER', 'TAG', 'WORD'], axis=1)).join(pentaGrams.drop(['SENTENCE_NUMBER', 'TAG', 'WORD'], axis=1)).fillna(0)
	del biGrams, triGrams, tetraGrams, pentaGrams
	train, test = combinedGrams.head(int(len(combinedGrams) * .75)), combinedGrams.tail(
		int(len(combinedGrams) - len(combinedGrams) * .75))
	print(train.head())
	print(test.head())
	print('read CSV')
	del combinedGrams
	X = list(trainTuple for trainTuple in train.drop(['SENTENCE_NUMBER', 'TAG', 'WORD'], axis=1).itertuples())
	y = list(train['TAG'])
	del train
	print('running RF...')
	rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=3)
	rfc.fit(X, y)
	del X, y
	print('testing...')
	X_test = list(testTuple for testTuple in test.drop(['SENTENCE_NUMBER', 'TAG', 'WORD'], axis=1).itertuples())
	y_test = list(test['TAG'])
	del test
	print('running report...')
	pred = rfc.predict(X_test)
	report = classification_report(y_pred=pred, y_true=y_test)
	print(report)
