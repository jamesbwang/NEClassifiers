import pandas as pd
import os
import SentenceGetter as sg
import features as ft
import constants
import sklearn_crfsuite
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import eli5
import scipy


plt.style.use('ggplot')


def trainAndTestCRF():
	data = pd.read_csv(constants.NER_DATASET_DIR).fillna(method='ffill')
	getter = sg.SentenceGetter(data)
	sentences = getter.sentences
	X = [ft.sent2features(s) for s in sentences]
	y = [ft.sent2labels(s) for s in sentences]
	labels = ['O', 'PATH', 'DIS', 'SYMP']
	print(labels)
	crf = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		max_iterations=100,
		all_possible_transitions=False
	)
	params_space = {
		'c1': scipy.stats.expon(scale=0.5),
		'c2': scipy.stats.expon(scale=0.05),
	}
	# use the same metric for evaluation
	f1_scorer = make_scorer(metrics.flat_f1_score,
							average='weighted', labels=labels)
	# search
	rs = RandomizedSearchCV(crf, params_space,
							cv=3,
							verbose=1,
							n_jobs=-1,
							n_iter=50,
							scoring=f1_scorer)
	rs.fit(X, y)
	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)
	print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
	crf = rs.best_estimator_
	y_pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
	print(metrics.flat_classification_report(
		y, y_pred, labels=sorted(labels), digits=3
	))

	if not os.path.exists(constants.CRF_DIR):
		os.makedirs(constants.CRF_DIR)
	with open(os.path.join(constants.CRF_DIR, 'weights_optimized.html'), 'w+', encoding='utf-8') as f:
		f.write(eli5.show_weights(crf, top=30).data)

	testdata = pd.read_csv(constants.TESTDATA_DIR)
	getter = sg.SentenceGetter(testdata)
	sentences = getter.sentences
	X = [ft.sent2features(s) for s in sentences]
	y = [ft.sent2labels(s) for s in sentences]
	y_pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
	print(metrics.flat_classification_report(
		y, y_pred, labels=sorted(labels), digits=3
	))
