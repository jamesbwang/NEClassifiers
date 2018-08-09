from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
import os
import constants
import matplotlib.pyplot as plt


class NGramAnalyzer:
	__NGRAM_N = 2
	__NGRAMS_DIR = str(__NGRAM_N) + r'Grams'
	__n_gram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(__NGRAM_N, __NGRAM_N), decode_error='ignore',
										  stop_words=None)

	def __init__(self, n):
		self.__NGRAM_N = n
		self.__NGRAMS_DIR = str(self.__NGRAM_N) + r'Grams'
		self.__n_gram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.__NGRAM_N, self.__NGRAM_N),
												   decode_error='ignore', stop_words=None)

	def __findNGramPatterns(self):
		df = pd.read_csv(constants.DATABASE_DIR)
		numPathGramCount, numDisGramCount, numSympGramCount = 0, 0, 0
		pathGramIndex, disGramIndex, sympGramIndex = {}, {}, {}
		for index, row in df.iterrows():
			if type(row['disease']) == str:
				disList = row['disease'].replace(';', ', ').split(', ')
			else:
				disList = []
			if type(row['symptoms']) == str:
				sympList = row['symptoms'].split(',')
			else:
				sympList = []
			try:
				(self.__n_gram_vectorizer.fit_transform([row['pathogen']]))
			except ValueError:
				print('error')
			numPathGramCount += len(self.__n_gram_vectorizer.get_feature_names())
			for pathNGram in self.__n_gram_vectorizer.get_feature_names():
				if pathNGram.lower() not in pathGramIndex:
					pathGramIndex[pathNGram.lower()] = 0
				pathGramIndex[pathNGram.lower()] += 1
			if len(disList) != 0:
				try:
					(self.__n_gram_vectorizer.fit_transform(disList))
				except ValueError:
					print('error')
				numDisGramCount += len(self.__n_gram_vectorizer.get_feature_names())
				for disNGram in self.__n_gram_vectorizer.get_feature_names():
					if disNGram.lower() not in disGramIndex:
						disGramIndex[disNGram.lower()] = 0
					disGramIndex[disNGram.lower()] += 1
			if len(sympList) != 0:
				try:
					(self.__n_gram_vectorizer.fit_transform(sympList))
				except ValueError:
					print('error')
				self.__n_gram_vectorizer.fit_transform(sympList)
				numSympGramCount += len(self.__n_gram_vectorizer.get_feature_names())
				for sympNGram in self.__n_gram_vectorizer.get_feature_names():
					if sympNGram.lower() not in sympGramIndex:
						sympGramIndex[sympNGram.lower()] = 0
					sympGramIndex[sympNGram.lower()] += 1
		return [(pathGramIndex, numPathGramCount), (disGramIndex, numDisGramCount), (sympGramIndex, numSympGramCount)]

	def __findIrrelevantNGrams(self):
		df = pd.read_csv(constants.TESTDATA_DIR, index_col=0)
		s, sentNum = [], df.loc[0, 'sentence_number']
		numGrams, gramIndex = 0, {}
		for index, row in df.iterrows():
			if type(row['word']) != str or row['word'] in string.punctuation:
				continue
			if sentNum == row['sentence_number']:
				s.append(str(row['word'].lower()))
				continue
			if len(s) != 0:
				words = ' '.join([x for x in s])
				try:
					(self.__n_gram_vectorizer.fit_transform([words]))
				except ValueError:
					print('error')
				numGrams += len(self.__n_gram_vectorizer.get_feature_names())
				for gram in self.__n_gram_vectorizer.get_feature_names():
					if gram.lower() not in gramIndex:
						gramIndex[gram.lower()] = 0
					gramIndex[gram.lower()] += 1
				s.clear()
			if row['tag'] == 'O':
				s.append(row['word'].lower())
			sentNum += 1
		return gramIndex, numGrams

	def __writeNGramsOutcomes(self, NGramTuple, otherNGrams, fileName):
		df = pd.DataFrame(columns=['NGram', 'percent_in_tag', 'percent_in_other'])
		# df.loc[df.shape[0]] = ['raw_number', NGramTuple[1], otherNGrams[1]]
		for key in NGramTuple[0]:
			entry = [key, NGramTuple[0][key] / NGramTuple[1], 'unique_to_tag']
			if key in otherNGrams[0]:
				entry[2] = otherNGrams[0][key] / otherNGrams[1]
			df.loc[df.shape[0]] = entry
		df.to_csv(fileName)
		return df

	def __dropUncommonEntries(self, allEntries, df):
		for index, row in df.iterrows():
			if row['NGram'] not in allEntries:
				df = df.drop(index)
		return df

	def analyzeNGrams(self):
		taggingNGrams = self.__findNGramPatterns()
		otherNGrams = self.__findIrrelevantNGrams()
		if not os.path.exists(self.__NGRAMS_DIR):
			os.makedirs(self.__NGRAMS_DIR)
		pathdf = self.__writeNGramsOutcomes(taggingNGrams[0], otherNGrams,
											os.path.join(self.__NGRAMS_DIR, 'pathNGrams.csv')).sort_values(by='NGram')
		disdf = self.__writeNGramsOutcomes(taggingNGrams[1], otherNGrams,
										   os.path.join(self.__NGRAMS_DIR, 'disNGrams.csv')).sort_values(by='NGram')
		sympdf = self.__writeNGramsOutcomes(taggingNGrams[2], otherNGrams,
											os.path.join(self.__NGRAMS_DIR, 'sympNGrams.csv')).sort_values(by='NGram')
		otherdf = self.__writeNGramsOutcomes(otherNGrams, otherNGrams,
											 os.path.join(self.__NGRAMS_DIR, 'otherNGrams.csv')).sort_values(by='NGram')
		allEntries = set(pathdf['NGram']) & set(disdf['NGram']) & set(sympdf['NGram']) & set(otherdf['NGram'])
		fulldf = pd.DataFrame(index=self.__dropUncommonEntries(allEntries, pathdf)['NGram'])
		nameli = ['other', 'symptom', 'disease', 'pathogen']
		for df in [pathdf, disdf, sympdf, otherdf]:
			name = nameli.pop()
			df = self.__dropUncommonEntries(allEntries, df)
			df.plot(x='NGram', y='percent_in_tag', kind='bar', title=name + ' relative frequencies',
					grid=True).get_figure().savefig(os.path.join(self.__NGRAMS_DIR, name + '.png'))
			df = df.set_index('NGram')
			fulldf = pd.concat([fulldf, df['percent_in_tag']], axis=1, ignore_index=True)
		fulldf.columns = ['pathogen', 'disease', 'symptom', 'other']
		fulldf.to_csv(os.path.join(self.__NGRAMS_DIR, 'fullNGrams.csv'))
		fulldf.plot(title=str(self.__NGRAM_N) + 'Grams_full', kind='bar', subplots=True, sharex=True, figsize=(20, 10),
					layout=(4, 1), sharey=True)
		fig = plt.gcf()
		fig.savefig(os.path.join(self.__NGRAMS_DIR, 'fullNGrams.png'))

	def createNGramsTrainingSet(self, name='test'):
		path = constants.TESTDATA_DIR
		newpath = constants.NGRAM_TESTING
		if name == 'train':
			path = constants.NER_DATASET_DIR
			newpath = constants.NGRAM_TRAINING
		taggingNGrams = self.__findNGramPatterns()
		otherNGrams = self.__findIrrelevantNGrams()
		if not os.path.exists(self.__NGRAMS_DIR):
			os.makedirs(self.__NGRAMS_DIR)
		pathdf = self.__writeNGramsOutcomes(taggingNGrams[0], otherNGrams,
											os.path.join(self.__NGRAMS_DIR, 'pathNGrams.csv')).sort_values(by='NGram')
		disdf = self.__writeNGramsOutcomes(taggingNGrams[1], otherNGrams,
										   os.path.join(self.__NGRAMS_DIR, 'disNGrams.csv')).sort_values(by='NGram')
		sympdf = self.__writeNGramsOutcomes(taggingNGrams[2], otherNGrams,
											os.path.join(self.__NGRAMS_DIR, 'sympNGrams.csv')).sort_values(by='NGram')
		otherdf = self.__writeNGramsOutcomes(otherNGrams, otherNGrams,
											 os.path.join(self.__NGRAMS_DIR, 'otherNGrams.csv')).sort_values(by='NGram')
		allPossibleNGrams = set(pathdf['NGram']) | set(disdf['NGram']) | set(sympdf['NGram']) | set(otherdf['NGram'])
		del pathdf, disdf, sympdf, otherdf, taggingNGrams, otherNGrams
		trainingDataSet = pd.read_csv(path, index_col=0).drop('pos', axis=1).rename(
			columns={'word': 'WORD', 'tag': 'TAG', 'sentence_number': 'SENTENCE_NUMBER'}).join(
			pd.DataFrame(columns=sorted(list(allPossibleNGrams)))).fillna(0)
		for index, row in trainingDataSet.iterrows():
			try:
				self.__n_gram_vectorizer.fit_transform([str(row['WORD'])])
			except ValueError:
				print('error')
			for i in self.__n_gram_vectorizer.get_feature_names():
				if i in allPossibleNGrams:
					trainingDataSet.loc[index, i] = 1
			self.__n_gram_vectorizer.get_feature_names().clear()
		trainingDataSet.to_csv(os.path.join(self.__NGRAMS_DIR, newpath))
