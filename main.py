import CRFClassifier as crf
import NGramAnalysis as nga
import constants
import RFClassifier as rf


def main():
	biGramAnalyzer = nga.NGramAnalyzer(2)
	print('made analyzer')
	biGramAnalyzer.createNGramsTrainingSet()
	print('created bigram dataset')
	del biGramAnalyzer

	triGramAnalyzer = nga.NGramAnalyzer(3)
	print('made analyzer')
	triGramAnalyzer.createNGramsTrainingSet()
	print('created trigram dataset')
	del triGramAnalyzer

	tetraGramAnalyzer = nga.NGramAnalyzer(4)
	print('made analyzer')
	tetraGramAnalyzer.createNGramsTrainingSet()
	print('created tetragram dataset')
	del tetraGramAnalyzer

	pentaGramAnalyzer = nga.NGramAnalyzer(5)
	print('made analyzer')
	pentaGramAnalyzer.createNGramsTrainingSet()
	print('created pentagram dataset')
	del pentaGramAnalyzer
	rf.trainandTestRF()


if __name__ == '__main__':
	main()
