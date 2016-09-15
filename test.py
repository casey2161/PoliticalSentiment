from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from nltk.classify.scikitlearn import SklearnClassifier

from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

train_subj_docs = subj_docs[:80]
train_obj_docs = obj_docs[:80]

test_subj_docs = subj_docs[80:100]
test_obj_docs = obj_docs[80:100]

training_docs = train_subj_docs + train_obj_docs
testing_docs = test_subj_docs + test_obj_docs

sentim = SentimentAnalyzer()

all_words_neg = sentim.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim.unigram_word_feats(all_words_neg, min_freq=4)
sentim.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim.apply_features(training_docs)
testing_set = sentim.apply_features(testing_docs)

trainer = SklearnClassifier(DecisionTreeClassifier()).train
#trainer = SklearnClassifier(LinearSVC()).train
classifier = sentim.train(trainer, training_set)

for key, value in sorted(sentim.evaluate(testing_set).items()):
	print('{0}: {1}'.format(key, value))

