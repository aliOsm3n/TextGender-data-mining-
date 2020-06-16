# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:53:25 2017

@author: AliOthman
"""

from nltk.corpus import names
import nltk
from nltk import NaiveBayesClassifier
import random


def gender_features(name):
	# return the last letter of the name
	return {'last_letter': name[-1]}

print(gender_features('Anurag'))

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
#print labeled_names

random.shuffle(labeled_names)
# print (labeled_names)
    
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = NaiveBayesClassifier.train(train_set)

print classifier.classify(gender_features('Addie'))
#print classifier.classify(gender_features('Anurag'))

print nltk.classify.accuracy(classifier ,test_set)
classifier.show_most_informative_features(5)