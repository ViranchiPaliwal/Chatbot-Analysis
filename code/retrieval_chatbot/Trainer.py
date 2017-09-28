"""
Trains agent using a list of prompts and a list of corresponding responses. Uses Tf-idf to rank responses.
"""

import string

from nltk import *
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from nltk.stem.porter import PorterStemmer
from math import log

from DialogueProcessor import processDialogues


def tokenize(prompt):
	prompt = prompt.lower()
	translator = prompt.maketrans('', '', string.punctuation)
	promptNoPuncs = prompt.translate(translator)
	tokens = word_tokenize(promptNoPuncs)
	tokens = [token for token in tokens if not token in corpus.stopwords.words('english')]
	stemmer = PorterStemmer()
	tokens = stemTokens(tokens, stemmer)
	return tokens


def stemTokens(tokens, stemmer):
	stemmedTokens = []
	for token in tokens:
		stemmedTokens.append(stemmer.stem(token))
	return stemmedTokens


def incDocFrequencies(tokens, docFrequencies):
	setOfTokens = set(tokens)
	for token in setOfTokens:
		docFrequencies[token] += 1
	return


def mapResponsesToWords(tokens, wordToResponsesMap, responses, responseIndex):
	for token in tokens:
		if not token in wordToResponsesMap.keys():
			wordToResponsesMap[token] = []
		wordToResponsesMap[token].append((responses[responseIndex], 1.0/len(tokens)))
	return


def tfIdfTrain(trainSize):
	docFrequencies = Counter()
	wordToResponsesMap = {}
	prompts, responses = processDialogues(trainSize)
	responseIndex = 0
	for prompt in prompts:
		tokens = tokenize(prompt)
		incDocFrequencies(tokens, docFrequencies)
		mapResponsesToWords(tokens, wordToResponsesMap, responses, responseIndex)
		responseIndex += 1

	numOfDocs = len(prompts)
	invDocFrequencies = Counter()
	for token in docFrequencies.keys():
		invDocFrequencies[token] = log((numOfDocs) / (docFrequencies[token]))

	return wordToResponsesMap, invDocFrequencies

