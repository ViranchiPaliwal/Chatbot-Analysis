

from collections import Counter, defaultdict

from Trainer import tfIdfTrain, tokenize


def getCandidateResponsesMap(wordToResponsesMap, invDocFrequencies, tokens):
	candidateResponsesMap = Counter()
	for token in tokens:
		if wordToResponsesMap.get(token):
			for responseTuple in wordToResponsesMap[token]:
				response = responseTuple[0]
				likelyhood = responseTuple[1] * invDocFrequencies[token]
				candidateResponsesMap[response] += likelyhood
	return candidateResponsesMap



def predict(wordToResponsesMap, invDocFrequencies, prompt):
	bestResponse = None
	tokens = tokenize(prompt)
	candidateResponsesMap = getCandidateResponsesMap (wordToResponsesMap, invDocFrequencies, tokens)

	if len(candidateResponsesMap) > 0:
		bestResponses = [(k, candidateResponsesMap[k]) for k in sorted(candidateResponsesMap, key=candidateResponsesMap.get, reverse=True)]
		if len(candidateResponsesMap) > 10:
			return bestResponses[0:9]
		else:
			return bestResponses

	else:
		return [("I dont know",10.0)]





