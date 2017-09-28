"""
Processes the cornell movie dialogues corpus data and create two lists 'prompts' and 'responses'
where prompts is a list of message prompts and responses is a list of corresponding responses.
"""

from collections import defaultdict
import random

def processDialogues(trainSize):
	"""
	Returns a tuple (prompts, responses) where each is a list of length trainSize and where
	prompts is a list of message prompts and responses is a list of corresponding responses.
	"""
	movieLines = getMovieLines()
	conversations = getConversations()

	allPrompts = []
	allResponses = []
	for conv in conversations:
		if(len(conv) % 2 != 0):
			conv = conv[0 : len(conv) - 1]
		for i in range(len(conv)):
			if(i % 2 == 0):
				allPrompts.append(movieLines[conv[i]])
			else:
				allResponses.append(movieLines[conv[i]])

	trainingIndexes = random.sample([i for i in range(len(allPrompts))], trainSize)

	prompts = []
	responses = []
	for i in trainingIndexes:
		prompts.append(allPrompts[i])
		responses.append(allResponses[i])

	return prompts, responses




def getMovieLines():
	'''
	Returns a dict movieLines containing (lineId line) key value pairs from file movie_lines.txt
	'''
	movieLines = {}
	lines = open("cornell movie-dialogs corpus/movie_lines.txt", encoding = 'utf-8', 
		errors = 'ignore').read().split('\n')
		    
	for line in lines:
		line = line.split(' +++$+++ ')
		if (len(line) == 5):
			lineId = line[0]
			movieLines[lineId] = line[4]

	return movieLines


def getConversations():
	"""
	Returns a list of (list of lineIds) from file movie_conversations.txt where each list of lineIds represents a 
	conversation.
	"""

	conversations = []
	lines = open("cornell movie-dialogs corpus/movie_conversations.txt", encoding = 'utf-8', 
		errors = 'ignore').read().split('\n')

	for line in lines:
		lineWithMetadata = line.split(' +++$+++ ')
		conv  = lineWithMetadata[len(lineWithMetadata) - 1]
		conv = conv.replace("'", "").replace(" ", "").replace("[", "").replace("]", "")
		conversations.append(conv.split(","))

	return conversations



