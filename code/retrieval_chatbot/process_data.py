from collections import defaultdict, Set
import random
import csv

def processDialogues(trainSize = 0.7, testSize = 0.15, validSize = 0.15):
    """
    Creates 3 files train.csv, test.csv and valid.csv for training, testing and validation purposes respectively.
    """
    movieLines = getMovieLines()
    conversations = getConversations()

    allPrompts = []
    allResponses = []
    for conv in conversations:
        if (len(conv) % 2 != 0):
            conv = conv[0: len(conv) - 1]
        for i in range(len(conv)):
            if (i % 2 == 0):
                allPrompts.append(movieLines[conv[i]].lower())
            else:
                allResponses.append(movieLines[conv[i]].lower())

    lenAllPrompts = len(allPrompts)
    lenAllResponses = lenAllPrompts

    trainPrompts = allPrompts[0: int(trainSize*lenAllPrompts)]
    trainResponses = allResponses[0: int(trainSize*lenAllResponses)]

    testPrompts = allPrompts[int(trainSize*lenAllPrompts) : int(trainSize*lenAllPrompts) + int(testSize*lenAllPrompts)]
    testResponses = allResponses[int(trainSize * lenAllResponses): int(trainSize * lenAllResponses)
                                                                   + int(testSize * lenAllResponses)]

    validPrompts = allPrompts[int(trainSize*lenAllPrompts) + int(testSize*lenAllPrompts) : ]
    validResponses = allResponses[int(trainSize * lenAllResponses) + int(testSize * lenAllResponses):]

    writeTrainFile(trainPrompts, trainResponses, allResponses)
    writeTestFile(testPrompts, testResponses, allResponses)
    writeValidationFile(validPrompts, validResponses, allResponses)


def getMovieLines():
    '''
    Returns a dict movieLines containing (lineId line) key value pairs from file movie_lines.txt
    '''
    movieLines = {}
    lines = open("../Project/cornell movie-dialogs corpus/movie_lines.txt", encoding='utf-8',
                 errors='ignore').read().split('\n')

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
    lines = open("../Project/cornell movie-dialogs corpus/movie_conversations.txt", encoding='utf-8',
                 errors='ignore').read().split('\n')

    for line in lines:
        lineWithMetadata = line.split(' +++$+++ ')
        conv = lineWithMetadata[len(lineWithMetadata) - 1]
        conv = conv.replace("'", "").replace(" ", "").replace("[", "").replace("]", "")
        conversations.append(conv.split(","))

    return conversations


def writeTrainFile(trainPrompts, trainResponses, allResponses):
    with open('train.csv', 'w') as csvfile:
        fieldnames = ['Context', 'Utterance', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(trainPrompts)):
            rand = random.random()
            if rand <= 0.5:
                randResp = trainResponses[i]
                while (randResp == trainResponses[i]):
                    randResp = random.choice(allResponses)
                writer.writerow({'Context' : trainPrompts[i], 'Utterance' : randResp, 'Label' : 0})
            else:
                writer.writerow({'Context': trainPrompts[i], 'Utterance': trainResponses[i], 'Label': 1})


def writeTestORValidationFile(prompts, responses, allResponses, type):
    with open(type+'.csv', 'w') as csvfile:
        fieldnames = ['Context', 'Ground Truth Utterance', 'Distractor_0',
                      'Distractor_1', 'Distractor_2', 'Distractor_3', 'Distractor_4', 'Distractor_5',
                      'Distractor_6', 'Distractor_7', 'Distractor_8']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        allUniqueResponses = list(set(allResponses))
        for i in range(len(prompts)):
            distractors = random.sample(allResponses, 9)
            while responses[i] in distractors:
                distractors = random.sample(allResponses, 9)

            writer.writerow({'Context' : prompts[i],
                             'Ground Truth Utterance' : responses[i],
                             'Distractor_0': distractors[0],
                             'Distractor_1': distractors[1],
                             'Distractor_2': distractors[2],
                             'Distractor_3': distractors[3],
                             'Distractor_4': distractors[4],
                             'Distractor_5': distractors[5],
                             'Distractor_6': distractors[6],
                             'Distractor_7': distractors[7],
                             'Distractor_8': distractors[8]})


def writeValidationFile(validPrompts, validResponses, allResponses):
    writeTestORValidationFile(validPrompts, validResponses, allResponses, "valid")

def writeTestFile(testPrompts, testResponses, allResponses):
    writeTestORValidationFile(testPrompts, testResponses, allResponses, "test")


if __name__ == "__main__":
    processDialogues()
