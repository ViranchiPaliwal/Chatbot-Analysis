from random import sample
import numpy as np

# batch generation of size 32
def yield_batch(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

# divide data set in the ratio 75%, 15% and 15% respectively for
# training, validation and testing
def divide_processed_data(ques, ans):
    proportion = [0.7, 0.15, 0.15]
    part = [ int(len(ques)*item) for item in proportion ]
    trainDataQ, trainDataA = ques[:part[0]], ans[:part[0]]
    validDataQ, validDataA = ques[-part[-1]:], ans[-part[-1]:]
    testDataQ, testDataA = ques[part[0]:part[0] + part[1]], ans[part[0]:part[0] + part[1]]
    return (trainDataQ,trainDataA), (validDataQ,validDataA), (testDataQ,testDataA)