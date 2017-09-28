import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import csv
from TfIdf_predictor import predict
from Trainer import tfIdfTrain
from data_utils import vectorize


vocab_size = 8000
n_dim = 50
n_epochs = 20000
batch_size = 32
prompt_len = 30
resp_len = prompt_len

rnn_size = 64

prompt = tf.placeholder('int64', [None, prompt_len])
response = tf.placeholder('int64', [None, resp_len])
label = tf.placeholder('float')

print ("here1")
train_prompts = list(np.load('datasets/prompts.npy'))
train_responses = list(np.load('datasets/responses.npy'))
train_labels = list(np.load('datasets/labels.npy'))
vocab = list(datasets/vocab.npy)
train_size = len(train_prompts)
print("here2")

wordToResponsesMap, invDocFrequencies = tfIdfTrain(100000)


def dual_encoder_lstm(prompt, response, n_inputs = batch_size):
    enc_layer = {'weights': tf.Variable(tf.random_normal([rnn_size, rnn_size])),
                  'biases': tf.Variable(tf.random_normal([rnn_size]))}

    word_embeddings = tf.get_variable("word_embeddings",
                                      shape=[vocab_size, n_dim],
                                      initializer=tf.random_uniform_initializer(-0.25, 0.25))

    prompt_embedded = tf.nn.embedding_lookup(word_embeddings, prompt, name="embed_prompt")
    prompt_embedded = tf.transpose(prompt_embedded, [1, 0, 2])
    prompt_embedded = tf.reshape(prompt_embedded, [-1, n_dim])
    prompt_embedded = tf.split(0, prompt_len, prompt_embedded)

    response_embedded = tf.nn.embedding_lookup(word_embeddings, response, name="embed_utterance")
    response_embedded = tf.transpose(response_embedded, [1, 0, 2])
    response_embedded = tf.reshape(response_embedded, [-1, n_dim])
    response_embedded = tf.split(0, resp_len, response_embedded)

    #Build the RNN
    with tf.variable_scope("lstm1") as vs:
        lstm1 = rnn_cell.BasicLSTMCell(rnn_size)
        outputs1, states1 = rnn.rnn(lstm1, prompt_embedded, dtype=tf.float32)

    with tf.variable_scope("lstm2") as vs:
        lstm2 = rnn_cell.BasicLSTMCell(rnn_size)
        outputs2, states2 = rnn.rnn(lstm2, response_embedded, dtype=tf.float32)

    prompt_encodings = states1[-1]
    response_encodings = states2[-1]

    generated_response = tf.matmul(prompt_encodings, enc_layer['weights']) + enc_layer['biases']
    generated_response = tf.expand_dims(generated_response, 2)
    given_response = tf.expand_dims(response_encodings, 2)

    logits = tf.batch_matmul(generated_response, given_response, True)
    logits = tf.squeeze(logits, [2])

    probs = tf.sigmoid(logits)
    return probs, logits


def train_model(prompt, response, label):
    probs, logits = dual_encoder_lstm(prompt, response)
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(label)))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(train_size/batch_size)):
                if epoch and epoch % 10000 == 0:
                    print ("Saved model at iteration ", epoch)
                    saver.save(sess, "/output/model"+str(epoch)+".ckpt")
                epoch_prompts = train_prompts[i*batch_size: (i*batch_size)+batch_size]
                epoch_responses = train_responses[i*batch_size: (i*batch_size)+batch_size]
                epoch_labels = train_labels[i*batch_size: (i*batch_size)+batch_size]

                _, c = sess.run([optimizer, cost], feed_dict={prompt: np.array(epoch_prompts),
                                                              response: np.array(epoch_responses),
                                                              label: epoch_labels})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', n_epochs, 'loss: ', epoch_loss)

        saver.save(sess, "/output/model"+n_epochs+".ckpt")

train_model(prompt, response, label)

def test_model():
    evaluation = dual_encoder_lstm(prompt, response)
    responses = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "model.ckpt")
    with open('train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        test_prompts = []
        for row in reader:
            test_prompts.append(row['Context'])

        for test_prompt in test_prompts:
            prompt_vector = np.expand_dims(np.array(vectorize(test_promptprompt, vocab)), axis=0)
            early_predictions = predict(wordToResponsesMap, invDocFrequencies, test_prompt)
            probs = []
            for prediction in early_predictions:
                pred_vector = np.expand_dims(np.array(vectorize(prediction, vocab)), axis=0)
                probs.append(sess.run(evaluation.eval(feed_dict = {prompt : prompt_vector,
                                                                   response : pred_vector,
                                                                    label : np.array([1])}))[0])
            responses.append(max(probs))
