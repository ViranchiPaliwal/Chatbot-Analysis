import tensorflow as tf
import numpy as num
import sys


class Seq2Seq(object):

# constructor of sequence model
    def __init__(self, q_len, a_len,
            qvocab_size, avocab_size,
            embedding, rnn_layers, output_path):
        self.q_len = q_len
        self.a_len = a_len
        self.output_path = output_path
        self.epochs = 40000
        self.model = "seqtoseqLSTM"
        self.learning_rate = 0.0001
        self.embedding_dim = 1024


        # initializing architecture of the model
        def __graph__():
            tf.reset_default_graph()

            self.encoder_input = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(q_len) ]

            self.labels = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(a_len) ]

            self.decoder_input = [ tf.zeros_like(self.encoder_input[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]

            self.output_keep = tf.placeholder(tf.float32)

            # Dropout Wrapper for preventing overfitting
            lstm_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(embedding, state_is_tuple=True),
                    output_keep_prob=self.output_keep)

            # Multi layering to make lstm shallower
            multilayer_rnn = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_cell]*rnn_layers, state_is_tuple=True)

            with tf.variable_scope('decoder') as scope:
                # sequence to sequence model
                self.decoder_outputs, self.decoder_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoder_input,self.decoder_input, multilayer_rnn,
                                                    qvocab_size, avocab_size, embedding)
                scope.reuse_variables()
                self.decoder_outputs_testing, self.decoder_states_testing = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.encoder_input, self.decoder_input, multilayer_rnn, qvocab_size, avocab_size,embedding,
                    feed_previous=True)

            loss_in_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decoder_outputs, self.labels, loss_in_weights, avocab_size)

            # adamoptimizer for weight loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        __graph__()

# Starting training the model and validating while saving sessions
    def train(self, training_batch, validation_batch, session=None ):

        saver = tf.train.Saver()
        if session == None:
            session = tf.Session()
            session.run(tf.global_variables_initializer())

        for i in range(self.epochs):
                self.batch_training(session, training_batch)
                print('number of iteration completed : {}'.format(i))
                if i!=0 and  i % 10000  == 0:
                    saver.save(session, self.output_path + self.model + '.ckpt', global_step=i)
                    validation_loss = self.loss_evaluation(session, validation_batch, 32)
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('Model loss : {0:.3f}'.format(validation_loss))
                    sys.stdout.flush()

# To start from last saved session or for testing the code
    def last_saved_session(self):
        saver = tf.train.Saver()
        session = tf.Session()
        last = tf.train.get_checkpoint_state(self.output_path)
        if last and last.model_checkpoint_path:
            saver.restore(session, last.model_checkpoint_path)
        return session

# decoder ouput generator
    def decoder_output_generator(self, sess, question):
        feed_dictionary = {self.encoder_input[t]: question[t] for t in range(self.q_len)}
        feed_dictionary[self.output_prob] = 1.
        decoder_output = sess.run(self.decoder_outputs_testing, feed_dictionary)
        output = num.array(decoder_output).transpose([1,0,2])
        return num.argmax(output, axis=2)

# feed dictionary updation
    def feed_dictionary_update(self, ques, ans, output_keep):
        feed_dict = {self.encoder_input[t]: ques[t] for t in range(self.q_len)}
        feed_dict.update({self.labels[t]: ans[t] for t in range(self.a_len)})
        feed_dict[self.output_keep] = output_keep
        return feed_dict

# batch training in epoch
    def batch_training(self, session, batch_generator):
        qbatch, abatch = batch_generator.__next__()
        feed_dictionary = self.feed_dictionary_update(qbatch, abatch, output_keep=0.5)
        _, loss = session.run([self.train_op, self.loss], feed_dictionary)
        return loss

# to get output for validation or testing
    def model_performance(self, session, batch_generator):
        qbatch, abatch = batch_generator.__next__()
        feed_dictionary = self.feed_dictionary_update(qbatch, abatch, output_keep=1)
        validation_loss, decoder_output = session.run([self.loss, self.decoder_outputs_testing], feed_dictionary)
        ouput = num.array(decoder_output).transpose([1,0,2])
        return validation_loss, ouput, qbatch, abatch

# loss calculation
    def loss_evaluation(self, session, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            validation_loss, decoder_output, qbatch, abatchY = self.model_performance(session, eval_batch_gen)
            losses.append(validation_loss)
        return num.mean(losses)




