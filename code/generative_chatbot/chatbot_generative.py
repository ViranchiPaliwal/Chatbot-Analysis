import tensorflow as tf
import numpy as num


import data_set.data as data
import data_utils
import seq2seq_wrapper


def main():
    metadata, q_indexed, a_indexed = data.load_data()
    (qtraindata, atraindata), (qvaliddata, avaliddata), (qtestdata, atestdata),  = data_utils.divide_processed_data(q_indexed, a_indexed)
    embedding = 1024
    sentencesize = 25
    sizeofbatch = 32
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size

# when using floyd hub for data training hence setting path as /output/
# presently setting for inside folder
    model = seq2seq_wrapper.Seq2Seq(q_len=sentencesize,
                                   a_len=sentencesize,
                                   qvocab_size=xvocab_size,
                                   avocab_size=yvocab_size,
                                   output_path = 'saved_generative_model/',
                                   embedding=embedding,
                                   rnn_layers=3
                                   )

    training_batch = data_utils.yield_batch(qtraindata, atraindata, sizeofbatch)
    validation_batch = data_utils.yield_batch(qvaliddata, avaliddata, sizeofbatch)
    model.train(training_batch, validation_batch)

if __name__ == "__main__":
    main()
