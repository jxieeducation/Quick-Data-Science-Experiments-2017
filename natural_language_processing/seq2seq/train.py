import data_loader 
from model import Seq2Seq
import numpy as np
import tensorflow as tf

vocab = data_loader.load_vocab("vocab")
train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()

seq2seq = Seq2Seq(vocab_size=len(vocab), embed_dim=100, lstm_state_size=100)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

pred = sess.run(seq2seq.decoder_prediction,
	feed_dict={
		seq2seq.encoder_input_: train_X[1:10,:],
		seq2seq.decoder_input_: train_y[1:10,:]
	})




