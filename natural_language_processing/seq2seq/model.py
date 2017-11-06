import tensorflow as tf 

def MakeFancyRNNCell(H, keep_prob=1.0, num_layers=1):
    """Make a fancy RNN cell.
    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.
    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers
    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in xrange(num_layers):
      cell = tf.contrib.rnn.BasicLSTMCell(H, forget_bias=0.0)
      # cell = tf.contrib.rnn.DropoutWrapper(
      #     cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.contrib.rnn.MultiRNNCell(cells)



class Seq2Seq():
	def __init__(self, vocab_size, embed_dim, lstm_state_size):
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim

		self.encoder_input_ = tf.placeholder(tf.int32, [None, None], name="encoder_input")
		self.decoder_input_ = tf.placeholder(tf.int32, [None, None], name="decoder_input")
		self.encoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.encoder_input_, 0)), 1,
            name="encoder_input_lengths")

		with tf.variable_scope ("embeddings", dtype=tf.float32) as scope:
			self.embedding = tf.get_variable(
				"embedding_share", [self.vocab_size, self.embed_dim], tf.float32)
			self.encoder_embedding = tf.nn.embedding_lookup(self.embedding, self.encoder_input_)
			self.decoder_embedding = tf.nn.embedding_lookup(self.embedding, self.decoder_input_)

		with tf.variable_scope ("enc", dtype=tf.float32) as scope:
			self.encoder_cells = MakeFancyRNNCell(lstm_state_size)
			self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
				self.encoder_cells,
				self.encoder_embedding,
				dtype=tf.float32,
				time_major=False)

		with tf.variable_scope ("dec", dtype=tf.float32) as scope:
			self.decoder_cells = MakeFancyRNNCell(lstm_state_size)
			self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
				self.decoder_cells, 
				self.decoder_embedding,
				initial_state=self.encoder_final_state,
				dtype=tf.float32, 
				time_major=False)

			self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, vocab_size)
			self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
		

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(self.decoder_input_, depth=vocab_size, dtype=tf.float32),
			logits=self.decoder_logits))
		train_op = tf.train.AdamOptimizer().minimize(self.loss)
		



