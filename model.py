import tensorflow as tf
import math

class Model:
	def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def random_bias(self, dim, name=None):
		return tf.Variable(tf.zeros([dim]), name=name)

	def __init__(self, options):
		self.options = options

		# Weights
		self.Wword_emb = self.random_weight(options['q_vocab_size'], options['embedding_size'], name = 'WWord_emb')
		self.Wimg_emb = self.random_weight(options['img_feature_length'], options['embedding_size'], name = 'WImg_emb')
		self.Wemb_hidden = self.random_weight(options['embedding_size'], options['rnn_size'], name = 'Wemb_hidden')
		self.Whidden_ans = self.random_weight(options['rnn_size'], options['ans_vocab_size'], name = 'Whidden_ans')

		# Biases
		self.Bimg_emb = self.random_bias(options['embedding_size'], name = 'BImg_emb')
		self.Bemb_hidden = self.random_bias(options['rnn_size'], name = 'BEmb_hidden')
		self.Bhidden_ans = self.random_bias(options['ans_vocab_size'], name = 'BHidden_ans')

	def build_model(self):
		lstm_image_feat = tf.placeholder(tf.float32, [self.options['batch_size'], self.options['img_feature_length']])
		lstm_q_vec = tf.placeholder(tf.float32, [self.options['batch_size'], self.options['lstm_steps']-1, 512])
		lstm_answer = tf.placeholder(tf.float32, [self.options['batch_size'], self.options['ans_vocab_size']], name = "answer")

		# LSTM preprocessing
		image_emb = tf.matmul(lstm_image_feat, self.Wimg_emb) + self.Bimg_emb
		image_emb = tf.nn.tanh(image_emb)
		image_emb = tf.nn.dropout(image_emb, self.options['image_dropout'], name = "Dropout")
		image_emb = tf.reshape(image_emb, [-1, 1, 512])	
		q_emb = tf.reshape(lstm_q_vec, [-1, 10, 512])
		# Concat question to image
		X = tf.concat(1, [image_emb, q_emb], name = "finalX")
		X = tf.reshape(X, [-1, self.options['embedding_size']])

		X_in = tf.matmul(X, self.Wemb_hidden) + self.Bemb_hidden
		X_in = tf.reshape(X_in, [-1, self.options['lstm_steps'], self.options['rnn_size']])

		# LSTM
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.options['rnn_size'], forget_bias=1.0, state_is_tuple=True)
		# lstm cell is divided into two parts (c_state, h_state)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
		init_state = cell.zero_state(self.options['batch_size'], dtype=tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

		logits = tf.matmul(final_state[1][1], self.Whidden_ans) + self.Bhidden_ans
		ce = tf.nn.softmax_cross_entropy_with_logits(logits, lstm_answer)
		answer_probab = tf.nn.softmax(logits)

		predictions = tf.argmax(answer_probab,1) #
		correct_predictions = tf.equal(tf.argmax(logits,1), tf.argmax(lstm_answer,1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) #
		loss = tf.reduce_sum(ce) #
		
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', loss)

		input_tensors = {
			'image' : lstm_image_feat,
			'sentence' : lstm_q_vec,
			'answer' : lstm_answer
		} #
		return input_tensors, loss, accuracy, predictions


