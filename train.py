import tensorflow as tf
import model
import data_loader
import argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_lstm_layers', type=int, default=2, help='num_lstm_layers')
	parser.add_argument('--img_feature_length', type=int, default=4096, help='img_feature_length')
	parser.add_argument('--rnn_size', type=int, default=512, help='rnn_size')
	parser.add_argument('--embedding_size', type=int, default=512, help='embedding_size'),
	parser.add_argument('--word_emb_dropout', type=float, default=0.5, help='word_emb_dropout')
	parser.add_argument('--image_dropout', type=float, default=0.5, help='image_dropout')
	parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
	parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Batch Size')
	parser.add_argument('--epochs', type=int, default=200, help='Expochs')
	parser.add_argument('--debug', type=bool, default=False, help='Debug')
	parser.add_argument('--resume_model', type=str, default=None,help='Trained Model Path')

	args = parser.parse_args()
	if args.data_dir is None:
		print 'Specify data directory path!'
		exit()

	print "Reading QA DATA"
	# TODO : load QA
	qa_data = {}

	print "Reading Image features"
	image_feat = data_loader.load_image_feat(args.data_dir)
	print "Image features", image_feat.shape

	modOpts = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'img_feature_length' : args.img_feature_length,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}
	
	# tf graph input
	# (batch_size, 56, 512)
	lstm_input = tf.placeholder(tf.float32, [None, modOpts['lstm_steps']-1, modOpts['embedding_size']])
	# (batch_size, 431)
	lstm_answer = tf.placeholder(tf.float32, [None, modOpts['ans_vocab_size']], name = "answer")

	# word embedding
	# (9738, 512)
	word_embedding = tf.Variable(tf.random_uniform([modOpts['q_vocab_size'], modOpts['embedding_size']],
		-1.0, 1.0), name = 'Wemb')
	

	# Define weights and biases
	weights = {
	# (4096, 512)
	'img_in': tf.Variable(tf.random_normal([modOpts['img_feature_length'], modOpts['embedding_size']])),
	# (512, 512)
	'input_hidden': tf.Variable(tf.random_normal([modOpts['embedding_size'], modOpts['rnn_size']])),
	# (512, 431)
	'hidden_output': tf.Variable(tf.random_normal([modOpts['rnn_size'], modOpts['ans_vocab_size']])
	}

	biases = {
	# (512, )
	'img_in': tf.Variable(tf.constant(0.1, shape=[modOpts['embedding_size'], ])),

	# (512, )
	'input_hidden': tf.Variable(tf.constant(0.1, shape=[modOpts['rnn_size'], ])),
	
	# (431, )
	'out': tf.Variable(tf.constant(0.1, shape=[modOpts['ans_vocab_size'], ]))
	}

	'''
	#TODO : these should be processed somewhere else
	lstm_input = []
	image_emb = tf.matmul(image_f, weights['img_in']) + biases['img_in']
	image_emb = tf.nn.tanh(image_embed)
	image_emb = tf.nn.dropout(image_embed, modOpts['image_dropout'], name = "vis")
	lstm_input.append(image_emb)

	for i in range(modOpts['lstm_steps']-1):
		word_emb = tf.nn.embedding_lookup(word_embedding, question[:,i])
		word_emb = tf.nn.dropout(word_emb, modOpts['word_emb_dropout'], name = "word_emb" + str(i))
		lstm_input.append(word_emb)
	'''

	def LSTM(opts, X, weights, biases):

		# input to cell
		###########################
		# transpose the inputs shape from
		# X ==> (batch_size * 56 steps, 512 inputs)
		X = tf.reshape(X, [-1, opts['embedding_size']])

		# into hidden
		# X_in = (batch_size * 56 steps, 512 hidden)
		X_in = tf.matmul(X, weights['input_hidden']) + biases['input_hiddwn']
		# X_in ==> (batch_size, 56 steps, 512 hidden)
		X_in = tf.reshape(X_in, [-1, opts['lstn_steps']-1, opts['rnn_size']])

		# cell
		###########################
		# basic LSTM Cell.
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(opts['rnn_size'], forget_bias=1.0, state_is_tuple=True)
		# lstm cell is divided into two parts (c_state, h_state)
		init_state = lstm_cell.zero_state(opts['batch_size'], dtype=tf.float32)
		outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

		# output as the final results
		###########################
		results = tf.matmul(final_state[1], weights['hidden_output']) + biases['hidden_output']

		return results

	pred = LSTM(modOpts, lstm_input, weights, biases)

	lstm_model = model.Model(modOpts)
	input_tensors, t_loss, t_accuracy, t_p = lstm_model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

if __name__ == '__main__':
	main()
