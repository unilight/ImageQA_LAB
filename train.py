import tensorflow as tf
import model
import data_loader
import argparse
import numpy as np
import parse

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
	qa_data = parse.load_question_answer(args)

	print "Reading Image features"
	image_feat = data_loader.load_image_feat(args.data_dir)
	print "Image features", image_feat.shape

	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

	modOpts = {
		# batch_size should not be in here
		'batch_size' : args.batch_size,
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
	'img_emb': tf.Variable(tf.random_normal([modOpts['img_feature_length'], modOpts['embedding_size']])),
	# (512, 512)
	'input_hidden': tf.Variable(tf.random_normal([modOpts['embedding_size'], modOpts['rnn_size']])),
	# (512, 431)
	'hidden_output': tf.Variable(tf.random_normal([modOpts['rnn_size'], modOpts['ans_vocab_size']]))
	}

	biases = {
	# (512, )
	'img_emb': tf.Variable(tf.constant(0.1, shape=[modOpts['embedding_size'], ])),

	# (512, )
	'input_hidden': tf.Variable(tf.constant(0.1, shape=[modOpts['rnn_size'], ])),
	
	# (431, )
	'hidden_output': tf.Variable(tf.constant(0.1, shape=[modOpts['ans_vocab_size'], ]))
	}

	def LSTM(opts, X, weights, biases):

		# input to cell
		###########################
		# transpose the inputs shape from
		# X ==> (batch_size * 56 steps, 512 inputs)
		X = tf.reshape(X, [-1, opts['embedding_size']])

		# into hidden
		# X_in = (batch_size * 56 steps, 512 hidden)
		X_in = tf.matmul(X, weights['input_hidden']) + biases['input_hidden']
		# X_in ==> (batch_size, 56 steps, 512 hidden)
		X_in = tf.reshape(X_in, [-1, opts['lstm_steps']-1, opts['rnn_size']])

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

	logits = LSTM(modOpts, lstm_input, weights, biases)	
	
	answer_probab = tf.nn.softmax(logits)
	predictions = tf.argmax(answer_probab,1)
	
	ce = tf.nn.softmax_cross_entropy_with_logits(logits, lstm_answer)
	correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(lstm_answer,1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
	loss = tf.reduce_sum(ce)
	
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	tf.initialize_all_variables().run()

	for i in xrange(args.epochs):
		batch_no = 0

		while (batch_no*modOpts['batch_size']) < len(qa_data['training_data']):
			img_q, answer = get_training_batch(batch_no, modOpts, image_feat, qa_data, weights, biases, word_embedding)
			sess.run([train_op], feed_dict={
					lstm_input:img_q,
					lstm_answer:answer
				}
			)
			batch_no += 1
			if args.debug:
				for idx, p in enumerate(pred):
					print ans_map[p], ans_map[ np.argmax(answer[idx])]

				print "Loss", loss, batch_no, i
				print "Accuracy", accuracy
				print "---------------"
			else:
				print "Loss", loss, batch_no, i
				print "Training Accuracy", accuracy


def get_training_batch(batch_no, opts, image_feat, qa_data, weights, biases, word_embedding):
	qa = qa_data['training_data']

	si = (batch_no * opts['batch_size'])%len(qa)
	ei = min(len(qa), si + opts['batch_size'])
	n = ei - si
	img_q = np.ndarray( (n, qa_data['max_question_length']+1, opts['rnn_size']), dtype = 'float32')
	sentence = np.ndarray( (qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	img = np.ndarray( (4096) )

	count = 0
	for i in range(si, ei):
		answer[count, qa[i]['answer']] = 1.0
		img = image_feat[ qa[i]['image_id'] ].toarray()[0]
		img = img.reshape((1, 4096))
		image_emb = tf.matmul(img, weights['img_emb']) + biases['img_emb']
		image_emb = tf.nn.tanh(image_emb)
		image_emb = tf.nn.dropout(image_emb, opts['image_dropout'], name = "vis")
		img_q[count, 0, :] = image_emb.eval()

		sentence = qa[i]['question'][:]
		for i in range(55):
			word_emb = tf.nn.embedding_lookup(word_embedding, int(sentence[i]))
			word_emb = tf.nn.dropout(word_emb, opts['word_emb_dropout'], name = "word_emb" + str(i))
			img_q[count, i+1, :] = word_emb.eval()

		count += 1

	return img_q, answer


if __name__ == '__main__':
	main()
