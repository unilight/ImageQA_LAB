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
	image_f = tf.placeholder('float32',[ None, modOpts['img_feature_length'] ], name = 'img')
	question = tf.placeholder('float32',[None,modOpts['lstm_steps'] - 1], name = "question")
	answer = tf.placeholder('float32', [None, modOpts['ans_vocab_size']], name = "answer")

	# word embedding
	word_embedding = tf.Variable(tf.random_uniform([modOpts['q_vocab_size'] + 1, options['embedding_size']],
		-1.0, 1.0), name = 'Wemb')
	

	# Define weights and biases
	weights = {
	# (4096, 512)
	'img_in': tf.Variable(tf.random_normal([modOpts['img_feature_length'], modOpts['embedding_size']])),
	# (128, 10)
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
	}

	biases = {
	# (512, )
	'img_in': tf.Variable(tf.constant(0.1, shape=[modOpts['embedding_size'], ])),

	# (128, )
	# 'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
	
	# (431, )
	'out': tf.Variable(tf.constant(0.1, shape=[modOpts['ans_vocab_size'], ]))
	}

	lstm_input = []
	image_emb = tf.matmul(image_f, weights['img_in']) + biases['img_in']
	image_emb = tf.nn.tanh(image_embed)
	image_emb = tf.nn.dropout(image_embed, modOpts['image_dropout'], name = "vis")
	lstm_input.append(image_emb)

	for i in range(modOpts['lstm_steps']-1):
		word_emb = tf.nn.embedding_lookup(word_embedding, question[:,i])
		word_emb = tf.nn.dropout(word_emb, modOpts['word_emb_dropout'], name = "word_emb" + str(i))
		lstm_input.append(word_emb)

	def LSTM(opts, weights, biases):

		# input to cell
		###########################

		

		# cell
		###########################

		# output as the final results
		###########################
		results = tf.matmul(final_state[1], weights['out']) + biases['out']

		return results

	pred = LSTM(modOpts, weights, biases)

	lstm_model = model.Model(modOpts)
	input_tensors, t_loss, t_accuracy, t_p = lstm_model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

if __name__ == '__main__':
	main()
