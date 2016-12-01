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

	model_options = {
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
	
	
	
	lstm_model = model.Model(model_options)
	input_tensors, t_loss, t_accuracy, t_p = lstm_model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()





if __name__ == '__main__':
	main()
