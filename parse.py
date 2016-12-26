import numpy as np
import pickle
def load_question_answer(opts):
	load_data = np.load('./cocoqa/train.npy', encoding='bytes')
	training_data = []
	for data_id in range(len(load_data[0])):
		training_data.append({
			'image_id' : load_data[0][data_id][0][0],
			'question' : np.zeros(55),
			'answer' : load_data[1][data_id][0]
			})
		for question_id in range(55):
			training_data[-1]['question'][question_id] = load_data[0][data_id][question_id+1][0]
	ques_vocab = get_ques_vocab('./cocoqa/vocab-dict.npy')
	ans_vocab = get_answer_vocab('./cocoqa/ansdict.pkl')
	data = {
		'training_data' : training_data,
		'max_question_length' : 55,
		'answer_vocab' : ans_vocab, 
		'question_vocab' : ques_vocab
		}
	return data
def get_ques_vocab(data_dir):
	load_ques_data = np.load(data_dir, encoding='bytes')
	#print load_ques_data[1][0]	
	return load_ques_data[1]

def get_answer_vocab(data_dir):
	load_answer_data = pickle.load(open(data_dir,'rb'))
	#print load_answer_data['skis']
	return load_answer_data
######################### usage #################################
model_options = {
	'num_lstm_layers' : 1
}
# load_question_answer(model_options)
		
