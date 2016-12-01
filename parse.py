import numpy as np
def load_question_answer(opts):
	load_data = np.load('./cocoqa/train.npy')
	training_data = []
	for data_id in range(len(load_data[0])):
		training_data.append({
			'image_id' : load_data[0][data_id][0][0],
			'question' : np.zeros(56),
			'answer' : load_data[1][data_id][0]
			})
		for question_id in range(55):
			training_data[-1]['question'][question_id] = load_data[0][data_id][question_id+1][0]
	print 'Training Data',len(training_data)
	data = {
		'training_data' : training_data,
		'max_question_length' : 55
		#'answer_vocab' : 
		#'question_vocab' : 
		}
	


	return data
#def get_ques_vocab(data_dir):
#abc	




#def get_ans_vocab(data_dir):




#TODO: parse qdict.pkl and ansdict.pkl
#print val_data[0]
#j[0][0~70854]  #imageID, wordID
#j[1][0~70854]  #ansID
