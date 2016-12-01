import numpy as np
def load_question_answer(opts)
	j = np.load('train.npy')
	#print len(j[0])
	training_data = []
	for i in range(len(j[0])):
		training_data.append({
			'image_id' : j[0][i][0][0],
			'question' : np.zeros(56),
			'answer' : j[1][i][0]
			})
		for k in range(55):
			training_data[-1]['question'][k] = j[0][i][k+1][0]
	print 'Training Data',len(training_data)

	data = {
		'training_data' : training_data,
		'max_question_length' : 55
		'answer_vocab' : 
		'question_vocab' : 
	}
	


	return data
def get_ques_vocab(data_dir)
	




def get_ans_vocab(data_dir)




#TODO: parse qdict.pkl and ansdict.pkl
#print val_data[0]
#j[0][0~70854]  #imageID, wordID
#j[1][0~70854]  #ansID
