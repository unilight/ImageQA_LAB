import tensorflow as tf
import model
import data_loader
import argparse
import numpy as np
import pickle
import parse
from imp import reload
from gensim.models import word2vec

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='num_lstm_layers')
    parser.add_argument('--img_feature_length', type=int, default=4096, help='img_feature_length')
    parser.add_argument('--rnn_size', type=int, default=512, help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5, help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5, help='image_dropout')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=51, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100, help='Expochs')
    parser.add_argument('--debug', type=bool, default=True, help='Debug')
    parser.add_argument('--resume_model', type=str, default=None,help='Trained Model Path')

    f = open('result.txt', 'w')

    args = parser.parse_args()
    if args.data_dir is None:
        print('Specify data directory path!')
        exit()

    print("Reading QA DATA")
    qa_data = parse.load_question_answer(args)
    #print("QA_data", len(qa_data))
    print("Reading Image features")
    image_feat = data_loader.load_image_feat(args.data_dir)
    print("Image features", image_feat.shape)

    ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

    load_data = pickle.load(open('./cocoqa/newdic', 'rb'))
    sentences = word2vec.Text8Corpus('./text8')
    model = word2vec.Word2Vec(sentences, size = 512, min_count = 2, workers = 4)

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
    #sentences = word2vec.Text8Corpus('./text8')
    #model = word2vec.Word2Vec(sentences, size = 512, min_count = 2, workers = 4)
    #load_data = np.load(open('./cocoqa/vocab-dict.npy','r'))
    #print("Hello!\n")
    # tf graph input
    # (batch_size, 4096)
    lstm_image_feat = tf.placeholder(tf.float32, [modOpts['batch_size'], modOpts['img_feature_length']])
    # (batch_size, 55)
    #lstm_q_idx = tf.placeholder(tf.float32, [modOpts['batch_size'], modOpts['lstm_steps']-1])
    lstm_q_vec = tf.placeholder(tf.float32, [modOpts['batch_size'], modOpts['lstm_steps']-1, 512])
    # (batch_size, 431)
    lstm_answer = tf.placeholder(tf.float32, [modOpts['batch_size'], modOpts['ans_vocab_size']], name = "answer")

    # Define weights and biases
    weights = {
    # (9738, 512)
    'word_emb': tf.Variable(tf.random_uniform([modOpts['q_vocab_size'], modOpts['embedding_size']],
        -1.0, 1.0), name = 'Wemb'),
    # (4096, 512)
    'img_emb': tf.Variable(tf.random_normal([modOpts['img_feature_length'], modOpts['embedding_size']])),
    # (512, 512)
    'input_hidden': tf.Variable(tf.random_normal([modOpts['embedding_size'], modOpts['rnn_size']])),
    # (512, 431)
    'hidden_output': tf.Variable(tf.random_normal([modOpts['rnn_size'], modOpts['ans_vocab_size']]))
    }

    biases = {
    # (512, )
    'img_emb': tf.Variable(tf.zeros([modOpts['embedding_size']])),

    # (512, )
    'input_hidden': tf.Variable(tf.zeros([modOpts['rnn_size']])),
    
    # (431, )
    'hidden_output': tf.Variable(tf.zeros([modOpts['ans_vocab_size']]))
    }

    def LSTM(opts, img_f, q_vec, weights, biases):
        # X : (batch_size, 56, 512)

        # 2D tensor with shape (batch_size, 1, 512)     
        image_emb = tf.matmul(img_f, weights['img_emb']) + biases['img_emb']
        image_emb = tf.nn.tanh(image_emb)
        image_emb = tf.nn.dropout(image_emb, opts['image_dropout'], name = "vis")
        image_emb = tf.reshape(image_emb, [-1, 1, 512])
        #print image_emb

        #q_idx = tf.to_int32(q_idx)
        # 1D tensor with shape (batch_size * 55)
        #flat_q_idx = tf.reshape(q_idx, [-1])
        #print flat_q_idx
        # 2D tensor with shape (batch_size * 55, 512)
        #flat_q_emb = tf.nn.embedding_lookup(weights['word_emb'], flat_q_idx)
        #flat_q_emb = tf.nn.dropout(flat_q_emb, opts['word_emb_dropout'])
        # 3D tensor with shape (batch_size, 55, 512)
        q_emb = tf.reshape(q_vec, [-1, 10, 512])
        #print q_emb

        X = tf.concat(1, [image_emb, q_emb], name = "finalX")
        #print X

        # input to cell
        ###########################
        # transpose the inputs shape from
        # X ==> (batch_size * 56 steps, 512 inputs)
        X = tf.reshape(X, [-1, opts['embedding_size']])

        # into hidden
        # X_in = (batch_size * 56 steps, 512 hidden)
        X_in = tf.matmul(X, weights['input_hidden']) + biases['input_hidden']
        # X_in ==> (batch_size, 56 steps, 512 hidden)
        X_in = tf.reshape(X_in, [-1, opts['lstm_steps'], opts['rnn_size']])

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
    logits = LSTM(modOpts, lstm_image_feat, lstm_q_vec, weights, biases)    
    #logits = LSTM(modOpts, lstm_image_feat, lstm_q_idx,  weights, biases)
    answer_probab = tf.nn.softmax(logits)
    predictions = tf.argmax(answer_probab,1)

    ce = tf.nn.softmax_cross_entropy_with_logits(logits, lstm_answer)
    correct_predictions = tf.equal(tf.argmax(logits,1), tf.argmax(lstm_answer,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    loss = tf.reduce_sum(ce)
    
    #train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    
    #sess = tf.InteractiveSession('''config=tf.ConfigProto(gpu_options=gpu_options)''')
    sess = tf.InteractiveSession()
    #sess = tf.Session()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    #if args.resume_model:
    #   saver.restore(sess, args.resume_model)
    for i in range(args.epochs):
        batch_no = 0

        while (batch_no*modOpts['batch_size']) < len(qa_data['training_data']):
            img_f, q_vec, answer = get_training_batch(batch_no, modOpts, image_feat, qa_data, load_data, model)
            _, loss_value, acc, pred = sess.run([train_op, loss, accuracy, predictions], feed_dict={
                    lstm_image_feat:img_f,
                    lstm_q_vec:q_vec,
                    lstm_answer:answer
                }
            )
            batch_no += 1
            if args.debug:
                for idx, p in enumerate(pred):
                    print(p, np.argmax(answer[idx]))
                print("Loss", loss_value, batch_no, i)
                print("Accuracy", acc)
                print("---------------")
                f.write(' '.join( ("Loss", str(loss_value), str(batch_no), str(i), '\n' ) ) )
                f.write(' '.join( ("Accuracy", str(acc), '\n') ) )
                f.write("---------------\n")
            else:
                print("Loss", loss_value, batch_no, i)
                print("Training Accuracy", acc)

        save_path = saver.save(sess, "./model/model{}.ckpt".format(i))
    f.close()


def get_training_batch(batch_no, opts, image_feat, qa_data, load_data, model):
    qa = qa_data['training_data']

    si = (batch_no * opts['batch_size'])%len(qa)
    ei = min(len(qa), si + opts['batch_size'])
    n = ei - si

    #q_idx = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
    q_vector = np.ndarray((n,qa_data['max_question_length'],512))
    answer = np.zeros( (n, len(qa_data['answer_vocab'])))
    img_f = np.ndarray( (n, 4096) )

    count = 0
    for i in range(si, ei):

        answer[count, qa[i]['answer']] = 1.0
        img_f[count, :] = image_feat[ qa[i]['image_id'] ].toarray()[0].reshape((1, 4096))
        for j in range(qa_data['max_question_length']):
            index = int(qa[i]['question'][j])
            if index == 0:
                index = 9738
                q_vector[count,j,:] = [0] * 512
                continue
            final_qstr = load_data[index-1].strip("'")
            try:
                q_vector[count,j,:] = model[final_qstr]
            except KeyError:
                q_vector[count,j,:] = [0] * 512 
        #q_idx[count, :] = qa[i]['question'][:]
        count += 1

    #return img_f, q_idx, answer
    return img_f, q_vector, answer


if __name__ == '__main__':
    main()
