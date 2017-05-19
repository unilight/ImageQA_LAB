import tensorflow as tf
import model
import argparse
import numpy as np
import pickle
import parse
from imp import reload
from gensim.models import word2vec
import random
#import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='num_lstm_layers')
    parser.add_argument('--img_feature_length', type=int, default=4096, help='img_feature_length')
    parser.add_argument('--rnn_size', type=int, default=512, help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5, help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5, help='image_dropout')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=49, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=65, help='Expochs')
    parser.add_argument('--debug', type=bool, default=True, help='Debug')
    parser.add_argument('--resume_model', type=str, default=None,help='Trained Model Path')
    parser.add_argument('--logdir', type=str, default='./tensorboard/',help='TensorBoard Path')
    parser.add_argument('--gensim_train', type=bool, default=False,help='Train gensim model')
    parser.add_argument('--gensim_model', type=str, default='gensim_model',help='Gensim trained model path')

    f = open('result.txt', 'w')

    args = parser.parse_args()
    if args.data_dir is None:
        print('Specify data directory path!')
        exit()

    print("Reading QA DATA")
    qa_data = parse.load_question_answer(args)
    print("Reading Image features")
    image_feat = parse.load_image_feat(args.data_dir)
    ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

    load_data = pickle.load(open('./cocoqa/newdic', 'rb'))

    # Gensim Word2Vec training
    if(args.gensim_train):
        sentences = word2vec.Text8Corpus('./text8')
        w2v_model = word2vec.Word2Vec(sentences, size = 512, min_count = 2, workers = 4)
        w2v_model.save(args.gensim_model)
    else:
        w2v_model = word2vec.Word2Vec.load(args.gensim_model)

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
    #load_data = np.load(open('./cocoqa/vocab-dict.npy','r')
    
    vis_lstm_model = model.Model(modOpts)
    input_tensors, loss, accuracy, predictions, idxs= vis_lstm_model.build_model()

    train_op = tf.train.RMSPropOptimizer(args.learning_rate).minimize(loss)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        
        #saver
    saver = tf.train.Saver()
        #saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    
    # tensorboard merge
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.logdir, sess.graph)

    tf.initialize_all_variables().run()

    #saver = tf.train.Saver()
    #if args.resume_model:
    #   saver.restore(sess, args.resume_model)
    #plot_acc = np.zeros(args.epochs)
    #plot_epoch = np.zeros(args.epochs)
    #1446
    indices = np.arange(1446)
    np.random.shuffle(indices)
    for i in range(args.epochs):
        print('Now epoch:',i,'\n')
        batch_no = 0
        LOSS = 0.0
        VAL_LOSS = 0.0
        ACC = 0.0
        VAL_ACC = 0.0
        while (batch_no*modOpts['batch_size'] < 1300*modOpts['batch_size']):
            img_f, q_vec, answer = get_training_batch(indices[batch_no], modOpts, image_feat, qa_data, load_data, w2v_model)
            _, loss_value, acc, pred, indexes, summary = sess.run([train_op, loss, accuracy, predictions, idxs, merged], feed_dict={
                input_tensors['image']:img_f,
                input_tensors['sentence']:q_vec,
                input_tensors['answer']:answer
            })
            batch_no += 1
            ACC += acc
            LOSS += loss_value
            '''
            if args.debug:
                for idx, p in enumerate(pred):
                    writer.add_summary(summary, i+batch_no*0.001)
                    print(p, np.argmax(answer[idx]), indexes[idx])
                print("Loss", loss_value, batch_no, i)
                print("Accuracy", acc)
                print("---------------")
            else:
                print("Loss", loss_value, batch_no, i)
                print("Training Accuracy", acc)
            '''
        #save_path = saver.save(sess, "Data/Models/model{}.ckpt".format(i))
        f.write(' '.join( ("Loss", str(LOSS/1300), str(i), '\n' ) ) )
        f.write(' '.join( ("Accuracy", str(ACC/1300), '\n') ) )
        f.write("---------------\n")
        while (batch_no*modOpts['batch_size'] < len(qa_data['training_data'])):
            img_f, q_vec, answer = get_training_batch(indices[batch_no], modOpts, image_feat, qa_data, load_data, w2v_model)
            _, loss_value, acc, pred, indexes, summary = sess.run([train_op, loss, accuracy, predictions, idxs, merged], feed_dict={
                input_tensors['image']:img_f,
                input_tensors['sentence']:q_vec,
                input_tensors['answer']:answer
            })
            batch_no += 1
            VAL_ACC += acc
            VAL_LOSS += loss_value
            '''
            if args.debug:
                for idx, p in enumerate(pred):
                    writer.add_summary(summary, i+batch_no*0.001)
                    print(p, np.argmax(answer[idx]), indexes[idx])
                print("Loss", loss_value, batch_no, i)
                print("Accuracy", acc)
                print("---------------")
            else:
                print("Loss", loss_value, batch_no, i)
                print("Training Accuracy", acc)
            '''
        print('Loss:',VAL_LOSS/146, ' epoch: ',i)
        print("Accuracy", VAL_ACC/146)
        print('-----------------------\n')
    f.close()
    writer.close()
    save_path = saver.save(sess, "./model/model65.ckpt")
    print('save path:',save_path)
    print('Bimg_emb:',sess.run(vis_lstm_model.Bimg_emb))
    
    '''
    print("Testing Start!\n")
    avg_acc = 0.0
    total = 0
    while(batch_no*modOpts['batch_size'] < len(qa_data['training_data'])):
        img_f, q_vec, answer = get_training_batch(batch_no, modOpts, image_feat, qa_data, load_data, w2v_model)
        _, loss_value, acc, pred = sess.run([train_op, loss, accuracy, predictions], feed_dict={
            input_tensors['image']:img_f,
            input_tensors['sentence']:q_vec,
            input_tensors['answer']:answer
        })
        batch_no += 1
        if args.debug:
            for idx, p in enumerate(pred):
                print(idx, p, np.argmax(answer[idx]))
                #print("Loss", loss_value, batch_no, i)
                #print("Accuracy", acc)
                #print("----------------\n")
            avg_acc += acc
            total += 1
        #f.write(' '.join( ("Avg acc",str(avg_acc/total),'\n') ) )
    print(sess.run(vis_lstm_model.Wemb_hidden))
    print("Avg Acc:",avg_acc/total)
    '''
    
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
        count += 1

    return img_f, q_vector, answer


if __name__ == '__main__':
    main()
