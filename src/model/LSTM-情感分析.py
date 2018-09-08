import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import word2vec
import jieba
import tensorflow as tf
import numpy as np
import time
from random import randint
from random import shuffle
import os
import pandas as pd

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
print(root_path)
timeA = time.time()
data_path = os.path.join(
    os.path.join(
        root_path, 'data'))

colname_list = ['location_traffic_convenience',
                'location_distance_from_business_district',
                'location_easy_to_find',
                'service_wait_time',
                'service_waiters_attitude',
                'service_parking_convenience',
                'service_serving_speed',
                'price_level',
                'price_cost_effective',
                'price_discount',
                'environment_decoration',
                'environment_noise',
                'environment_space',
                'environment_cleaness',
                'dish_portion',
                'dish_taste',
                'dish_look',
                'dish_recommendation',
                'others_overall_experience',
                'others_willing_to_consume_again']


def makeStopWord():
    with open(os.path.join(data_path, '停用词.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    stopWord = []
    for line in lines:
        words = jieba.lcut(line, cut_all=False)
        for word in words:
            stopWord.append(word)
    return stopWord


def words2Array(lineList):
    linesArray = []
    wordsArray = []
    steps = []
    for line in lineList:
        t = 0
        p = 0
        for i in range(MAX_SIZE):
            if i < len(line):
                try:
                    wordsArray.append(model.wv.word_vec(line[i]))
                    p = p + 1
                except KeyError:
                    t = t + 1
                    continue
            else:
                wordsArray.append(np.array([0.0] * dimsh))
        for i in range(t):
            wordsArray.append(np.array([0.0] * dimsh))
        steps.append(p)
        linesArray.append(wordsArray)
        wordsArray = []
    linesArray = np.array(linesArray)
    steps = np.array(steps)
    return linesArray, steps


def makeData(data_path, colname):
    # 获取词汇，返回类型为[[word1,word2...],[word1,word2...],...]
    train_data = pd.read_csv(data_path)
    print("================")
    print(train_data)
    new_train_data = train_data[['content', colname]].head(5000)
    print("================")
    print(new_train_data)
    # new_train_data = new_train_data[new_train_data['location_traffic_convenience'] != -2]
    # print("================")
    # print(new_train_data)
    # 将评价数据转换为矩阵，返回类型为array
    labels = pd.get_dummies(new_train_data[colname])
    print(labels)
    wordArray, wordSteps = words2Array([list(jieba.cut(val)) for val in new_train_data['content']])
    print(wordSteps)
    new_train_data['steps'] = wordSteps
    print("================")
    print(new_train_data)
    Data, Steps, Labels = wordArray, wordSteps, labels
    return new_train_data['content'], Data, Steps, Labels


# ----------------------------------------------
# Word60.model   60维
# word2vec.model        200维

def create_model(colname):
    num_nodes = 32
    batch_size = 10
    output_size = 4

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, MAX_SIZE, dimsh))
        tf_train_steps = tf.placeholder(tf.int32, shape=(batch_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size))

        tf_test_dataset = tf.constant(testData, tf.float32)
        tf_test_steps = tf.constant(testSteps, tf.int32)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_nodes,
                                                 state_is_tuple=True)

        w1 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes // 4], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal([num_nodes // 4], stddev=0.1))

        w2 = tf.Variable(tf.truncated_normal([num_nodes // 4, 4], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal([4], stddev=0.1))

        def model(dataset, steps):
            outputs, last_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                     dtype=tf.float32,
                                                     sequence_length=steps,
                                                     inputs=dataset)
            hidden = last_states[-1]

            hidden = tf.matmul(hidden, w1) + b1
            logits = tf.matmul(hidden, w2) + b2
            return logits

        train_logits = model(tf_train_dataset, tf_train_steps)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                    logits=train_logits))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        prediction = tf.nn.softmax(model(tf_train_dataset, tf_train_steps))
        test_pre = tf.nn.softmax(model(tf_test_dataset, tf_test_steps))
    num_steps = 2000
    summary_frequency = 100

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        acc_val = 0
        for step in range(num_steps):
            offset = (step * batch_size) % (len(trainLabels) - batch_size)
            feed_dict = {tf_train_dataset: trainData[offset:offset + batch_size],
                         tf_train_labels: trainLabels[offset:offset + batch_size],
                         tf_train_steps: trainSteps[offset:offset + batch_size]}
            _, l = session.run([optimizer, loss],
                               feed_dict=feed_dict)
            mean_loss += l
            if step > 0 and step % summary_frequency == 0:
                mean_loss = mean_loss / summary_frequency
                print("The step is: %d" % (step))
                print("In train data,the loss is:%.4f" % (mean_loss))
                # mean_loss = 0
                # acrc = 0
                y_pre = session.run(prediction, feed_dict)
                correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(tf_train_labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                result = session.run(accuracy, feed_dict=feed_dict)

                print(result)
                # for i in range(len(prediction)):
                #     if prediction[i][testLabels[i].index(1)] > 0.5:
                #         acrc = acrc + 1
                # print("In test data,the accuracy is:%.2f%%" % ((acrc / len(testLabels)) * 100))
        #####################################
        test_prediction = session.run(test_pre)
        new_df[colname] = session.run(tf.arg_max(test_prediction, 1))
    timeB = time.time()
    print("time cost:", int(timeB - timeA))


if __name__ == '__main__':
    word2vec_path = os.path.join(
        os.path.join(data_path, 'w2v'),
        'word2vec.model')
    train_path = os.path.join(
        os.path.join(os.path.join(data_path, 'train'),
                     'sentiment_analysis_trainingset.csv'))
    test_path = os.path.join(
        os.path.join(os.path.join(data_path, 'test'),
                     'sentiment_analysis_testa.csv'))
    model = gensim.models.Word2Vec.load(word2vec_path)
    dimsh = model.vector_size
    MAX_SIZE = 100
    stopWord = makeStopWord()
    new_df = pd.DataFrame()
    for col in colname_list:
        # print("In train data:")
        ori_train_data, trainData, trainSteps, trainLabels = makeData(train_path, col)
        # print(trainData)
        # print("In test data:")
        ori_test_data, testData, testSteps, testLabels = makeData(test_path, col)
        trainLabels = np.array(trainLabels)
        new_df['content'] = ori_test_data.tolist()
        # print(new_df)
        #
        # print("-" * 30)
        # print("The trainData's shape is:", trainData.shape)
        # print("The trainSteps's shape is:", trainSteps.shape)
        # print("The trainLabels's shape is:", trainLabels.shape)
        create_model(col)
    new_df.to_csv('test.csv')
