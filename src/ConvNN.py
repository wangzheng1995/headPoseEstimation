import tensorflow as tf
import os
import numpy as np
import time
from scipy import ndimage
from sklearn.metrics import confusion_matrix
import pandas as pd
from input_data import read_data_sets
from input_data import DataSet

datasets_file_path = "/home/onegene/file/Face/datasets/"
set1_origin_file = datasets_file_path+"set1_origin"
set2_origin_file = datasets_file_path+"set2_origin"
set1_augment_file = datasets_file_path+"set1_augment_rgb_jitter_flip"
set2_augment_file = datasets_file_path+"set2_augment_rgb_jitter_flip"
ut_origin_file = datasets_file_path+"frms_inter"
ut_augment_file = datasets_file_path+"frms_inter_augment"

class DataSets(object):
        pass

data_sets = DataSets()
data_sets.set1_origin = read_data_sets(set1_origin_file)
data_sets.set2_origin = read_data_sets(set2_origin_file)
data_sets.set1_augment = read_data_sets(set1_augment_file)
data_sets.set2_augment = read_data_sets(set2_augment_file)
data_sets.ut_origin = read_data_sets(ut_origin_file)
data_sets.ut_augment = read_data_sets(ut_augment_file)
print('After Resize')
print('set1_origin image size: '+str(data_sets.set1_origin.images.shape))
print('set1_origin label size: '+str(data_sets.set1_origin.labels.shape))
print('set2_origin image size: '+str(data_sets.set2_origin.images.shape))
print('set2_origin label size: '+str(data_sets.set2_origin.labels.shape))
print('set1_augment image size: '+str(data_sets.set1_augment.images.shape))
print('set1_augment label size: '+str(data_sets.set1_augment.labels.shape))
print('set2_augment image size: '+str(data_sets.set2_augment.images.shape))
print('set2_augment label size: '+str(data_sets.set2_augment.labels.shape))
print('ut_origin image size: '+str(data_sets.ut_origin.images.shape))
print('ut_origin label size: '+str(data_sets.ut_origin.labels.shape))
print('ut_augment image size: '+str(data_sets.ut_augment.images.shape))
print('ut_augment label size: '+str(data_sets.ut_augment.labels.shape))

def concate_dataset(dataset1, dataset2):
    return DataSet(np.concatenate((dataset1.images, dataset2.images)),
                   np.concatenate((dataset1.labels, dataset2.labels)), 
                   True)

train_dataset = data_sets.set1_origin#concate_dataset(data_sets.set2_origin, data_sets.set2_augment)
test_dataset = data_sets.ut_origin
print('train image size: '+str(train_dataset.images.shape))
print('train label size: '+str(train_dataset.labels.shape))
print('test image size: '+str(test_dataset.images.shape))
print('test label size: '+str(test_dataset.labels.shape))


def my_confusion_matrix(predictions, labels):
    prediction_type = np.argmax(predictions, 1)
    label_type = np.argmax(labels, 1)#np.where(labels[:]==1)[1]
    #print(prediction_type.shape)
    #print(label_type.shape)
    y_actu = pd.Series(label_type, name='Actual')
    y_pred = pd.Series(prediction_type, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print('-'*40)
    print(df_confusion / df_confusion.sum(axis=1))
    print('-'*40)
    return pd.crosstab(y_actu, y_pred)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


'''
The structure of the net:

input: 32X32 gray scale

Conv1: 5X5 stride=1 out:28X28X16

Maxpooling1: 2X2 out:14X14X16

Conv2: 5X5 stride=1 out:10X10X20

Maxpooling2: 2X2 out:5X5X20

Conv3: 3X3 stride=1 out:3X3X20=180

Conv4: 3X3 out:1X1X120=120 to fc

Fc1: 120

Fc2: 84

output: 5 (for me)
'''

image_size = 32
num_labels = 5
num_channels = 1#grayscale

batch_size = 256
conv1_size = 5
conv2_size = 5
conv3_size = 3
conv4_size = 3
depth1 = 16
depth2 = 20
depth3 = 20
depth4 = 120
fc1_size = 120
fc2_size = 84

graph = tf.Graph()

with graph.as_default():
    
    #Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset.images)
    tf_valid_dataset = tf.constant(valid_dataset.images)
    keep_prob = tf.placeholder(tf.float32)
    #Variables.
    with tf.name_scope('conv1') as scope1:
        layer1_weights = tf.Variable(tf.truncated_normal(
                [conv1_size, conv1_size, num_channels, depth1], stddev=0.1), name='weight')
        layer1_biases = tf.Variable(tf.zeros([depth1]), name='biases')
    with tf.name_scope('conv2') as scope2:
        layer2_weights = tf.Variable(tf.truncated_normal(
            [conv2_size, conv2_size, depth1, depth2], stddev=0.1), name='weight')
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]), name='biases')
    with tf.name_scope('conv3') as scope3:
        layer3_weights = tf.Variable(tf.truncated_normal(
            [conv3_size, conv3_size, depth2, depth3], stddev=0.1), name='weight')
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth3]), name='biases')
    with tf.name_scope('conv4') as scope4:
        layer4_weights = tf.Variable(tf.truncated_normal(
           [conv4_size, conv4_size, depth3, depth4], stddev=0.1), name='weight')
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[depth4]), name='biases')
    with tf.name_scope('fc1') as scope5:
        fc1_weights = tf.Variable(tf.truncated_normal(
            [fc1_size, fc2_size], stddev=0.1), name='weight')
        fc1_biases = tf.Variable(tf.constant(1.0, shape=[84]), name='biases')
    with tf.name_scope('fc2') as scope6:
        fc2_weights = tf.Variable(tf.truncated_normal(
            [fc2_size, num_labels], stddev=0.1))
        fc2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    #Model.
    def model(data, keep_prob):
#         print(data.get_shape())
        with tf.name_scope(scope1):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID', name='conv')
            hidden = tf.nn.relu(conv + layer1_biases, name='acti')
            hidden_dropout = tf.nn.dropout(hidden, keep_prob, name='dropout')
#             print(hidden_dropout.get_shape())
            hidden = tf.nn.max_pool(hidden_dropout, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
#             print(hidden.get_shape())
        with tf.name_scope(scope2):
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='VALID', name='conv')
            hidden = tf.nn.relu(conv + layer2_biases, name='acti')
#             print(hidden.get_shape())
            hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool')
#             print(hidden.get_shape())
        with tf.name_scope(scope3):
            conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='VALID', name='conv')
            hidden = tf.nn.relu(conv + layer3_biases, name='acti')
#             print(hidden.get_shape())
        with tf.name_scope(scope4):
            conv = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='VALID', name='conv')
            hidden = tf.nn.relu(conv + layer4_biases, name='acti')
            shape = hidden.get_shape().as_list()
#             print(shape)
        with tf.name_scope(scope5):
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]], name='reshape')
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases, name='acti')
        with tf.name_scope(scope6):
            hidden = tf.tanh(tf.matmul(hidden, fc2_weights) + fc2_biases, name='acti')
        #print('size of last hidden layer: '+str(hidden.get_shape()))
        #print(hidden)
        return hidden
    #Training computation.
    logits = model(tf_train_dataset, keep_prob)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        summary_loss = tf.scalar_summary('summary_loss', loss)
    #Learining rate decay.
    global_step = tf.Variable(0) #count the number of steps taken.
    with tf.name_scope('learning_rate'):
        learning_rate = 0.1#tf.train.exponential_decay(0.1, global_step, 800, 0.8, staircase=True)
        summary_learning_rate = tf.scalar_summary('summary_learning_rate', learning_rate)
    #Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #Predictions for the training and test data
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
    #Accuarcy.
#     print train_prediction.shape
#     print tf_train_labels
# Evaluate model

    with tf.name_scope('valid_accuracy'):
        #valid_accuracy = accuracy(valid_prediction, valid_dataset.labels)
        correct_pred = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(valid_dataset.labels, 1))
        valid_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        summary_valid_accuracy = tf.scalar_summary('summary_valid_accuracy', valid_accuracy)
    with tf.name_scope('valid_loss'):
        valid_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(valid_prediction, valid_dataset.labels))
        summary_valid_loss = tf.scalar_summary('summary_valid_loss', valid_loss)
#     print(test_dataset.images.shape)
    merged_train = tf.merge_summary([summary_loss, summary_learning_rate])
    merged_valid = tf.merge_summary([summary_valid_loss, summary_valid_accuracy])

set1_confusion_matrix = 0
set2_confusion_matrix = 0


num_steps = 10001
ISOTIMEFORMAT='%Y-%m-%d %X'
current_time = time.strftime( ISOTIMEFORMAT, time.localtime() )
model_name = current_time+"_test_set1_trained_on_set2 "
summary_file_path = "/home/onegene/file/Face/summary"
model_file_path = "/home/onegene/file/Face/model/"+model_name+str(num_steps)
log_file = current_time+"_"+str(batch_size)+"_"+str(num_steps)+"_train-"+str(train_dataset.images.shape[0])+"_test-"+str(test_dataset.images.shape[0])
log = open("/home/onegene/file/Face/log/"+log_file, 'w')
with tf.Session(graph=graph) as session:
    start = time.time()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    train_writer = tf.train.SummaryWriter(summary_file_path + '/train')#, session.graph_def)
    valid_writer = tf.train.SummaryWriter(summary_file_path + '/valid')
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_dataset.images.shape[0] - batch_size)
        start, end, batch_data, batch_labels = train_dataset.next_batch(batch_size)
        #print('feed '+str(start)+' - '+str(end))
        #batch_data = datasets.train[offset:(offset + batch_size), :, :, :]
        #batch_labels = datasets.test[offset:(offset + batch_size), :]
        #if step % 40 != 0:
        my_feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.8}
        summary_train, _, l, predictions = session.run([merged_train, optimizer, loss, train_prediction], feed_dict=my_feed_dict)
        if (step % 400 == 0):
            print(step)
            my_feed_dict={keep_prob : 1.0}
            summary_valid, my_valid_loss, my_valid_accuracy =session.run([merged_valid, valid_loss, valid_accuracy], feed_dict=my_feed_dict)
            valid_writer.add_summary(summary_valid, step)
        if step == num_steps-1:
            saver.save(session, model_file_path, global_step = step)
        train_writer.add_summary(summary_train, step)
    end = time.time()
    print("training for %f minutes" % float((end-start)/60))
    log.writelines("training for %f minutes\n" % float((end-start)/60))
    my_feed_dict={keep_prob : 1.0}
    start_test = time.time()
    test_prediction = test_prediction.eval(feed_dict=my_feed_dict)
    end_test = time.time()
    print("testing for %f minutes" % float((end_test-start_test)/60))
    log.writelines("testing for %f minutes\n" % float((end_test-start_test)/60))
    print(' ')
    print('backwards:0  frontal_left:1  frontal_right:2  profile_left:3  profile_right:4')
    set1_confusion_matrix = my_confusion_matrix(my_prediction, test_dataset.labels)
    test_accuracy = accuracy(test_prediction, test_dataset.labels)
    print('Test accuray: %.1f%%' % test_accuracy)
    log.writelines('Test accuray: %.1f%%\n' % test_accuracy)
    session.close()
    log.close()
