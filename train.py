import glob
import tensorflow as tf
import os
import inference
import numpy as np
import matplotlib.pyplot as plt

# 处理好之后的数据文件。
INPUT_DATA = './dataset/orl_faces.npy'
# 保存训练好的模型的路径。
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'face_model.ckpt'
#100
BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
#0.0001
REGULARIZATION_RATE = 0.0001
#6000
TRAINING_STEPS = 4000
MOVING_AVERAGE_DECAY = 0.99

def next_batch(images, labels, num_examples, batch_size, index_in_epoch):
    x = []
    y = []
    next_index = index_in_epoch + batch_size
    if next_index > num_examples:
        next_index = next_index % num_examples
        x1 = images[index_in_epoch:]
        x2 = images[:next_index]
        y1 = labels[index_in_epoch:]
        y2 = labels[:next_index]
        x_ = np.vstack((x1, x2))
        y_ = np.vstack((y1, y2))
    elif next_index < num_examples:
        x.append(images[index_in_epoch:next_index])
        y.append(labels[index_in_epoch:next_index])
        x_ = np.asarray(x)
        y_ = np.asarray(y)
    else:
        next_index = next_index % num_examples
        x.append(images[index_in_epoch:])
        y.append(labels[index_in_epoch:])
        x_ = np.asarray(x)
        y_ = np.asarray(y)
    index_in_epoch = next_index
    
    return (x_, y_, index_in_epoch)
        

def train(training_data):
#     print(type(training_data[0]))
    index_in_epoch = 0
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)
    print("11111111111")
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(training_data[0]) / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    print('222222222222222')
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    print('3333333333333333')
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#         print('initlized')
        print('444444444444444')
    
    
#----------------------------------------    
    
#         for i in range(TRAINING_STEPS):
#             print('5555555555-------',i)
# #             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#             xs, ys, index_in_epoch= next_batch(training_data[0],training_data[1],len(training_data[0]), BATCH_SIZE, index_in_epoch)
#             reshaped_xs = np.reshape(xs, (
#                 BATCH_SIZE,
#                 inference.IMAGE_SIZE,
#                 inference.IMAGE_SIZE,
#                 inference.NUM_CHANNELS))
#             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

#----------------------------------------------------------            
        start = 0
        end = BATCH_SIZE
        n_training_example = len(training_data[0])
        for i in range(1, TRAINING_STEPS + 1):
#             xs = training_data[0][start:end]
#             ys = training_data[1][start:end]
#             start = end
#             if start == n_training_example:
#                 start = 0
#             end = start + BATCH_SIZE
#             if end > n_training_example: 
#                 end = end - n_training_example
#                 xs = np.vstack((xs, training_data[0][:end]))
#                 ys = np.vstack((ys, training_data[1][:end]))
            if start > end:
                x1 = training_data[0][start:]
                x2 = training_data[0][:end]
                y1 = training_data[1][start:]
                y2 = training_data[1][:end]
                xs = np.vstack((x1, x2))
                ys = np.vstack((y1, y2))
            else:
                xs = training_data[0][start:end]
                ys = training_data[1][start:end]
            start = end
            end = end + BATCH_SIZE
            if start == n_training_example:
                start = 0
            if end > n_training_example:
                end = end - n_training_example
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
#--------------------------------------------------

            if i % 1000 == 0:
                print("i = %d, Afdter %d training step(s), loss on training batch is %g." % (i, step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)


if __name__ == '__main__':
    photo_data = np.load(INPUT_DATA)
    training_images = np.asarray(photo_data[0])
    training_labels = np.asarray(photo_data[1])
    n_training_example = len(training_images)
    print("%d training examples has loaded." % (n_training_example))
    print(type(training_images))
    train([training_images, training_labels])