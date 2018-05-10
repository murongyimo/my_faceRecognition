import time
import tensorflow as tf
import numpy as np
#加载程序中定义的常量和函数
import inference
import train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(photo_data):
    n_valid_examples = len(photo_data[0])
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            n_valid_examples,
            inference.IMAGE_SIZE,
            inference.IMAGE_SIZE,
            inference.NUM_CHANNELS],
        name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: photo_data[0], y_: photo_data[1]}
        #训练集
        train_feed = {x: photo_data[2], y_: photo_data[3]}
        y = inference.inference(x, False, None)
        #预测值y和真实值y_中相同的为1，不同的为0
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #将数组求平均值，得到预测结果的准确率即为神经网络的准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score_valid = sess.run(accuracy, feed_dict=validate_feed)
                    accuracy_score_train = sess.run(accuracy, feed_dict=train_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score_valid))
                    print("After %s training step(s), training accuracy = %g" % (global_step, accuracy_score_train))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

if __name__ == '__main__':
    photo_data = np.load(train.INPUT_DATA)
    valid_images = np.asarray(photo_data[2])
    valid_labels = np.asarray(photo_data[3])
    test_images = np.asarray(photo_data[4])
    test_labels = np.asarray(photo_data[5])
    print("%d training examples, %d validation examples and %d testing examples." % (
        len(photo_data[0]), len(photo_data[2]), len(photo_data[4])))
    #合并valid与test
    valid_images = np.vstack((valid_images,test_images))
    valid_labels = np.vstack((valid_labels, test_labels))
    n_valid_examples = len(valid_images)
    print(type(valid_labels),valid_labels.shape)
    
    train_images = np.asarray(photo_data[0][:n_valid_examples])
    train_labels = np.asarray(photo_data[1][:n_valid_examples])

    evaluate([valid_images,valid_labels,train_images,train_labels])