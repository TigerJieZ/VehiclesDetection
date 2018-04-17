import tensorflow as tf
import time
import numpy as np
import os

from input_data import mydata


class model:
    def variable_with_weight_loss(self, shape, stddev):
        weight = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        return weight

    # , images, batch_size, n_classes
    def inference(self, train_batch, batch_size, n_classes):
        # 卷积层
        weight1 = self.variable_with_weight_loss([11, 11, 1, 32], stddev=5e-2)
        bias1 = tf.constant(0, tf.float32, [32])
        kernel1 = tf.nn.conv2d(train_batch, weight1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout1 = tf.nn.dropout(pool1, keep_prob=0.5)
        # shape=[batch_size,23,28,16]

        weight2 = self.variable_with_weight_loss([11, 11, 32, 64], stddev=5e-2)
        bias2 = tf.constant(0.1, tf.float32, [64])
        kernel2 = tf.nn.conv2d(dropout1, weight2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout2 = tf.nn.dropout(pool2, keep_prob=0.5)
        # shape=[batch_size,12,14,32]

        weight3 = self.variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2)
        bias3 = tf.constant(0.1, tf.float32, [64])
        kernel3 = tf.nn.conv2d(dropout2, weight3, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, bias3))
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout3 = tf.nn.dropout(pool3, keep_prob=0.5)
        print(dropout3.get_shape())
        # shape=[batch_size,6,7,32]

        weight4 = self.variable_with_weight_loss([5, 5, 64, 128], stddev=5e-2)
        bias4 = tf.constant(0.1, tf.float32, [128])
        kernel4 = tf.nn.conv2d(dropout3, weight4, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.relu(tf.nn.bias_add(kernel4, bias4))
        pool4 = tf.nn.max_pool(conv4, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout4 = tf.nn.dropout(pool4, keep_prob=0.5)
        print(dropout4.get_shape())
        # shape=[batch_size,3,4,64]

        weight5 = self.variable_with_weight_loss([5, 5, 128, 128], stddev=5e-2)
        bias5 = tf.constant(0.1, tf.float32, [128])
        kernel5 = tf.nn.conv2d(dropout4, weight5, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.relu(tf.nn.bias_add(kernel5, bias5))
        pool5 = tf.nn.max_pool(conv5, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        dropout5 = tf.nn.dropout(pool5, keep_prob=0.5)
        # print(dropout5.get_shape())
        # # print(dropout5)

        ##########################
        # 全连接层
        reshape = tf.reshape(dropout5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight6 = self.variable_with_weight_loss([dim, 896], stddev=0.005)
        # 全连接层的偏置x是上一个卷积核个数
        bias6 = tf.constant(0.1, shape=[896], dtype=tf.float32)
        local6 = tf.nn.relu(tf.add(tf.matmul(reshape, weight6), bias6))

        weight7 = self.variable_with_weight_loss([896, 448], stddev=0.005)
        bias7 = tf.constant(0.1, shape=[448], dtype=tf.float32)
        local7 = tf.nn.relu(tf.add(tf.matmul(local6, weight7), bias7))

        ##################################
        # 输出层
        weight8 = self.variable_with_weight_loss(shape=[448, n_classes], stddev=1 / 448.0)
        bias8 = tf.constant(0.0, shape=[n_classes])
        logits = tf.add(tf.matmul(local7, weight8), bias8)

        return logits

    def new_inference(self, train_batch, batch_size, n_classes):
        print(train_batch.shape)
        conv_1 = self.conv_layer(1, train_batch, filters=64, size=7, stride=2, channels=3)
        pool_2 = self.pooling_layer(2, conv_1, 2, 2)
        conv_3 = self.conv_layer(3, pool_2, filters=192, size=3, stride=1, channels=64)
        pool_4 = self.pooling_layer(4, conv_3, 2, 2)
        conv_5 = self.conv_layer(5, pool_4, filters=128, size=1, stride=1, channels=192)
        conv_6 = self.conv_layer(6, conv_5, filters=256, size=3, stride=1, channels=128)
        conv_7 = self.conv_layer(7, conv_6, filters=256, size=1, stride=1, channels=256)
        conv_8 = self.conv_layer(8, conv_7, filters=512, size=3, stride=1, channels=256)
        pool_9 = self.pooling_layer(9, conv_8, 2, 2)
        conv_10 = self.conv_layer(10, pool_9, filters=256, size=1, stride=1, channels=512)

        # 全连接层
        reshape = tf.reshape(conv_10, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight6 = self.variable_with_weight_loss([dim, 896], stddev=0.005)
        # 全连接层的偏置x是上一个卷积核个数
        bias6 = tf.constant(0.1, shape=[896], dtype=tf.float32)
        local6 = tf.nn.relu(tf.add(tf.matmul(reshape, weight6), bias6))

        weight7 = self.variable_with_weight_loss([896, 448], stddev=0.005)
        bias7 = tf.constant(0.1, shape=[448], dtype=tf.float32)
        local7 = tf.nn.relu(tf.add(tf.matmul(local6, weight7), bias7))

        ##################################
        # 输出层
        weight8 = self.variable_with_weight_loss(shape=[448, n_classes], stddev=1 / 448.0)
        bias8 = tf.constant(0.0, shape=[n_classes])
        logits = tf.add(tf.matmul(local7, weight8), bias8)

        return logits

    def conv_layer(self, idx, inputs, filters, size, stride, channels):
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))
        conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
            idx, size, size, stride, filters, int(channels)))
        return conv_biased

    def pooling_layer(self, idx, inputs, size, stride):
        print('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride))
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        print('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
            idx, hiddens, int(dim), int(flat), 1 - int(linear)))
        if linear: return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def losses(self, logits, labels):
        '''

        :param logits: logits tensor, float, [batch_size, n_classes]
        :param labels: label tensor, tf.int32, [batch_size]
        :return: loss tensor of float type
        '''
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=labels,
                                                                           name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
            return loss

    def trainning(self, loss, learning_rate):
        '''
        Trainning ops
        :param loss: loss tensor, from losses()
        :param learning_rate: 学习率
        :return: the op of trainning
        '''
        with tf.name_scope('optimizer'):
            # 优化学习算法(降低loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        '''
        Evaluation the quality of the logits at predicting the label
        :param logits: Logits tensor,float - [batch_size, NUM_CLASS]
        :param labels: Labels tensor,int32 - [batch_size, with values in the range(0, NUM_CLASS)
        :return: a scalar int32 tensor with the number of examples(out of batch_size)
        '''
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy', accuracy)

        return accuracy


class train:
    def run_training(self, train, train_label):
        logs_train_dir = 'E:/checkpoint/'
        BATCH_SIZE = 16
        CAPACITY = 200
        N_CLASSES = 2
        learning_rate = 0.00001
        MAX_STEP = 2000
        train_batch, train_label_batch = data.get_batch(imgs=train,
                                                        labels=train_label,
                                                        img_width=64,
                                                        img_height=64,
                                                        batch_size=BATCH_SIZE,
                                                        capacity=CAPACITY)
        train_logits = model.new_inference(train_batch,
                                           BATCH_SIZE,
                                           N_CLASSES)
        train_loss = model.losses(train_logits, train_label_batch)
        train_op = model.trainning(loss=train_loss, learning_rate=learning_rate)
        # 准确率
        train_acc = model.evaluation(train_logits, train_label_batch)

        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                start_time = time.time()
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
                time_cost = time.time() - start_time
                if step % 50 == 0:
                    print("cost time:", time_cost,
                          'Step %d,train loss = %.2f,train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()


if __name__ == '__main__':
    trainer = train()
    model = model()
    data = mydata()
    train, label = data.get_datas_dirs_lists()
    trainer.run_training(train, label)
