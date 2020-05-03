'''
Purpose:Detection of pilots
Key words：grant-free，one source block, synchronous, zc sequence
Log:2019/6/5-----1.综合所有资源
'''
import tensorflow as tf
import numpy as np
from numpy import random as nr
import os
import time
from math import*
import model_inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'# 只显示 warning 和 Error

def seq_buff():

    seq_mat = np.array([[1.+0.j, 0.84125-0.54064j, -0.14231-0.98982j, -0.95949+0.28173j,0.84125+0.54064j, -0.65486-0.75575j,
                         0.84125+0.54064j, -0.95949+0.28173j,-0.14231-0.98982j,0.84125-0.54064j, 1.-0.j],
                        [ 0.84125+0.54064j, -0.65486-0.75575j,  0.84125+0.54064j, -0.95949+0.28173j,-0.14231-0.98982j,
                          0.84125-0.54064j, 1.-0.j, 1.+0.j, 0.84125-0.54064j, -0.14231-0.98982j, -0.95949+0.28173j],
                         [-0.14231-0.98982j, 0.84125-0.54064j,  1.-0.j, 1.+0.j,0.84125-0.54064j, -0.14231-0.98982j,
                          -0.95949+0.28173j,0.84125+0.54064j, -0.65486-0.75575j, 0.84125+0.54064j, -0.95949+0.28173j],
                         [ 0.84125-0.54064j, -0.14231-0.98982j, -0.95949+0.28173j, 0.84125+0.54064j, -0.65486-0.75575j,
                           0.84125+0.54064j, -0.95949+0.28173j, -0.14231-0.98982j, 0.84125-0.54064j, 1.-0.j,1.+0.j],
                         [-0.65486-0.75575j, 0.84125+0.54064j, -0.95949+0.28173j, -0.14231-0.98982j, 0.84125-0.54064j, 1.-0.j,
                          1.+0.j, 0.84125-0.54064j, -0.14231-0.98982j, -0.95949+0.28173j, 0.84125+0.54064j],
                         [ 0.84125-0.54064j, 1.-0.j, 1.+0.j, 0.84125-0.54064j, -0.14231-0.98982j, -0.95949+0.28173j,
                           0.84125+0.54064j, -0.65486-0.75575j,0.84125+0.54064j, -0.95949+0.28173j, -0.14231-0.98982j]])
    return seq_mat

'''
Purpose : 生成含噪声的训练集与验证集,按照MNIST数据集的方式，包含大量相同同信噪比噪声的信号
Plan : 根据概率分布对训练集中的数据进行调整
'''
def generater_fixSNR(repetition_base, SNR):
    JJ = 1j  # 复数符号

    rs_list = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])  # 资源选择列表

    # 噪声
    variance = 10 ** (-0.1 * SNR)
    # 用户活跃概率
    prob_user = 1 / 6

    user_pilots = seq_buff()  # 导频列表
    u_num = user_pilots.shape[0]  # 用户数量
    pilot_length = user_pilots.shape[1]  # 导频长度
    rs_num = 4  # 资源数量

    repetition_count = 0
    image_matrix = np.zeros(2 * pilot_length * 4)
    label_matrix = np.zeros(u_num * 4)

    while repetition_count < repetition_base:
        repetition_count = repetition_count + 1

        active_user = nr.uniform(0, 1, size=(1, u_num))
        active_user = np.array(active_user < prob_user, dtype=int)

        # 逐个用户生成资源占用矩阵
        user_occupy = np.zeros([rs_num, u_num])
        for i in range(u_num):
            choose_index = rs_list[nr.randint(0, 6)]
            user_occupy[choose_index[0], i] = 1
            user_occupy[choose_index[1], i] = 1
        # 与用户活跃状态结合，生成稀疏生成矩阵
        F = np.zeros([rs_num, u_num])
        for i in range(u_num):
            F[:, i] = user_occupy[:, i] * active_user.T[i]

        signal = np.zeros([rs_num, pilot_length], dtype=complex)
        for line in range(rs_num):
            for row in range(u_num):
                signal[line, :] = signal[line, :] + user_pilots[row, :] * F[line, row]

        for line in range(rs_num):
            for row in range(pilot_length):
                noise_phase = np.random.rand(1)
                noise = np.random.randn(1)
                AWGN_I = sqrt(variance) * noise * cos(2 * pi * noise_phase)
                AWGN_Q = sqrt(variance) * noise * sin(2 * pi * noise_phase)
                signal[line, row] = signal[line, row] + AWGN_I + JJ * AWGN_Q
            # 通过选择向量选择可能的导频序列进行叠加，即接受端可能收到的信号，共2**length

        carrier1 = np.zeros(2 * pilot_length * 4)
        for l in range(pilot_length * 4):
            carrier1[l * 2] = signal[divmod(l, 11)].real
            carrier1[l * 2 + 1] = signal[divmod(l, 11)].imag

        carrier2 = np.zeros(u_num * 4)
        for z in range(u_num * 4):
            carrier2[z] = F[divmod(z, 6)]

        image_matrix = np.row_stack((image_matrix, carrier1))
        label_matrix = np.row_stack((label_matrix, carrier2))

    return image_matrix, label_matrix

#生成训练集、验证集，测试集
SNR_dB = 9
train_images, train_labels = generater_fixSNR(20000, SNR_dB)
test_images, test_labels =generater_fixSNR(30000, SNR_dB)
validation_images, validation_labels = generater_fixSNR(100, SNR_dB)
print(test_images.shape)

BATCH_SIZE = 100  # 每次batch打包的样本个数

# 模型相关的参数
LEARNING_RATE_BASE = 0.8 # 基础的学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARAZTION_RATE = 0.0001 # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 20000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率

def train():
    x = tf.placeholder(tf.float32, [None, model_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, model_inference.OUTPUT_NODE], name='y-input')

    # 计算不含滑动平均类的前向传播结果
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = model_inference.inference(x, regularizer)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵及其平均值，不进行softmax计算
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(train_labels) / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率，先将输出层数组转化为0/1，再将输出层与标签数组中的数进行一对一对比，计算正确率
    one = tf.ones_like(y)
    zero = tf.zeros_like(y)
    y = tf.where(y < 0.5, x=zero, y=one)
    correct_prediction = tf.equal(y, y_)
    # 当神经元全部正确时判断为正确
    correct_prediction_int = tf.to_int32(correct_prediction)
    correct_sum = tf.reduce_sum(correct_prediction_int, axis=1)
    result = tf.equal(tf.reduce_sum(correct_prediction_int, axis=1), \
                      tf.reduce_sum(tf.ones_like(correct_prediction_int), axis=1))
    accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

    # 保存模型
    saver = tf.train.Saver()

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        validate_feed = {x: validation_images, y_: validation_labels}

        test_feed = {x: test_images, y_: test_labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=test_feed)
                losss = sess.run(loss, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
                print("After %d training step(s), loss is %g " % (i, losss))

            train_feed = {x: train_images[i:BATCH_SIZE + i], y_: train_labels[i:i + BATCH_SIZE]}
            sess.run(train_op, feed_dict=train_feed)
        saver.save(sess, 'E:\graduate\model9\modeltest.ckpt')
        print('model saved successfully')

        time_start = time.time()
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        time_end = time.time()
        print('totally cost', time_end - time_start)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

if __name__=='__main__':
    train()


