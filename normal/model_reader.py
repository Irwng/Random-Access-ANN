import tensorflow as tf
import numpy as np
import os
import time
from numpy import random as nr
import model_inference
from math import*
# from neural_network import*
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '4'# 只显示 warning 和 Error

'''
Purpose : 生成含噪声的训练集与验证集,按照MNIST数据集的方式，包含大量相同同信噪比噪声的信号
Plan : 根据概率分布对训练集中的数据进行调整
'''
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
    print(F)
    return image_matrix[1:,], label_matrix[1:,]

SNR_dB = 10
test_images, test_labels =generater_fixSNR(1, SNR_dB)
# print(test_images, test_labels)

with tf.Graph().as_default() as g:

    x = tf.placeholder(tf.float32, [None, model_inference.INPUT_NODE], name='x-input')
    # y_ = tf.placeholder(tf.float32, [None, model_inference.OUTPUT_NODE], name='y-input')
    test_feed = {x: test_images}

    y = model_inference.inference(x,None)
    one = tf.ones_like(y)
    zero = tf.zeros_like(y)
    y = tf.where(y < 0.5, x=zero, y=one)

    # correct_prediction = tf.equal(y,y_)
    # correct_prediction_int = tf.to_int32(correct_prediction)
    # correct_sum = tf.reduce_sum(correct_prediction_int, axis=1)
    # result = tf.equal(tf.reduce_sum(correct_prediction_int, axis=1),
    #                   tf.reduce_sum(tf.ones_like(correct_prediction_int), axis=1))
    # accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

    # 计算不含滑动平均类的前向传播结果

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'E:\graduate\model\model10\modeltest.ckpt')
        y_out = sess.run(y, feed_dict=test_feed)
        # time_start = time.time()
        # test_acc = sess.run(accuracy, feed_dict=test_feed)
        # time_end = time.time()
        # print('totally cost', time_end - time_start)
        # print(("test accuracy using average model is %g" % test_acc))
        print(test_labels.reshape((4, 6)))
        print(y_out.reshape((4, 6)))
