''''
Purpose : 使用ZC序列导频列表，产生星座图组合
Data : 2019/5/21-2019/5/22
备注：2019/5/29 加入m序列作为对比
'''
import numpy as np
from numpy import random as nr
from math import*
import zcgenerate
import mgenerate

'''
读取导频列表与星座图列表
'''

# ZC序列，长度为11
# starmat,labelmat = zcgenerate.generate_standard() # 星座图列表
# user_pilots = zcgenerate.seq_buff() # 导频列表

# m序列，长度为15/7
feedback_index = np.array([1, 0, 0, 1])  # 反馈序列
starmat,labelmat = mgenerate.generate_standard(feedback_index) # 星座图列表
user_pilots = mgenerate.mseq(feedback_index)[0:6] # 导频列表

'''
基本参数设置
'''
JJ = 1j # 复数符号
u_num = user_pilots.shape[0] # 用户数量
pilot_length = user_pilots.shape[1] # 导频长度
rs_num = 4 # 资源数量

rs_list = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # 资源选择列表

# 噪声
SNR_dB = np.arange(1,8)
variance2 = 10**(-0.1*SNR_dB)

# 标记每个星座点对应的用户数量
switch = np.zeros(2**u_num)
for i in range(2**u_num):
    switch[i] = np.sum(labelmat[i,] == 1)

def Build_F():

    # 生成决定用户是否活跃的用户活跃向量
    active_user = nr.uniform(0, 1, size=(1, u_num))
    # print(active_user)
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
    # print('----------生成矩阵--------------')
    # print(F)
    return F


def Polit(FF):
    F = FF
    # 发送信息
    signal = np.zeros([rs_num, pilot_length], dtype=complex)
    for line in range(rs_num):
        for row in range(u_num):
            signal[line, :] = signal[line, :] + user_pilots[row, :] * F[line, row]

    for line in range(rs_num):
        for row in range(pilot_length):
            noise_phase = np.random.rand(1)
            noise = np.random.randn(1)
            AWGN_I = sqrt(variance2[j]) * noise * cos(2 * pi * noise_phase)
            AWGN_Q = sqrt(variance2[j]) * noise * sin(2 * pi * noise_phase)
            signal[line, row] = signal[line, row] + AWGN_I + JJ * AWGN_Q
    return signal

def Politdetection(signal_re):

    signal = signal_re

    F_R = np.zeros([rs_num, u_num])
    prob = np.zeros(2 ** u_num)

    # 逐个资源检测
    for rs in range(rs_num):
        # 逐个星座点计算最大后验概率
        for i in range(2 ** u_num):
            # 计算欧式距离
            distance = np.sum(abs(signal[rs, :] - starmat[i, :]))
            # 根据不同的先验概率计算AP
            if switch[i] == 0:
                prob[i] = prob_init[0] * exp(-distance)
            elif switch[i] == 1:
                prob[i] = prob_init[1] * exp(-distance)
            elif switch[i] == 2:
                prob[i] = prob_init[2] * exp(-distance)
            elif switch[i] == 3:
                prob[i] = prob_init[3] * exp(-distance)
            elif switch[i] == 4:
                prob[i] = prob_init[4] * exp(-distance)
            elif switch[i] == 5:
                prob[i] = prob_init[5] * exp(-distance)
            else:
                prob[i] = prob_init[6] * exp(-distance)

        # 根据MAP准则，确定对应的标签，即每个资源上的用户组合
        maxindex = np.argmax(prob)
        F_R[rs, :] = labelmat[maxindex]
    return F_R



for p in range(3):
    if p==0:
        prob_init = [0.33490, 0.40188, 0.20094, 0.05358, 0.00804, 0.00064, 2.1433e-05]  # 先验概率
        prob_user = 1/6
    if p==1:
        prob_init = [0.26214, 0.39322, 0.24576, 0.08192, 0.01536, 0.00154, 6.4000e-05]  # 先验概率
        prob_user = 1/5
    if p==2:
        prob_init = [0.17798, 0.35596, 0.29663, 0.13184, 0.03296, 0.00439, 0.00024]  # 先验概率
        prob_user = 1/4
    # if p==3:
    #     prob_init = [0.08779, 0.26337, 0.32922, 0.21948, 0.08230, 0.01646, 0.00137]  # 先验概率
    #     prob_user = 1/3
    print('用户活跃概率'+ str(prob_user))

    wrong_rate = np.zeros(12)  # 错误率曲线

    # 初始化
    # with open('ZC序列导频错检率.txt', mode='a') as w:
    #     w.write('SNR_dB' + '\t\t\t\t' + str(SNR_dB) + '\n')
    #     w.write('用户活跃概率' + '\t\t\t\t' + str(prob_user) + '\n')
    #     w.close()
    with open('m序列导频错检率.txt', mode='a') as w:
        w.write('SNR_dB' + '\t\t\t\t' + str(SNR_dB) + '\n')
        w.write('用户活跃概率' + '\t\t\t\t' + str(prob_user) + '\n')
        w.write('导频长度' + '\t\t\t\t' + str(pilot_length) + '\n')
        w.close()

    for j in range(len(SNR_dB)):
        error = 0  # 正确数计数器
        count = 0  # 循环数计数器
        repetition = 10 ** 5  # 循环次数
        repetition_fix = repetition
        while count<repetition:
            count = count + 1

            F = Build_F()
            if np.sum(F) == 0:
                repetition = repetition + 1
                continue
            signal = Polit(F)
            F_R = Politdetection(signal)

            if not (F == F_R).all():
                # 计算与生成矩阵完全相符的检测矩阵数量
                error = error + 1
        # print(count, repetition, count1, error, )
        wrong_rate[j] = error / repetition_fix
        print(wrong_rate[j])
    # 输出正确率
    # with open('ZC序列导频错检率.txt', mode='a') as w:
    #     w.write('错检率'+ '\t\t\t\t' + str(wrong_rate) + '\n')
    #     w.close()
    print(wrong_rate)
    with open('m序列导频错检率.txt', mode='a') as w:
        w.write('错检率'+ '\t\t\t\t' + str(wrong_rate[j])+ '\n')
        w.close()

