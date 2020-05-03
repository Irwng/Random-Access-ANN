''''
Purpose : 上行免授权SCMA-MPA与SCMA-ML检测（生成矩阵无差错,无编码）
Data : 2019/5/27-2019/5/29
检测性能：误比特率，丢包率
'''
import numpy as np
from math import*
import matplotlib.pyplot as plt
import time
from numpy import random as nr

'''
与生成矩阵无关变量即参数定义
'''
N_pilot = 10 ** 5  # 数据的组数
N_data = 10 ** 2   # 每组数据中数据的数
LOOP = 0           # 数据包的循环指针
loop = 0           # 数据的循环指针
L = 1              # BPSK bit per symbol
Du = 2             # Number of resources connected to user
M = 2              # BPSK modulation order
JJ = 1j            # complex index
prob_user = 1/6    # 用户活跃概率
rs_list = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]) # 资源选择列表

EbN0_dB = np.arange(12,16)
# EbN0_dB = np.array([17])

Eb_N0 = 10 ** (0.1 * EbN0_dB)
variance = 10 ** (-0.1 * EbN0_dB)  # variance of noise
Noise_Variance = 0
BER_TOTAL = np.zeros(2)            # 分别记录数据的MPA与ML的误比特数
BER = np.zeros(2)                  # 分别记录数据的MPA与ML的误比特率
Imax = 20                          # maximum number of iteration
TH = 0.0001                        # threshold value

RF = pi/5          # Rotation Factor
a = np.zeros(6, dtype=complex)
for i in range(len(a)):
    a[i] = cos(i * RF) + JJ * sin(i * RF)

G = np.array(
    [[a[0], a[1], a[2], a[3], a[4], a[5]],
     [a[1], a[2], a[3], a[4], a[5], a[0]],
     [a[2], a[3], a[4], a[5], a[0], a[1]],
     [a[3], a[4], a[5], a[0], a[1], a[2]]], dtype=complex)


# 生成原始生成矩阵
def Build_F():

    # 生成决定用户是否活跃的用户活跃向量
    active_user = nr.uniform(0, 1, size=(1, 6))
    # print(active_user)
    active_user = np.array(active_user < prob_user, dtype=int)

    # 逐个用户生成资源占用矩阵
    user_occupy = np.zeros([4, 6])
    for i in range(6):
        choose_index = rs_list[nr.randint(0, 6)]
        user_occupy[choose_index[0], i] = 1
        user_occupy[choose_index[1], i] = 1
    FF = np.zeros([4, 6])
    # 与用户活跃状态结合，生成稀疏生成矩阵
    for i in range(6):
        FF[:, i] = user_occupy[:, i] * active_user.T[i]
    return FF

# 获取原始的生成矩阵，并简化，之后根据简化的生成矩阵，设置由生成矩阵决定的参数与变量
def Variable(FF):

    # 根据检测到的生成矩阵，处理生成矩阵，将全部为0的行或列删去，简化矩阵
    FF = FF.astype(complex)

    for row in range(FF.shape[0]):
        for line in range(FF.shape[1]):
            FF[row, line] = FF[row, line] * G[row, line]

    GG = np.zeros(FF.shape[1])
    for i in range(FF.shape[0]):
        if np.sum(FF[i, :]) != 0:
            GG = np.row_stack((GG, FF[i, :]))
    # print(F)
    GG = GG[1:, :]
    F1 = GG.T

    F2 = np.zeros(F1.shape[1])
    for i in range(F1.shape[0]):
        if np.sum(F1[i, :]) != 0:
            F2 = np.row_stack((F2, F1[i, :]))
    global F
    F = F2[1:, :].T

    # 定义全局变量
    global K, J, Dr, MP, SC
    global data, data_estimate, Symbol, TxSymbol, RxSymbol, Decode
    global Msg, Mpoint, SubMsg, Spoint, unorm0, Phi
    global ap, I_v2g, Vc ,Fc

    K = F.shape[0]  # Number of resources
    J = F.shape[1]  # Number of users

    Dr = np.zeros(K, dtype=int)  # Number of users connected to resources
    for k in range(K):
        Dr[k] = np.sum(abs(F[k, :]) != 0)
    MP = 2 ** J  # Master-constellation Points
    SC = 2 ** Dr  # Sub-constellation Points

    data = np.zeros(J)  # transmitter binary data
    data_estimate = np.zeros(J)  # decode binary sequence
    Symbol = np.zeros(J)
    TxSymbol = np.zeros(K, dtype=complex)
    RxSymbol = np.zeros(K, dtype=complex)
    Decode = np.zeros(J)
    Msg = np.zeros([MP, J], dtype=complex)  # Master-constellation Points mapping on user
    Mpoint = np.zeros([MP, K], dtype=complex)  # Master-constellation Points mapping on resources

    # 由于Dr不确定，在不同的资源数下，都需要对每一个资源都生成子星座图
    if K == 2:
        SubMsg = np.array([np.zeros([SC[0], Dr[0]]), np.zeros([SC[1], Dr[1]])])
        Spoint = np.array([np.zeros(SC[0], dtype=complex), np.zeros(SC[1], dtype=complex)])
        Fc = np.array([np.zeros(Dr[0], dtype=int), np.zeros(Dr[1], dtype=int)])
        unorm0 = np.array([np.zeros(SC[0]), np.zeros(SC[1])])
        Phi = np.array([np.zeros(SC[0]), np.zeros(SC[1])])

    if K == 3:
        SubMsg = np.array([np.zeros([SC[0], Dr[0]]), np.zeros([SC[1], Dr[1]]),
                           np.zeros([SC[2], Dr[2]])])
        Spoint = np.array([np.zeros(SC[0], dtype=complex), np.zeros(SC[1], dtype=complex),
                           np.zeros(SC[2], dtype=complex)])
        Fc = np.array([np.zeros(Dr[0], dtype=int), np.zeros(Dr[1], dtype=int),
                       np.zeros(Dr[2], dtype=int)])
        unorm0 = np.array([np.zeros(SC[0]), np.zeros(SC[1]), np.zeros(SC[2])])
        Phi = np.array([np.zeros(SC[0]), np.zeros(SC[1]), np.zeros(SC[2])])

    if K == 4:
        SubMsg = np.array([np.zeros([SC[0], Dr[0]]), np.zeros([SC[1], Dr[1]]),
                           np.zeros([SC[2], Dr[2]]), np.zeros([SC[3], Dr[3]])])
        Spoint = np.array([np.zeros(SC[0], dtype=complex), np.zeros(SC[1], dtype=complex),
                           np.zeros(SC[2], dtype=complex), np.zeros(SC[3], dtype=complex)])
        Fc = np.array([np.zeros(Dr[0], dtype=int), np.zeros(Dr[1], dtype=int),
                       np.zeros(Dr[2], dtype=int), np.zeros(Dr[3], dtype=int)])
        unorm0 = np.array([np.zeros(SC[0]), np.zeros(SC[1]), np.zeros(SC[2]), np.zeros(SC[3])])
        Phi = np.array([np.zeros(SC[0]), np.zeros(SC[1]), np.zeros(SC[2]), np.zeros(SC[3])])

    # prior probability
    ap = np.zeros([J, M])
    for i in range(J):
        for j in range(M):
            ap[i, j] = 1 / M

    # variable nodes connected with function
    for k in range(K):
        flag = 0
        for j in range(J):
            if abs(F[k, j]) != 0:
                Fc[k][flag] = j
                flag = flag + 1

    # function nodes connected with variable
    Vc = np.zeros([J, 2], dtype=int)
    for j in range(J):
        flag = 0
        for k in range(K):
            if abs(F[k, j]) != 0:
                Vc[j, flag] = k
                flag = flag + 1

    I_v2g = np.zeros([J, Du, M])  # message passing from VN to its neighboring FNs

# 主星座图
def MasterConstellation():

    # 生成每一个星座图对应的消息组合，例如当第i个消息为[1，-1, 1]时，对应第i个星座点
    # 消息的顺序为[1,1,1,1]-->[1,1,1,-1]-->[1,1,-1,1]
    for i in range(MP):
        for j in range(J):
            a = 2 ** (j + 1)
            b = 2 ** j
            if i % a >= b:
                Msg[i,J - j - 1] = -1
            else:
                Msg[i,J - j - 1] =  1

    for i in range(MP):
        for k in range(K):
            # print(Msg[i, :], F[k, :])
            Mpoint[i,k] =np.dot(Msg[i, :], F[k, :])

# 子星座图
def SubConstellation():

    for k in range(K):  # 在每一个资源上计算子星座图
        for i in range(SC[k]): # 该资源对应的星座点数
            for j in range(Dr[k]): # 该资源连接的用户数，他们的组合对应星座点
                a = 2 ** (j + 1)
                b = 2 ** j
                if i % a >= b:
                    SubMsg[k][i, Dr[k] - j - 1] = -1
                else:
                    SubMsg[k][i, Dr[k] - j - 1] =  1
            # 根据每个资源上的每一种组合结合生成矩阵，获得子星座图，每个星座图有2**Dr(k)点
            flag = 0
            for index in range(J):
                if (abs(F[k,index])!=0):
                    Spoint[k][i] = Spoint[k][i] + F[k,index] * SubMsg[k][i,flag]
                    flag = flag + 1

# 初始化，开启记事本
def Initialize():

    print("Eb/N0\t\t\t   BER  \t\t\t  BLER")
    with open('上行免授权SCMA-MPA误比特率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:数据帧长度为' + str(N_data) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('误比特率：' + '\n')
        w.close()
    with open('上行免授权SCMA-ML误比特率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:数据帧长度为' + str(N_data) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('误比特率：' + '\n')
        w.close()
    with open('上行免授权SCMA-MPA丢包率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:数据帧长度为' + str(N_data) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('丢包率：' + '\n')
        w.close()
    with open('上行免授权SCMA-ML丢包率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:数据帧长度为' + str(N_data) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('丢包率：' + '\n')
        w.close()

# 生成数据并调制
def DataandModu():

    for j in range(J):
        data[j] = np.random.randint(0,2)
        Symbol[j] = (data[j]) * (-2) + 1

# 星座图旋转
def PhaseRotation():

    for k in range(K):
        TxSymbol[k] = np.dot(F[k, :], Symbol)

# 接收端加Guass噪声
def Receiver():

    for k in range(K):
        noise_phase = np.random.rand(1) + np.random.rand(1)/10000
        G = sqrt(-2 * log(np.random.rand(1)))
        # G = np.random.randn(1)
        AWGN_I = sqrt(Noise_Variance) * G * cos(2 * pi * noise_phase)
        AWGN_Q = sqrt(Noise_Variance) * G * sin(2 * pi * noise_phase)
        RxSymbol[k] = TxSymbol[k] + AWGN_I + JJ * AWGN_Q


def MPA():  # 重中之重啊！！！！！！！！

    '''
    不根据K改变的参数和变量
    '''
    # g = 0
    unorm = np.zeros(M)  # u-normalization probability
    Q_prev = np.zeros([J, M])  # final probability before iterative
    Q_post = np.zeros([J, M])  # final probability after iterative

    '''
    由K取值决定的参数和变量,基本所有需要逐个资源计算的变量都需要重新设计
    '''
    # message passing from FN to its neighboring VNs
    if K == 2:
        I_g2v = np.array([np.zeros([Dr[0], M]), np.zeros([Dr[1], M])])

    if K == 3:
        I_g2v = np.array([np.zeros([Dr[0], M]), np.zeros([Dr[1], M]), np.zeros([Dr[2], M])])

    if K == 4:
        I_g2v = np.array([np.zeros([Dr[0], M]), np.zeros([Dr[1], M]), np.zeros([Dr[2], M]), np.zeros([Dr[3], M])])

    # sub starpoints
    for k in range(K):
        sum = 0
        for i in range(SC[k]):
            f =  -abs(RxSymbol[k] - Spoint[k][i])**2 / Noise_Variance
            unorm0[k][i] = exp(f)
            sum = sum + unorm0[k][i]

        for i in range(SC[k]):
            Phi[k][i] = unorm0[k][i]/sum

    # I_v2g[J][Du][M]
    for j in range(J):
        for i in range(Du):
            for m in range(M):
                I_v2g[j,i,m] = 1/M

    # iterative
    for I in range(Imax):
        counter = 0
        # FN update
        for k in range(K):
            # Vnode = np.zeros(Dr[k])
            for i in range(Dr[k]):
                if Dr[k] == 1:
                    for m in range(M):
                        I_g2v[k][i, m] = Phi[k][m]

                if Dr[k] == 2:
                    if i == 0:
                        VNode1 = Fc[k][1]
                    if i == 1:
                        VNode1 = Fc[k][0]

                    for p in range(Du):
                        if Vc[VNode1][p] == k:
                            FNode1 = p

                    for m in range(M):  # m,e1,e2表示与资源相连的三个用户节点
                        temp = 0
                        for e1 in range(M):  # UE1
                            if i == 0:
                                t = m * (M ** (Dr[k] - 1)) + e1 * (M ** (Dr[k] - 2))
                            if i == 1:
                                t = e1 * (M ** (Dr[k] - 1)) + m * (M ** (Dr[k] - 2))
                            temp = temp + Phi[k][t] * I_v2g[VNode1, FNode1, e1]
                        I_g2v[k][i, m] = temp

                if Dr[k] == 3:
                    # find the other VN connected to this FN
                    if i == 0:
                        VNode1 = Fc[k][1]
                        VNode2 = Fc[k][2]
                    if i == 1:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][2]
                    if i == 2:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                    # 确定与该资源在与之相连的用户节点连接的资源中是第几个，
                    # 以从I_v2g中取出相应的数值，也让I_g2v可以将计算过后的值更新回去
                    for p in range(Du):
                        if Vc[VNode1, p] == k:
                            FNode1 = p
                        if Vc[VNode2,p] == k:
                            FNode2 = p

                    for m in range(M):   # m,e1,e2表示与资源相连的三个用户节点
                        temp = 0
                        for e1 in range(M):  # UE1
                            for e2 in range(M):  # UE2
                                if i == 0:
                                    t = m * (M ** (Dr[k] - 1)) + e1 * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3))
                                if i == 1:
                                    t = e1 * (M ** (Dr[k] - 1)) + m * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3))
                                if i == 2:
                                    t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + m * (M ** (Dr[k] - 3))
                                temp = temp + Phi[k][t] * I_v2g[VNode1,FNode1,e1] * I_v2g[VNode2,FNode2,e2]
                        I_g2v[k][i,m] = temp

                if Dr[k] == 4:
                    # find the other VN connected to this FN
                    if i == 0:
                        VNode1 = Fc[k][1]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                    if i == 1:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                    if i == 2:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][3]
                    if i == 3:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                    # 确定与该资源在与之相连的用户节点连接的资源中是第几个，
                    # 以从I_v2g中取出相应的数值，也让I_g2v可以将计算过后的值更新回去
                    for p in range(Du):
                        if Vc[VNode1, p] == k:
                            FNode1 = p
                        if Vc[VNode2, p] == k:
                            FNode2 = p
                        if Vc[VNode3, p] == k:
                            FNode3 = p

                    for m in range(M):  # m,e1,e2表示与资源相连的三个用户节点
                        temp = 0
                        for e1 in range(M):  # UE1
                            for e2 in range(M):  # UE2
                                for e3 in range(M):
                                    if i == 0:
                                        t = m * (M ** (Dr[k] - 1)) + e1 * (M ** (Dr[k] - 2)) + \
                                            e2 * (M ** (Dr[k] - 3)) + e3 * (M ** (Dr[k] - 4))
                                    if i == 1:
                                        t = e1 * (M ** (Dr[k] - 1)) + m * (M ** (Dr[k] - 2)) + \
                                            e2 * (M ** (Dr[k] - 3)) + e3 * (M ** (Dr[k] - 4))
                                    if i == 2:
                                        t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + \
                                            m * (M ** (Dr[k] - 3)) + e3 * (M ** (Dr[k] - 4))
                                    if i == 3:
                                        t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + \
                                            e3 * (M ** (Dr[k] - 3)) + m * (M ** (Dr[k] - 4))
                                    temp = temp + Phi[k][t] * I_v2g[VNode1, FNode1, e1] * I_v2g[VNode2, FNode2, e2] \
                                           * I_v2g[VNode3, FNode3, e3]
                        I_g2v[k][i, m] = temp

                if Dr[k] == 5:
                    # find the other VN connected to this FN
                    if i == 0:
                        VNode1 = Fc[k][1]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                    if i == 1:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                    if i == 2:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                    if i == 3:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                        VNode4 = Fc[k][4]
                    if i == 4:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                        VNode4 = Fc[k][3]
                        # 确定与该资源在与之相连的用户节点连接的资源中是第几个，
                    # 以从I_v2g中取出相应的数值，也让I_g2v可以将计算过后的值更新回去
                    for p in range(Du):
                        if Vc[VNode1, p] == k:
                            FNode1 = p
                        if Vc[VNode2, p] == k:
                            FNode2 = p
                        if Vc[VNode3, p] == k:
                            FNode3 = p
                        if Vc[VNode4, p] == k:
                            FNode4 = p

                    for m in range(M):  # m,e1,e2表示与资源相连的三个用户节点
                        temp = 0
                        for e1 in range(M):  # UE1
                            for e2 in range(M):  # UE2
                                for e3 in range(M):
                                    for e4 in range(M):
                                        if i == 0:
                                            t = m * (M ** (Dr[k] - 1)) + e1 * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3)) + \
                                                e3 * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5))
                                        if i == 1:
                                            t = e1 * (M ** (Dr[k] - 1)) + m * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3)) + \
                                                e3 * (M ** (Dr[k]- 4)) + e4 * (M ** (Dr[k] - 5))
                                        if i == 2:
                                            t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + m * (M ** (Dr[k] - 3)) + \
                                                e3 * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5))
                                        if i == 3:
                                            t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + e3 * (M ** (Dr[k] - 3)) +\
                                                m * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5))
                                        if i == 4:
                                            t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + e3 * (M ** (Dr[k] - 3)) + \
                                                e4 * (M ** (Dr[k] - 4)) + m * (M ** (Dr[k] - 5))
                                        temp = temp + Phi[k][t] * I_v2g[VNode1, FNode1, e1] * I_v2g[VNode2, FNode2, e2] \
                                               * I_v2g[VNode3, FNode3, e3] * I_v2g[VNode4, FNode4, e4]
                        I_g2v[k][i, m] = temp

                if Dr[k] == 6:
                    # find the other VN connected to this FN
                    if i == 0:
                        VNode1 = Fc[k][1]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                        VNode5 = Fc[k][5]
                    if i == 1:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][2]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                        VNode5 = Fc[k][5]
                    if i == 2:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][3]
                        VNode4 = Fc[k][4]
                        VNode5 = Fc[k][5]
                    if i == 3:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                        VNode4 = Fc[k][4]
                        VNode5 = Fc[k][5]
                    if i == 4:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                        VNode4 = Fc[k][3]
                        VNode5 = Fc[k][5]
                    if i == 5:
                        VNode1 = Fc[k][0]
                        VNode2 = Fc[k][1]
                        VNode3 = Fc[k][2]
                        VNode4 = Fc[k][3]
                        VNode5 = Fc[k][4]
                        # 确定与该资源在与之相连的用户节点连接的资源中是第几个，
                    # 以从I_v2g中取出相应的数值，也让I_g2v可以将计算过后的值更新回去
                    for p in range(Du):
                        if Vc[VNode1, p] == k:
                            FNode1 = p
                        if Vc[VNode2, p] == k:
                            FNode2 = p
                        if Vc[VNode3, p] == k:
                            FNode3 = p
                        if Vc[VNode4, p] == k:
                            FNode4 = p
                        if Vc[VNode5, p] == k:
                            FNode5 = p

                    for m in range(M):  # m,e1,e2表示与资源相连的三个用户节点
                        temp = 0
                        for e1 in range(M):  # UE1
                            for e2 in range(M):  # UE2
                                for e3 in range(M):
                                    for e4 in range(M):
                                        for e5 in range(M):
                                            if i == 0:
                                                t = m * (M ** (Dr[k] - 1)) + e1 * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3)) + \
                                                    e3 * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5)) + e5 * (M ** (Dr[k] - 6))
                                            if i == 1:
                                                t = e1 * (M ** (Dr[k] - 1)) + m * (M ** (Dr[k] - 2)) + e2 * (M ** (Dr[k] - 3)) + \
                                                    e3 * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5)) + e5 * (M ** (Dr[k] - 6))
                                            if i == 2:
                                                t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + m * (M ** (Dr[k] - 3)) + \
                                                    e3 * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5)) + e5 * (M ** (Dr[k] - 6))
                                            if i == 3:
                                                t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k]- 2)) + e3 * (M ** (Dr[k] - 3)) + \
                                                    m * (M ** (Dr[k] - 4)) + e4 * (M ** (Dr[k] - 5)) + e5 * (M ** (Dr[k] - 6))
                                            if i == 4:
                                                t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + e3 * (M ** (Dr[k] - 3)) + \
                                                    e4 * (M ** (Dr[k] - 4)) + m * (M ** (Dr[k] - 5)) + e5 * (M ** (Dr[k] - 6))
                                            if i == 5:
                                                t = e1 * (M ** (Dr[k] - 1)) + e2 * (M ** (Dr[k] - 2)) + e3 * (M ** (Dr[k] - 3)) + \
                                                    e4 * (M ** (Dr[k] - 4)) + e5 * (M ** (Dr[k] - 5)) + m * (M ** (Dr[k] - 6))
                                            temp = temp + Phi[k][t] * I_v2g[VNode1, FNode1, e1] * I_v2g[VNode2, FNode2, e2] \
                                                   * I_v2g[VNode3, FNode3, e3] * I_v2g[VNode4, FNode4, e4] * I_v2g[VNode5, FNode5, e5]
                        I_g2v[k][i, m] = temp

        # VN update
        for j in range(J):
            for i in range(Du):
                sum = 0
                g =int(not i)
                FNode = Vc[j,g]
                for k in range(Dr[FNode]):
                    if Fc[FNode][k] == j:
                        VNode = k

                for m in range(M):
                    unorm[m] = ap[j,m] * I_g2v[FNode][VNode,m]
                    sum = sum + unorm[m]

                for m in range(M):
                    I_v2g[j,i,m] = unorm[m] / sum

        # probability accounts
        for j in range(J):
            found = 0
            flag = 0
            fnode = np.zeros(M,dtype=int)
            vnode = np.zeros(M,dtype=int)
            for k in range(K):
                for i in range(Dr[k]):
                    if Fc[k][i] == j:
                        fnode[flag] = k
                        vnode[flag] = i
                        found = found + 1
                        flag = flag + 1
                if flag == Du:
                    break
            for m in range(M):
                Q_post[j, m] = ap[j, m] * I_g2v[fnode[0]][vnode[0],m] * I_g2v[fnode[1]][vnode[1],m]
                if (abs(Q_post[j, m] - Q_prev[j, m]))>TH:
                    Q_prev[j, m] = Q_post[j, m]
                else:
                    counter = counter + 1

        if counter == M * J:
            break

    # decode
    for j in range(J):
        if Q_post[j,0] > Q_post[j,1] :
            data_estimate[j] = 0
        else:
            data_estimate[j] = 1
        if data[j] != data_estimate[j]:
            BER_TOTAL[0] = BER_TOTAL[0] + 1
            BLER_COUNT[j, 0] = 1


# 计算MPA译码的误比特率
def BER_BLER_MPA():

    BER[0] = BER_TOTAL[0] / ( J_total * N_data)
    BLER[0] = BLER_TOTAL[0] / J_total
    print(Noise_Variance_dB,'\t\t\t',BER[0], '\t\t\t', BLER[0])
    # 写入txt文件
    with open('上行免授权SCMA-MPA误比特率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BER[0]) + '\n')
        w.close()
    with open('上行免授权SCMA-MPA丢包率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BLER[0]) + '\n')
        w.close()

# ML译码
def ML():

    prob_ml = np.zeros(MP)
    for i in range(MP):
        f = 0
        for k in range(K):
            f = f + (abs(RxSymbol[k] - Mpoint[i, k]))** 2
        prob_ml[i] = f
    b = np.where(prob_ml == np.min(prob_ml))
    data_estimate_ml = (Msg[int(b[0]),] - 1) * (-0.5)
    BER_TOTAL[1] = BER_TOTAL[1] + np.sum(data_estimate_ml!=data)
    for j in range(J):
        if data[j] != data_estimate[j]:
            BLER_COUNT[j, 1] = 1



# 计算ML的误比特率
def BER_BLER_ML():

    BER[1] = BER_TOTAL[1] / (J_total * N_data)
    BLER[1] = BLER_TOTAL[1] / J_total
    print(Noise_Variance_dB, '\t\t\t', BER[1],'\t\t\t',BLER[1])
    # 写入txt文件
    with open('上行免授权SCMA-ML误比特率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BER[1]) + '\n')
        w.close()
    with open('上行免授权SCMA-ML丢包率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BLER[1]) + '\n')
        w.close()

'''
主程序入口
'''

Initialize()
for i in range(len(EbN0_dB)):

    BER_TOTAL = np.zeros(2)           # MPA、ML误比特率
    BLER_TOTAL = np.zeros(2)          # MPA、ML的丢包数
    BLER = np.zeros(2)                # MPA、ML的丢包数
    LOOP = 0
    Noise_Variance_dB = EbN0_dB[i]
    Noise_Variance = variance[i]
    J_total = 0

    while LOOP < N_pilot:
        LOOP = LOOP + 1
        # 结合导频检测的生成矩阵

        # 随机生成矩阵，用于导频检测无误差时的误比特率检测
        FF = Build_F()
        # 规则SCMA(6,4,3)的测试
        # FF = np.array([[1, 1, 1, 0, 0, 0],
        #       [1, 0, 0, 1, 1, 0],
        #       [0, 1, 0, 1, 0, 1],
        #       [0, 0, 1, 0, 1, 1]])
        if np.sum(FF) == 0:
            N_pilot = N_pilot + 1
            continue
        Variable(FF)
        # print(F)
        J_total = J_total + J
        SubConstellation()
        MasterConstellation()

        loop = 0

        # 丢包率指针
        BLER_COUNT = np.zeros([J, 2])
        while loop<N_data:
            loop = loop + 1
            DataandModu()
            PhaseRotation()
            Receiver()
            MPA()
            ML()
        for j in range(J):
            BLER_TOTAL[0] = BLER_TOTAL[0] + BLER_COUNT[j, 0]
            BLER_TOTAL[1] = BLER_TOTAL[1] + BLER_COUNT[j, 1]
            if BLER_COUNT[j, 1]==1:
                print(BLER_COUNT[j, 1])
    BER_BLER_MPA()
    BER_BLER_ML()

# 计时工具
# time_start = time.time()
# time_end = time.time()
# print('MPA time cost', time_end - time_start, 's')
