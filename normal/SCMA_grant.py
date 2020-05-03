'''
Purpose : SCMA in granted system
Date： 2019/5/24
'''

import numpy as np
from math import*
import matplotlib.pyplot as plt

'''
参数定义
'''

N_Loop = 10**5   # Simulation loop number
loop = 0
L = 1              # BPSK bit per symbol
J = 6              # Number of users
K = 4              # Number of resources
Du = 2             # Number of resources connected to user
Dr = 3             # Number of users connected to resources
M = 2              # BPSK modulation order
MP = 64	           # Master-constellation Points
SC = 8             # Sub-constellation Points
JJ = 1j            # complex index

# Signal constellation
RF = pi/5                                         # Rotation Factor
a0 = 1
a1 = cos(RF) + JJ * sin(RF)
a2 = cos(2 * RF) + JJ * sin(2 * RF)
F = np.array(
    [[a0, a1, a2, 0, 0, 0],
     [a1, 0, 0, a2, a0, 0],
     [0, a2, 0, a0, 0, a1],
     [0, 0, a0, 0, a1, a2]], dtype=complex)       # Generate Matrix

'''
变量定义
'''

EbN0_dB = np.arange(11)
Eb_N0 = 10 ** (0.1 * EbN0_dB)
variance = 10 ** (-0.1 * EbN0_dB)  # variance of noise
Noise_Variance = 0
data = np.zeros(J)              # transmitter binary data
data_estimate = np.zeros(J)     # decode binary sequence
Symbol = np.zeros(J, dtype=complex)
TxSymbol = np.zeros(K, dtype=complex)
RxSymbol = np.zeros(K, dtype=complex)

Decode= np.zeros(J)
BER = np.zeros(2)
Msg = np.zeros([MP,J],dtype=complex) # Master-constellation Points mapping on user
Mpoint = np.zeros([MP,K],dtype=complex) # Master-constellation Points mapping on resources
SubMsg = np.zeros([SC,Dr],dtype=complex) # Sub-constellation Points mapping on user
Spoint = np.zeros([K,SC],dtype=complex) # Sub-constellation Points mapping on resources

def MasterConstellation():

    for i in range(MP):
        for j in range(J):
            a = 2 ** (j + 1)
            b = 2 ** j
            if i % a >= b:
                Msg[i, J - j - 1] = -1
            else:
                Msg[i, J - j - 1] =  1

    for i in range(MP):
        for j in range(J):
            r = Msg[i, j]
            for k in range(K):
                Mpoint[i,k] = Mpoint[i,k] + r * F[k, j]

def SubConstellation():

    for i in range(SC):
        for j in range(Dr):
            a = 2 ** (j + 1)
            b = 2 ** j
            if i % a >= b:
                SubMsg[i, Dr - j - 1] = -1
            else:
                SubMsg[i, Dr - j - 1] =  1

    for k in range(K):
        for i in range(SC):
            flag = 0
            for index in range(J):
                if F[k,index]!=0 + 0*J:
                    Spoint[k,i] = Spoint[k,i] + F[k,index] * SubMsg[i,flag]
                    flag = flag + 1

def Initialize():

    print("Eb/N0\t\t\t   BER\n")
    with open('上行授权SCMA-MPA误比特率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:长度为' + str(N_Loop) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('误比特率：' + '\n')
        w.close()
    with open('上行授权SCMA-ML误比特率.txt', mode='a') as w:
        w.write(".................分割线................" + '\n')
        w.write('条件:长度为' + str(N_Loop) + '\n' + 'Eb/N0为' + str(EbN0_dB) + '\n' + '旋转因子：' + str(RF)+ '\n')
        w.write('误比特率：' + '\n')
        w.close()
    MasterConstellation()
    SubConstellation()

def DataandModu():

    for j in range(J):
        data[j] = np.random.randint(0,2)
        Symbol[j] = (data[j]) * (-2) + 1

def PhaseRotation():

    for k in range(K):
        TxSymbol[k] = np.dot(F[k, :], Symbol)

def Receiver():

    for k in range(K):
        noise_phase = np.random.rand(1)
        G = sqrt(-2 * log(np.random.rand(1)))
        AWGN_I = sqrt(Noise_Variance) * G * cos(2 * pi * noise_phase)
        AWGN_Q = sqrt(Noise_Variance) * G * sin(2 * pi * noise_phase)
        RxSymbol[k] = TxSymbol[k] + AWGN_I + JJ * AWGN_Q

def MPA():

    Imax = 20
    Vc = np.array([[0, 1],[0, 2],[0, 3],
                   [1, 2],[1, 3],[2, 3]])  # function nodes connected with variable
    Fc =np.array( [[0, 1, 2], [0, 3, 4],
                   [1, 3, 5], [2, 4, 5]])  # variable nodes connected with function
    TH = 0.0001                  # threshold value
    unorm = np.zeros(M)          # u-normalization probability
    unorm0 = np.zeros(SC)        # u-normalization probability
    ap = np.zeros([J, M])        # prior probability
    Phi = np.zeros([K, SC])      # (int)pow(M,Dr)=8
    I_g2v = np.zeros([K, Dr, M]) # message passing from FN to its neighboring VNs
    I_v2g = np.zeros([J, Du, M]) # message passing from VN to its neighboring FNs
    Q_prev = np.zeros([J, M])    # final probability before iterative
    Q_post = np.zeros([J, M])    # final probability after iterative

    # prior probability
    for i in range(J):
        for j in range(M):
            ap[i,j] = 1/M

    # sub starpoints
    for k in range(K):
        sum = 0
        for i in range(SC):
            f = -2*abs(RxSymbol[k] - Spoint[k, i]) ** 2 / Noise_Variance
            unorm0[i] = exp(f)
            sum = sum + unorm0[i]

        for i in range(SC):
            Phi[k,i] = unorm0[i]/sum

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
            for i in range(Dr):
                # find the other VN connected to this FN
                if i == 0:
                    VNode1 = Fc[k, 1]
                    VNode2 = Fc[k, 2]
                if i == 1:
                    VNode1 = Fc[k, 0]
                    VNode2 = Fc[k, 2]
                if i == 2:
                    VNode1 = Fc[k, 0]
                    VNode2 = Fc[k, 1]
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
                                t = m * (M ** (Dr - 1)) + e1 * (M ** (Dr - 2)) + e2 * (M ** (Dr - 3))
                            if i == 1:
                                t = e1 * (M ** (Dr - 1)) + m * (M ** (Dr - 2)) + e2 * (M ** (Dr - 3))
                            if i == 2:
                                t = e1 * (M ** (Dr - 1)) + e2 * (M ** (Dr - 2)) + m * (M ** (Dr - 3))
                            temp = temp + Phi[k,t] * I_v2g[VNode1,FNode1,e1] * I_v2g[VNode2,FNode2,e2]
                    I_g2v[k,i,m] = temp

        # VN update
        for j in range(J):
            for i in range(Du):
                sum = 0
                g =int(not i)
                FNode = Vc[j,g]

                for k in range(Dr):
                    if Fc[FNode,k] == j:
                        VNode = k

                for m in range(M):
                    unorm[m] = ap[j,m] * I_g2v[FNode,VNode,m]
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
                for i in range(Dr):
                    if Fc[k, i] == j:
                        fnode[flag] = k
                        vnode[flag] = i
                        found = found + 1
                        flag = flag + 1
                if flag == Du:
                    break
            for m in range(M):
                Q_post[j, m] = ap[j, m] * I_g2v[fnode[0],vnode[0],m] * I_g2v[fnode[1],vnode[1],m]
                if (abs(Q_post[j, m] - Q_prev[j, m]))>TH:
                    Q_prev[j, m] = Q_post[j, m]
                else:
                    counter = counter + 1

        if counter == M * J:
            break

    # decode
    for i in range(J):
        if Q_post[i,0] > Q_post[i,1] :
            data_estimate[i] = 0
        else:
            data_estimate[i] = 1
        if data[i] != data_estimate[i]:
            BER_TOTAL[0] = BER_TOTAL[0] + 1

# Error Bit Rate Calculation
def BER_MPA():
    BER[0] = BER_TOTAL[0] / (J * N_Loop)
    print(Noise_Variance_dB,'\t\t\t',BER[0])
    # 写入txt文件
    with open('上行授权SCMA-MPA误比特率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BER) + '\n')
        w.close()

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

# 计算ML的误比特率+
def BER_ML():

    BER[1] = BER_TOTAL[1] / (J * N_Loop)
    print(Noise_Variance_dB, '\t\t\t', BER[1])
    # 写入txt文件
    with open('上行授权SCMA-ML误比特率.txt', mode='a') as w:
        w.write(str(Noise_Variance_dB) + '\t\t\t\t' + str(BER) + '\n')
        w.close()

# main()
Initialize()

for i in range(len(EbN0_dB)):

    BER_TOTAL = np.zeros(2)
    Noise_Variance_dB = EbN0_dB[i]
    Noise_Variance = variance[i]
    loop = 0
    while loop<N_Loop:
        loop = loop + 1
        DataandModu()
        PhaseRotation()
        Receiver()
        MPA()
        ML()
    BER_MPA()
    BER_ML()


