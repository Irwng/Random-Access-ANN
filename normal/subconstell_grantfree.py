''''
Purpose : 在免授权场景下，探讨多个旋转算子对应子星座图的
Data : 2019/5/26
备注：user_number = 6，旋转因子小于等于pi/7时，子星座图无重合
'''

import numpy as np
from math import*
import matplotlib.pyplot as plt

JJ = 1j


def seq_buff():

    RF = pi/7
    seq_mat = np.zeros(6, dtype=complex)
    for i in range(len(seq_mat)):
        seq_mat[i] = cos(RF * i) + JJ * sin(RF * i)
    return seq_mat


def generate_standard():  # 每个Eb/N0的基础重复数量

    seq_mat = seq_buff()

    data_length = 1
    user_number = 6
    image_matrix = np.zeros(1, dtype=complex)
    label_matrix = np.zeros(user_number, dtype=complex)

    for i in range(2 ** user_number):
        # i = 2**length-i-1
        # 将值为0~2**length的十进制数转为二进制，之后去掉‘0b’的开头，并将二进制转为列表，此时列表内为字符
        j = list(bin(i).split('b')[1])
        # 将不满长度的列表部分补0`
        add = list(np.zeros([user_number - len(j)], dtype=int))
        add.extend(j)
        # 选择向量：将列表转为数组并将数据类型转为int，至此生成选择向量，0表示该沉默用户，1代表活跃用户
        j = np.array(add).astype(np.int)

        # 通过选择向量选择可能的导频序列进行叠加，即接受端可能收到的信号，共2**length种
        pilots = np.zeros(data_length)
        for jj in range(user_number):
            pilots = pilots + seq_mat[jj] * j[jj]

        image_matrix = np.row_stack((image_matrix, pilots))
        label_matrix = np.row_stack((label_matrix, j))

    return image_matrix[1:2**user_number+1], label_matrix[1:2**user_number+1]

# 验证用户组合是否有重复数据
def validation():
    image,label = generate_standard()
    print(image.shape)
    print(label.shape)
    count = 0

    for i in range(len(image)-1):
        for j in range(len(image)-1-i):
            if ((np.round(image[i].real, 2) == np.round(image[i+j+1].real, 2))
                &(np.round(image[i].imag, 2) ==np.round(image[i+j+1].imag,2))):
                count = count + 1
    print(count)


if __name__=='__main__':

    validation()
    image , label= generate_standard()
    plt.scatter(image.real, image.imag)
    plt.show()

