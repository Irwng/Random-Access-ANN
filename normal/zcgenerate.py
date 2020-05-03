''''
Purpose : 使用ZC序列导频列表，产生星座图组合
Data : 2019/5/21-2019/5/22
备注：序列长度必须为质数，N=11, user_number = 6
'''
import numpy as np
from math import*

JJ = 1j

def zcseq(length):

    b = np.array(np.zeros(length), dtype=complex)
    for i in range(len(b)):
        b[i] = cos(pi * i * (i + 1) / len(b)) - JJ * sin(pi * i * (i + 1) / len(b))
    # 定义导频阵列矩阵

    seq_mat = np.array(np.zeros([6, len(b)]), dtype=complex)
    Cv = 15  # 偏移量
    for ii in range(6):
        for i in range(len(b)):  # 首先按序生成每一个序列中的数
            seq_mat[ii, i] = b[(Cv * ii + i) % len(b)]

    return seq_mat

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

def generate_standard():  # 每个Eb/N0的基础重复数量

    seq_mat = seq_buff()
    # seq_mat = zcseq(11)

    pilot_length = len(seq_mat[0,:])
    user_number  = len(seq_mat[:,0])
    image_matrix = np.zeros(pilot_length)
    label_matrix = np.zeros(user_number)

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
        pilots = np.zeros([pilot_length])
        for jj in range(user_number):
            pilots = pilots + seq_mat[jj,] * j[jj]

        image_matrix = np.row_stack((image_matrix, pilots))
        label_matrix = np.row_stack((label_matrix, j))

    return image_matrix[1:2**user_number+1], label_matrix[1:2**user_number+1]

# 验证用户组合是否有重复数据
def validation():
    image,label = generate_standard()
    # print(st.shape,labelmat.shape)
    print(image.shape)
    print(label.shape)
    count = 0

    for i in range(len(image[:, 1])-1):
        for j in range(len(image[:, 1])-1-i):
            if (image[i, :] == image[i+j+1, :]).all():
                count = count + 1
    print(count)

if __name__=='__main__':
    np.set_printoptions(precision=5, suppress=True)
    starmat,labelmat = generate_standard()
    print(starmat.shape,labelmat.shape)
    print(starmat)
    print(labelmat)
