''''
Purpose : 使用m序列导频列表，产生星座图组合
Data : 2019/5/29
备注：序列长度必须为质数，N=7, user_number = 6
'''
import numpy as np

def mseq(): #导入参数为反馈系数

    m = len(feedback_index)
    global length
    length = 2**m-1 #生成序列的长度
    seq = np.zeros((length,),np.int)
    registers = np.ones((m,),np.int)

    for i in range(length):
        seq[i] = registers[m-1]
        # print(seq[i])
        backdata = np.dot(feedback_index,registers)%2
        # print(backdata)
        registers[1:m] = registers[0:(m-1)]
        registers[0] = backdata
        # print(registers)
    # 二进制调制的m序列根序列
    sequence = 1 - 2*seq
    # 定义导频阵列矩阵
    sequence_mat = np.array(np.zeros([length, length]))

    # 通过循环移位产生所有导频序列,构成导频阵列矩阵
    for i in range(length):
        sequence_mat[i, i:length] = sequence[0:length - i]
        sequence_mat[i, 0:i] = sequence[length - i:length]
    # print(sequence_mat)
    return sequence_mat[1:]


def seq_buff():
    seq_mat = np.array([[1, -1, -1, -1, 1, 1, -1],
               [-1, -1, -1, 1, 1, -1 ,1],
               [-1, -1, 1, 1, -1 ,1, -1],
               [-1, 1, 1, -1 ,1, -1, -1],
               [1, 1, -1 ,1, -1, -1, -1],
               [1, -1 ,1, -1, -1, -1, 1]])
    return seq_mat

# 生成星座图
def generate_standard():

    seq_mat = seq_buff()

    pilot_length = seq_mat.shape[1]
    user_number  = seq_mat.shape[0]
    image_matrix = np.zeros(pilot_length)
    label_matrix = np.zeros(user_number)

    for i in range(2 ** user_number):

        # 将值为0~2**length的十进制数转为二进制，之后去掉‘0b’的开头，并将二进制转为列表，此时列表内为字符
        j = list(bin(i).split('b')[1])
        # 将不满长度的列表部分补0`
        add = list(np.zeros([user_number - len(j)], dtype=int))
        add.extend(j)
        # 选择向量：将列表转为数组并将数据类型转为int，至此生成选择向量，0表示该沉默用户，1代表活跃用户
        j = np.array(add).astype(np.int)

        # 通过选择向量选择可能的导频序列进行叠加，即接受端可能收到的信号，共2**length种
        pilots = np.zeros(pilot_length)
        for jj in range(user_number):
            pilots = pilots + seq_mat[jj,]*j[jj]
        image_matrix = np.row_stack((image_matrix, pilots))
        label_matrix = np.row_stack((label_matrix, j))

    return image_matrix[1:2**user_number+1], label_matrix[1:2**user_number+1]


def validation():
    image,label = generate_standard()
    print(image.shape)
    print(label.shape)
    count = 0

    for i in range(len(image[:, 1])-1):
        for j in range(len(image[:, 1])-1-i):
            if (image[i, :] == image[i+j+1, :]).all():
                count = count + 1
    print(count)

if __name__ == '__main__':
    feedback_index = np.array([0, 1, 1])  # 反馈序列
    print(mseq())
