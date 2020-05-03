import tensorflow as tf

INPUT_NODE = 88  # 输入节点，导频长度是7，所以这里的输入层神经元个数是7
OUTPUT_NODE = 24  # 输出节点，通过每个节点的的值判断是否有活跃用户，当value>0.5时，判断为用户活跃，
                 # 反之不活跃，所以输出层神经元个数为潜在用户数
LAYER1_NODE = 500  # 隐藏层1的神经元个数
LAYER2_NODE = 500  # 隐藏层2的神经元个数
LAYER3_NODE = 500  # 隐藏层2的神经元个数


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    with tf.variable_scope('layer3'):
        weights = get_weight_variable([LAYER2_NODE, LAYER3_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER3_NODE], initializer=tf.constant_initializer(0.0))
        layer3 = tf.matmul(layer2, weights) + biases

    with tf.variable_scope('layer4'):
        weights = get_weight_variable([LAYER3_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer4 = tf.matmul(layer3, weights) + biases

    return layer4
