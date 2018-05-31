# -*- coding:utf8 -*-
import tensorflow as tf

run_metadata = tf.RunMetadata()

# # 计算整个网络的参数量
# # 利用 tf.trainable_variables
# def count_model_params():
#     total_parameters = 0
#     for variable in tf.trainable_variables():
#         # shape is an array of tf.Dimension
#         shape = variable.get_shape()
#         variable_parameters = 1
#         for dim in shape:
#             variable_parameters *= dim.value
#         total_parameters += variable_parameters
#     print('  + Number of params: %.2fM' % (total_parameters / 1e6))
    
# def print_model_params():
#     import numpy as np
#     num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
#     print('  + Number of params: %.2fM' % (num_params / 1e6))

with tf.Session(graph=tf.Graph()) as sess:

    opt = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

    opt = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    param_count = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opt)

    print('  + Number of FLOPs: %.4fG' % (flops.total_float_ops / 1e9))
    print('  + Number of params: %.4fM' % (param_count.total_parameters / 1e6))
