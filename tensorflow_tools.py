# 计算整个网络的参数量
# 利用 tf.trainable_variables
def count_model_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('  + Number of params: %.2fM' % (total_parameters / 1e6))
    
def print_model_params():
    import numpy as np
    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('  + Number of params: %.2fM' % (num_params / 1e6))
    
