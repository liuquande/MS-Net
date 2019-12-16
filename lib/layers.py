import tensorflow as tf
import tensorflow.contrib.slim as slim

def weight_variable(shape, stddev=0.1, trainable=True, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)

    return tf.get_variable(name = name, shape = shape, trainable = trainable, initializer = tf.truncated_normal_initializer(stddev = stddev))
    #return tf.Variable(initial, trainable=trainable, name=name)


def sharable_weight_variable(shape, stddev=0.1, trainable = True, name = None):
    """
    sharable through variable scope reuse
    """
    return tf.get_variable(name = name, shape = shape, trainable = trainable, initializer = tf.truncated_normal_initializer(stddev = stddev))


def max_pool2d(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def ave_pool2d(x, n):
    return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def conv2d(x, k, c_o, keep_prob_=1, stride=1, trainable=True, name=None):
    """
    :param x: input to the layer
    :param k: kernel size
    :param c_o: output channel
    :param keep_prob_: keep rate for dropout
    :return: convolution results with dropout
    """
    c_i = x.get_shape().as_list()[-1]
    c_o = int(c_o)
    #print k.dtype
    #print c_o.dtype
    w = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=name)

    conv_2d = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding='SAME')
    return conv_2d
    #return tf.nn.dropout(conv_2d, keep_prob_)

def depthwise_conv2d(x, k, c_o, keep_prob_=1, stride=1, trainable=True, name=None):
    """
    :param x: input to the layer
    :param k: kernel size
    :param c_o: output channel
    :param keep_prob_: keep rate for dropout
    :return: convolution results with dropout
    """
    c_i = x.get_shape().as_list()[-1]
    c_o = int(c_o)
    #print k.dtype
    #print c_o.dtype
    w = weight_variable(shape=[k, k, 1, c_i], trainable=trainable, name=name)
    conv_2d = tf.nn.depthwise_conv2d(x, w, strides=[1,stride,stride,1], padding='SAME')
    
    return net

def bn_relu_conv2d_adapter(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')
    
    w_d = weight_variable(shape=[1, 1, c_i, c_i], trainable=trainable, name=scope+'/'+bn_scope +'_kernel')

    conv2d_layer = tf.nn.conv2d(x, w_d, strides=[1,1,1,1], padding='SAME')
    bn_layer = batch_norm(conv2d_layer, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')
    #
    return tf.nn.dropout(conv2d_layer, keep_prob_)

def bn_relu_conv2d(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')
    bn_layer = batch_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')
    #
    return tf.nn.dropout(conv2d_layer, keep_prob_)

def bn_relu_depth_conv2d(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_i, 1], trainable=trainable, name=scope+'_kernel')
    bn_layer = batch_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.depthwise_conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')
    #
    return tf.nn.dropout(conv2d_layer, keep_prob_)

def in_relu_conv2d(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')
    bn_layer = instance_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')
    #
    return tf.nn.dropout(conv2d_layer, keep_prob_)


def bn_relu_conv2d_series(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w_1 = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')
    w_2 = weight_variable(shape=[1, 1, c_i, c_o], trainable=trainable, name=scope+'/'+bn_scope+'_kernel')
    bn_layer = batch_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w_1, strides=[1,stride,stride,1], padding='SAME')


    #bn_layer_2 = batch_norm(conv2d_layer, is_training=is_training, scope=scope+'/'+bn_scope + '_series', trainable=trainable)
    bn_layer_2 = batch_norm(conv2d_layer, is_training=is_training, scope=scope+'/'+bn_scope+'2', trainable=trainable)
    conv2d_domain = tf.nn.conv2d(bn_layer_2, w_2, strides=[1,1,1,1], padding='SAME')

    conv2d = conv2d_layer + conv2d_domain
    #
    return tf.nn.dropout(conv2d, keep_prob_)

def bn_relu_conv2d_parallel(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w_1 = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')
    w_2 = weight_variable(shape=[1, 1, c_i, c_o], trainable=trainable, name=scope+'/'+bn_scope+'_kernel')
    bn_layer = batch_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable=trainable)
    relu_layer = tf.nn.relu(bn_layer)
    conv2d_layer = tf.nn.conv2d(relu_layer, w_1, strides=[1,stride,stride,1], padding='SAME')
    conv2d_domain = tf.nn.conv2d(relu_layer, w_2, strides=[1,1,1,1], padding='SAME')

    conv2d = conv2d_layer + conv2d_domain
    #
    return tf.nn.dropout(conv2d, keep_prob_)

def bn_leaky_relu_conv2d_layer(x, k, c_o, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):
    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_i, c_o], trainable=trainable, name=scope+'_kernel')

    #bn_layer = batch_norm(x, is_training = is_training, scope = scope+'_bn', trainable = trainable)
    relu_layer = tf.nn.leaky_relu(x, 0.2)
    conv2d_layer = tf.nn.conv2d(relu_layer, w, strides=[1,stride,stride,1], padding='SAME')

    return tf.nn.dropout(conv2d_layer, keep_prob_)

def bn_relu_deconv2d(x, k, c_o, output_shape, keep_prob_, stride=1, is_training=True, scope='', trainable=True, bn_scope=''):

    c_i = x.get_shape().as_list()[-1]
    w = weight_variable(shape=[k, k, c_o, c_i], trainable=trainable, name=scope+'_kernel')

    bn_layer = batch_norm(x, is_training=is_training, scope=scope+'/'+bn_scope, trainable = trainable)
    relu_layer = tf.nn.relu(bn_layer)
    deconv2d_layer = tf.nn.conv2d_transpose(relu_layer, w, output_shape, strides=[1,stride,stride,1], padding='SAME')

    return tf.nn.dropout(deconv2d_layer, keep_prob_)

def batch_norm(x, is_training, scope = None, trainable = True):
    # Important: set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
    return tf.contrib.layers.batch_norm(x, is_training = is_training, decay = 0.9, scale = True, center = True, \
                                        scope = scope, variables_collections = ["internal_batchnorm_variables"], \
                                        updates_collections = None, trainable = trainable)

def instance_norm(x, is_training, scope = None, trainable = True):
    # Important: set updates_collections=None to force the updates in place, but that can have a speed penalty, especially in distributed settings.
    return tf.contrib.layers.instance_norm(x, scale = True, center = True, \
                                        scope = scope, variables_collections = ["internal_batchnorm_variables"], trainable = trainable)

def transition_layer(x, keep_prob_, stride=1, layers_num=5, is_training=True, scope='', bn_scope=''):
    """
        Args:
        is_train: whether the mode is training, for setting of the bn layer, moving_mean and moving_variance
        param: scope: setting for batch_norm variables
    """
    c_o = x.get_shape().as_list()[-1]

    x = bn_relu_conv2d(x, 1, int(0.5 * c_o), keep_prob_, is_training=is_training, scope=scope + '_conv1', bn_scope=bn_scope)
    x = ave_pool2d(x, 2)

    return x

def bottleneck_layer(x, c_o, keep_prob_, stride=1, is_training=True, scope='', bn_scope=''):
    x = bn_relu_conv2d(x, 1, 4 * c_o, keep_prob_, is_training=is_training, scope=scope + '_conv1', bn_scope=bn_scope)
    x = bn_relu_conv2d(x, 3, c_o, keep_prob_, is_training=is_training, scope=scope + '_conv2', bn_scope=bn_scope)

    return x

def dense_block(x, _, c_o, keep_prob_, stride=1, layers_num=4, is_training=True, scope='', bn_scope=''):
    """
        Args:
        is_train: whether the mode is training, for setting of the bn layer, moving_mean and moving_variance
        param: scope: setting for batch_norm variables
    """

    _loc_scope_ = range(layers_num)
    _inner_conv_ = range(layers_num)

    for i in range(layers_num):
        if scope is None:
                _loc_scope_[i] = None
        else:
            _loc_scope_[i] = scope + "layer_" + str(i)

    for i in range(layers_num):
        _inner_conv_[i] = bottleneck_layer(x, c_o, keep_prob_, is_training=is_training, scope=_loc_scope_[i], bn_scope=bn_scope)
        x = concat2d(x, _inner_conv_[i])
    return x


def res_block(x, k, c_o, keep_prob_, stride=1, inc_dim = False, layers_num=2, is_training=True, trainable=True, scope='', bn_scope=''):
    """
        Args:
        is_train: whether the mode is training, for setting of the bn layer, moving_mean and moving_variance
        param: scope: setting for batch_norm variables
    """

    _loc_scope_ = range(layers_num)
    _inner_conv_ = range(layers_num)

    for i in range(layers_num):
        if scope is None:
                _loc_scope_[i] = None
        else:
            _loc_scope_[i] = scope + "layer_" + str(i)

    _inner_conv_[0] = bn_relu_conv2d(x, k, c_o, keep_prob_, is_training=is_training, scope=_loc_scope_[0], trainable=trainable, bn_scope=bn_scope)
    _inner_conv_[1] = bn_relu_conv2d(_inner_conv_[0], k, c_o, keep_prob_, is_training=is_training, scope=_loc_scope_[1], trainable=trainable, bn_scope=bn_scope)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x
    
    return x_s + _inner_conv_[1]

def global_ave_pool2d(x):
    return tf.reduce_mean(x, [1, 2], keep_dims=True)


def fully_connected(x, units, layer_name='fully_connected') :
    return tf.layers.dense(inputs=x, use_bias=True, units=units, name=layer_name)

def domain_adapter(x, is_training=True, trainable=True, bank_num=3, scope=''):
    _inner_scale = range(bank_num)
    _inner_ave = global_ave_pool2d(x)
    for i in range(bank_num):
        _inner_scale[i] = fully_connected(_inner_ave, tf.divide(_inner_ave.get_shape().as_list()[3],16), layer_name=scope+'_fc_'+str(i)+'_1')
        _inner_scale[i] = tf.nn.relu(_inner_scale[i])
        _inner_scale[i] = fully_connected(_inner_scale[i], _inner_ave.get_shape().as_list()[3], layer_name=scope+'_fc_'+str(i)+'_2')
    concat = tf.concat([_inner_scale[0], _inner_scale[1]], 2)
    concat = tf.concat([concat, _inner_scale[2]], 2)
    attention = fully_connected(_inner_ave, bank_num, layer_name=scope+'fc_attention')
    attention = tf.nn.softmax(attention)
    attention = tf.transpose(attention, perm=[0, 1, 3, 2])
    scale = attention * concat
    scale = tf.reduce_sum(scale, axis=2, keepdims=True)
    scale = tf.sigmoid(scale)

    output = x * scale
    return  output, scale 


def res_block_leaky(x, k, c_o, keep_prob_, stride=1, inc_dim = False, layers_num=2, is_training=True, trainable=True, scope=''):
    """Args:
        adapt_scope: a flag indicating the variable scope for batch_norm
        what else can i do? tensorflow sucks!
    param: scope: setting for batch_norm variables
    """
    _loc_scope_ = range(layers_num)
    _inner_conv_ = range(layers_num)

    for i in range(layers_num):
        if scope is None:
                _loc_scope_[i] = None
        else:
            _loc_scope_[i] = scope + "layer_" + str(i)

    _inner_conv_[0] = bn_relu_conv2d(x, k, c_o, keep_prob_, is_training=is_training, scope=_loc_scope_[0], trainable=trainable)
    _inner_conv_[1] = bn_relu_conv2d(_inner_conv_[0], k, c_o, keep_prob_, is_training=is_training, scope=_loc_scope_[1], trainable=trainable)

    if inc_dim is True:
        # pooled_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_s = tf.pad(x, [ [0,0], [0,0], [0,0], [_x_channel // 2, _x_channel // 2]])
    else:
        x_s = x

    return x_s + _inner_conv_[1]    

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")


def concat2d(x1,x2):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 3)

def sum2d(x1,x2):
    """ sum two tensors"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.reduce_sum([x1, x2], 0)

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1,  tf.shape(output_map)[3]]))
    return tf.clip_by_value( tf.div(exponential_map,tensor_sum_exp), -1.0 * 1e15, 1.0* 1e15, name = "pixel_softmax_2d")
    
def _phase_shift(I, r, batch_size):
    # Helper function with main phase shift operation

    _, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (batch_size, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    if batch_size == 1:
        X = tf.expand_dims( X, 0 )
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    if batch_size == 1:
        X = tf.concat([x for x in X], 2 )
    else:
        X = tf.concat([tf.squeeze(x) for x in X], 2)  #
    out =  tf.reshape(X, (batch_size, a*r, b*r, 1))
    if batch_size == 1:
        out = tf.transpose( out, (0,2,1,3)  )
    return out

def PS(X, r, batch_size, n_channel = 8):
  # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, n_channel, -1 )
    X = tf.concat([_phase_shift(x, r, batch_size) for x in Xc], 3)
    return X
