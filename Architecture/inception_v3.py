import tensorflow as tf
import numpy
import tensorflow.contrib.slim as slim


def _variable(name, shape):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape)
  return var

def conv_layer(input,shape,stride,activation=True,padding='VALID',name=None):
    in_channel=shape[2]
    out_channel=shape[3]
    k_size=shape[0]
    with tf.variable_scope(name) as scope:
        kernel=_variable('conv_weights',shape = shape)
        conv=tf.nn.conv2d(input = input,filter = kernel,strides = stride,padding =padding)
        biases=_variable('biases',[out_channel])
        bias=tf.nn.bias_add(conv,biases)
        if activation is True:
            conv_out=tf.nn.relu(bias,name = 'relu')
        else:
            conv_out=bias
        return conv_out

def conv_inception(input, shape, stride= [1,1,1,1], activation = True, padding = 'SAME', name = None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
        kernel = _variable('conv_weights', shape = shape)
        conv = tf.nn.conv2d(input = input, filter = kernel, strides = stride, padding = padding)
        biases = _variable('biases', [out_channel])
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(bias, name = 'relu')
        else:
            conv_out = bias
        return conv_out

def inception_block_tradition(input, name=None):
#传统的inception
    with tf.variable_scope(name) as scope:
        with tf.variable_scope("Branch_0"):
            branch_0=conv_inception(input,shape = [1,1,288,64],name = '0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,288,48],name = '0a_1x1')
            branch_1=conv_inception(branch_1,shape = [5,5,48,64],name = '0b_5x5')
        with tf.variable_scope("Branch_2"):
            branch_2=conv_inception(input,shape = [1,1,288,64],name = '0a_1x1')
            branch_2=conv_inception(branch_2,shape = [3,3,64,96],name = '0b_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3=tf.nn.avg_pool(input,ksize = (1,3,3,1),strides = [1,1,1,1],padding = 'SAME',name = 'Avgpool_0a_3x3')
            branch_3=conv_inception(branch_3,shape = [1,1,288,64],name = '0b_1x1')
        inception_out=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        b=1 # for debug
        return inception_out

def inception_grid_reduction_1(input,name=None):

    with tf.variable_scope(name) as scope:
        with tf.variable_scope("Branch_0"):
            branch_0=conv_inception(input,shape = [1,1,288,384],name = '0a_1x1')
            branch_0=conv_inception(branch_0,shape = [3,3,384,384],stride = [1,2,2,1],padding = 'VALID',name = '0b_3x3')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,288,64],name = '0b_1x1')
            branch_1=conv_inception(branch_1,shape = [3,3,64,96],name = '0b_3x3')
            branch_1=conv_inception(branch_1,shape = [3,3,96,96],stride = [1,2,2,1],padding = 'VALID',name = '0c_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2=tf.nn.max_pool(input,ksize = (1,3,3,1),strides = [1,2,2,1],padding = 'VALID',name = 'maxpool_0a_3x3')
        inception_out=tf.concat([branch_0,branch_1,branch_2],3)
        c=1 # for debug
        return inception_out

def inception_grid_reduction_2(input,name=None):

    with tf.variable_scope(name) as scope:
        with tf.variable_scope("Branch_0"):
            branch_0=conv_inception(input,shape = [1,1,768,192],name = '0a_1x1')
            branch_0=conv_inception(branch_0,shape = [3,3,192,320],stride = [1,2,2,1],padding = 'VALID',name = '0b_3x3')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,768,192],name = '0b_1x1')
            branch_1=conv_inception(branch_1,shape = [3,3,192,192],name = '0b_3x3')
            branch_1=conv_inception(branch_1,shape = [3,3,192,192],stride = [1,2,2,1],padding = 'VALID',name = '0c_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2=tf.nn.max_pool(input,ksize = (1,3,3,1),strides = [1,2,2,1],padding = 'VALID',name = 'maxpool_0a_3x3')
        inception_out=tf.concat([branch_0,branch_1,branch_2],3)
        c=1 # for debug
        return inception_out

def inception_block_factorization(input,name=None):

    with tf.variable_scope(name) as scope:
        with tf.variable_scope('Branch_0'):
            branch_0=conv_inception(input,shape = [1,1,768,192],name = '0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,768,128],name = '0a_1x1')
            branch_1=conv_inception(branch_1,shape = [1,7,128,128],name = '0b_1x7')
            branch_1=conv_inception(branch_1,shape = [7,1,128,128],name = '0c_7x1')
            branch_1=conv_inception(branch_1,shape = [1,7,128,128],name = '0d_1x7')
            branch_1=conv_inception(branch_1,shape = [7,1,128,192],name = '0e_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2=conv_inception(input,shape = [1,1,768,128],name = '0a_1x1')
            branch_2=conv_inception(branch_2,shape = [1,7,128,128],name = '0b_1x7')
            branch_2=conv_inception(branch_2,shape = [7,1,128,192],name = '0c_7x1')
        with tf.variable_scope('Branch_3'):
            branch_3=tf.nn.avg_pool(input,ksize = (1,3,3,1),strides = [1,1,1,1],padding = 'SAME',name = 'Avgpool_0a_3x3')
            branch_3=conv_inception(branch_3,shape = [1,1,768,192],name = '0b_1x1')
        inception_out=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        d=1 # for debug
        return inception_out

def inception_block_expanded_pre(input,name=None):
    with tf.variable_scope(name) as scope:
        with tf.variable_scope('Branch_0'):
            branch_0=conv_inception(input,shape = [1,1,1280,320],name = '0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,1280,448],name = '0a_1x1')
            branch_1=conv_inception(branch_1,shape = [3,3,448,384],name = '0b_3x3')
            branch_1=tf.concat([conv_inception(branch_1,shape = [1,3,384,384],name = '0c_1x3'),
                                conv_inception(branch_1,shape = [3,1,384,384],name = '0d_3x1')],3)
        with tf.variable_scope('Branch_2'):
            branch_2=conv_inception(input,shape = [1,1,1280,384],name = '0a_1x1')
            branch_2=tf.concat([conv_inception(branch_2,shape = [1,3,384,384],name = '0b_1x3'),
                                conv_inception(branch_2,shape = [3,1,384,384],name = '0c_3x1')],3)
        with tf.variable_scope('Branch_3'):
            branch_3=tf.nn.avg_pool(input,ksize = (1,3,3,1),strides = [1,1,1,1],padding = 'SAME',name = 'Avgpool_0a_3x3')
            branch_3=conv_inception(branch_3,shape = [1,1,1280,192],name = '0b_1x1')
        inception_out=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        e=1 # for debug
        return inception_out

def inception_block_expanded(input,name=None):
    with tf.variable_scope(name) as scope:
        with tf.variable_scope('Branch_0'):
            branch_0=conv_inception(input,shape = [1,1,2048,320],name = '0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1=conv_inception(input,shape = [1,1,2048,448],name = '0a_1x1')
            branch_1=conv_inception(branch_1,shape = [3,3,448,384],name = '0b_3x3')
            branch_1=tf.concat([conv_inception(branch_1,shape = [1,3,384,384],name = '0c_1x3'),
                                conv_inception(branch_1,shape = [3,1,384,384],name = '0d_3x1')],3)
        with tf.variable_scope('Branch_2'):
            branch_2=conv_inception(input,shape = [1,1,2048,384],name = '0a_1x1')
            branch_2=tf.concat([conv_inception(branch_2,shape = [1,3,384,384],name = '0b_1x3'),
                                conv_inception(branch_2,shape = [3,1,384,384],name = '0c_3x1')],3)
        with tf.variable_scope('Branch_3'):
            branch_3=tf.nn.avg_pool(input,ksize = (1,3,3,1),strides = [1,1,1,1],padding = 'SAME',name = 'Avgpool_0a_3x3')
            branch_3=conv_inception(branch_3,shape = [1,1,2048,192],name = '0b_1x1')
        inception_out=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        e=1 # for debug
        return inception_out





def inception_v3_base(inputs,
                      final_endpoint='',
                      scope=None):
    end_points={}




    #创建一个变量域
    with tf.variable_scope(scope,'Inception_v3',[inputs]):
        # conv0         input_size=299X299X3  output_size=149X149X32
        conv0=conv_layer(inputs,shape = [3,3,3,32],stride = [1,2,2,1],name = 'conv0')
        endpoint='conv0'
        end_points[endpoint]=conv0

        # conv1         input_size=149X149X32 output_size=147X147X32
        conv1=conv_layer(conv0,shape = [3,3,32,32],stride = [1,1,1,1],name = 'conv1')
        endpoint='conv1'
        end_points[endpoint]=conv1

        # conv2_padded  input_size=147X147X32 output_size=147X147X64
        conv2=conv_layer(conv1,shape = [3,3,32,64],stride = [1,1,1,1],padding ='SAME' ,name = 'conv2')
        endpoint='conv2'
        end_points[endpoint]=conv2

        # pool          input_size=147X147X64 output_size=73X73X64
        pool=tf.nn.max_pool(conv2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID',name = 'pool')
        endpoint='pool'
        end_points[endpoint]=pool

        # conv3         input_size=73X73X64   output_size=71X71X80
        conv3=conv_layer(pool,shape = [3,3,64,80],stride = [1,1,1,1],name = 'conv3')
        endpoint='conv3'
        end_points[endpoint]=conv3

        # conv4         input_size=71X71X80   output_size=35X35X192
        conv4=conv_layer(conv3,shape = [3,3,80,192],stride = [1,2,2,1],name = 'conv4')
        endpoint='conv4'
        end_points[endpoint]=conv4

        # conv5         input_size=35X35X192  output_size=35X35X288
        conv5=conv_layer(conv4,shape = [3,3,192,288],stride = [1,1,1,1],padding = 'SAME',name = 'conv5')
        endpoint='conv5'
        end_points[endpoint]=conv5

        # inception block(tradition one)Mixed_6a input_size=35x35x288 output_size=35x35x288
        Mixed_6a=inception_block_tradition(conv5,name = 'Mixed_6a')
        endpoint='Mixed_6a'
        end_points[endpoint]=Mixed_6a

        # inception block(tradition one)Mixed_6b input_size=output_size
        Mixed_6b=inception_block_tradition(Mixed_6a,name = 'Mixed_6b')
        endpoint = 'Mixed_6b'
        end_points[endpoint] = Mixed_6b

        # inception block(tradition one)Mixed_6c input_size=output_size
        Mixed_6c=inception_block_tradition(Mixed_6b,name = 'Mixed_6c')
        endpoint = 'Mixed_6c'
        end_points[endpoint] = Mixed_6c

        # inception grid reduction
        grid_reduction0=inception_grid_reduction_1(Mixed_6c,name = 'grid_reduction_0')
        endpoint = 'grid_reduction_0'
        end_points[endpoint] = grid_reduction0

        # inception block(factorization)Mixed_7a input_size=output_size=17x17x768
        Mixed_7a=inception_block_factorization(grid_reduction0,name = 'Mixed_7a')
        endpoint = 'Mixed_7a'
        end_points[endpoint] = Mixed_7a

        # inception block(factorization)Mixed_7b
        Mixed_7b=inception_block_factorization(Mixed_7a,name = 'Mixed_7b')
        endpoint = 'Mixed_7b'
        end_points[endpoint] = Mixed_7b

        # inception block(factorization)Mixed_7c
        Mixed_7c=inception_block_factorization(Mixed_7b,name = 'Mixed_7c')
        endpoint = 'Mixed_7c'
        end_points[endpoint] = Mixed_7c

        # inception block(factorization)Mixed_7d
        Mixed_7d=inception_block_factorization(Mixed_7c,name = 'Mixed_7d')
        endpoint = 'Mixed_7d'
        end_points[endpoint] = Mixed_7d

        # inception block(factorization)Mixed_7d
        Mixed_7e=inception_block_factorization(Mixed_7d,name = 'Mixed_7e')
        endpoint = 'Mixed_7e'
        end_points[endpoint] = Mixed_7e

        # inception grid reduction input_size=17x17x768 output_size=8x8x1280
        grid_reduction1=inception_grid_reduction_2(Mixed_7e,name = 'grid_reduction_1')
        endpoint = 'grid_reduction_1'
        end_points[endpoint] = grid_reduction1

        # inception block(expanded)Mixed_8a
        Mixed_8a=inception_block_expanded_pre(grid_reduction1,name = 'Mixed_8a')
        endpoint = 'Mixed_8a'
        end_points[endpoint] = Mixed_8a

        # inception block(expanded)Mixed_8b
        Mixed_8b=inception_block_expanded(Mixed_8a,name = 'Mixed_8b')
        endpoint = 'Mixed_8b'
        end_points[endpoint] = Mixed_8b
        a=1;
        return Mixed_8b,end_points





inputs=tf.get_variable('inputs',shape = [1,299,299,3])

inception_v3_base(inputs)

def inception_v3(inputs,
                 num_classes=20,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='Inception_v3'):

    with tf.variable_scope(scope,'Inception_v3',[inputs,num_classes],reuse = reuse) as scope:
        net, end_points=inception_v3_base(inputs,scope = scope)


        # Auxiliary
        aux_logits=end_points['Mixed_7e']
        with tf.variable_scope('AuxLogits'):
            aux_logits=tf.nn.avg_pool(aux_logits,ksize = (5,5),strides = [1,3,3,1],padding = 'VALID',name = 'Avgpool_1a_5x5')
            aux_logits=conv_inception(aux_logits,shape = [1,1,768,128],name = 'conv_1b_1x1')
            aux_logits=conv_layer(aux_logits,shape = [5,5,128,768],stride = [1,1,1,1],name = 'conv_2a_5x5')
            aux_logits=conv_inception(aux_logits,shape = [1,1,768,num_classes],activation = False ,name = 'con_2b_1x1')
            end_points['AuxLogits']= aux_logits

        with tf.variable_scope('Logits'):
            net=tf.nn.avg_pool(net,ksize = [8,8,2048,2048],strides = [1,1,1,1],padding = 'VALID',name = 'Avgpool_1a_8x8')

            # 1x1x2048
            net=tf.nn.dropout(net,keep_prob = dropout_keep_prob,name = 'Dropout_1b')
            end_points['PreLogits']=net
            # 2048
            logits=conv_inception(net,shape = [1,1,2048,num_classes],activation = False,name = 'conv_1c_1x1')

            end_points['Logits']=logits
            end_points['Predictions']=tf.nn.softmax(logits,name = 'Predictions')
            return logits,end_points










