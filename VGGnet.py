import tensorflow as tf
import math
import time
from datetime import datetime


num_batches=100
batch_size=64

def conv_op(input_op,name,ksize_h,ksize_w,n_out,
            stride_h,stride_w,paramters):
    n_in=input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(name=name+"w",shape=[ksize_h,ksize_w,n_in,n_out],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernel,(1,stride_h,stride_w,1),padding="SAME")
        bias_init=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init,trainable=True,name='b')
        z=tf.nn.bias_add(conv,biases)
        activation=tf.nn.relu(z,name=scope)
        paramters+=[kernel,biases]
        return activation

def fc_op(input_op,name,n_out,paramters):
    n_in=input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name="b")
        activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        paramters += [kernel, biases]
        return activation

def mpool_op(input_op,name,ksize_h,ksize_w, stride_h,stride_w):
    return tf.nn.max_pool(input_op,ksize=[1,ksize_h,ksize_w,1],
                          strides=[1,stride_h,stride_w,1],padding="SAME",name=name)

def network(input_op,keep_prob):
    p=[]

    conv1_1=conv_op(input_op,name="conv1",ksize_h=3,ksize_w=3,n_out=64,
            stride_h=1,stride_w=1,paramters=p)
    conv1_2 = conv_op(conv1_1, name="conv2", ksize_h=3, ksize_w=3, n_out=64,
                      stride_h=1, stride_w=1, paramters=p)
    pool_1=mpool_op(conv1_2,name="pool1",ksize_h=2,ksize_w=2, stride_h=2,stride_w=2)

    conv2_1 = conv_op(pool_1, name="conv3", ksize_h=3, ksize_w=3, n_out=128,
                      stride_h=1, stride_w=1, paramters=p)
    conv2_2 = conv_op(conv2_1, name="conv4", ksize_h=3, ksize_w=3, n_out=128,
                      stride_h=1, stride_w=1, paramters=p)
    pool_2 = mpool_op(conv2_2, name="pool2", ksize_h=2, ksize_w=2,stride_h=2, stride_w=2)

    conv3_1 = conv_op(pool_2, name="conv5", ksize_h=3, ksize_w=3, n_out=256,
                      stride_h=1, stride_w=1, paramters=p)
    conv3_2 = conv_op(conv3_1, name="conv6", ksize_h=3, ksize_w=3, n_out=256,
                      stride_h=1, stride_w=1, paramters=p)
    conv3_3 = conv_op(conv3_2, name="conv7", ksize_h=3, ksize_w=3, n_out=256,
                      stride_h=1, stride_w=1, paramters=p)
    pool_3 = mpool_op(conv3_3, name="pool3", ksize_h=2, ksize_w=2, stride_h=2, stride_w=2)

    conv4_1 = conv_op(pool_3, name="conv8", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    conv4_2 = conv_op(conv4_1, name="conv9", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    conv4_3 = conv_op(conv4_2, name="conv10", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    pool_4 = mpool_op(conv4_3, name="pool3", ksize_h=2, ksize_w=2, stride_h=2, stride_w=2)

    conv5_1 = conv_op(pool_4, name="conv11", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    conv5_2 = conv_op(conv5_1, name="conv12", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    conv5_3 = conv_op(conv5_2, name="conv13", ksize_h=3, ksize_w=3, n_out=512,
                      stride_h=1, stride_w=1, paramters=p)
    pool_5 = mpool_op(conv5_3, name="pool5", ksize_h=2, ksize_w=2, stride_h=2, stride_w=2)

    shp=pool_5.get_shape()
    flat_shape=shp[1].value*shp[2].value*shp[3].value
    resh1=tf.reshape(pool_5,[-1,flat_shape],name="resh1")

    fc6=fc_op(resh1,name="fc6",n_out=4096,paramters=p)
    fc6_prob=tf.nn.dropout(fc6,keep_prob,name="fc6_drop")

    fc7 = fc_op(fc6_prob, name="fc7", n_out=4096, paramters=p)
    fc7_prob = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_prob, name="fc8", n_out=1000, paramters=p)

    softmax=tf.nn.softmax(fc8)
    prediction=tf.argmax(softmax,1)
    return prediction,softmax,fc8,p

def time_run(session,target,feed,info_string):

    total_time=0
    total_time_squared=0.0

    for i in range(num_batches + 10):
        start_time=time.time()
        _=session.run(target,feed_dict=feed)
        durtion=time.time()-start_time
        if i>=10:
            if not i%10==0:
                print ('%s:step %d,time=%.3f'%(datetime.now(),i-10,durtion))
            total_time+=durtion
            total_time_squared+=durtion*durtion
    mn=total_time/num_batches
    vr=total_time_squared/num_batches-mn*mn
    sd=math.sqrt(vr)
    print('%s:%s across %d steps,%.3f +/- %.3f sec /batch'%(datetime.now(),
                                                            info_string,num_batches,mn,sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                            dtype=tf.float32,stddev=0.1))
        keep_prob=tf.placeholder(tf.float32)
        prediction,softmax,fc8,p=network(images,keep_prob)

        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        time_run(sess,prediction,{keep_prob:1.0},"forward")

run_benchmark()