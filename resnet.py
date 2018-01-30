# -*- coding: utf-8  -*-
import collections
import tensorflow as tf
import time
from datetime import datetime
import math

slim=tf.contrib.slim

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    """A named tuple describing a ResNet block.

      Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
          returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
          contains one (depth, depth_bottleneck, stride) tuple for each unit in the
          block to serve as argument to unit_fn.
      """

#采样
def subsample(inputs,factor,scope=None):
    if factor ==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)
    #卷积层 tf.pad填充
#输入的预处理
def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    """Strided 2-D convolution with 'SAME' padding.

        When stride > 1, then we do explicit zero-padding, followed by conv2d with
        'VALID' padding.

        Note that

           net = conv2d_same(inputs, num_outputs, 3, stride=stride)

        is equivalent to

           net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
           net = subsample(net, factor=stride)

        whereas

           net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

        is different when the input's height or width is even, which is why we add the
        current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

        Args:
          inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
          num_outputs: An integer, the number of output filters.
          kernel_size: An int with the kernel_size of the filters.
          stride: An integer, the output stride.
          rate: An integer, rate for atrous convolution.
          scope: Scope.

        Returns:
          output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.
        """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)

    #blocks函数
#把残差block和单元unit链接起来
@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):

        for block in blocks:
                #block取别名sc
            with tf.variable_scope(block.scope,'block',[net]) as sc:
                    #一个块的学习单元的拼接
                for i,unit in enumerate(block.args):
                        
                    with tf.variable_scope('unit_%d' % (i+1),values=[net]):
                        unit_depth,unit_depth_bottleneck,unit_stride=unit
                            #block.unit_fn残差单元的生成函数顺序创建连接所有残差学习单元
                        net =block.unit_fn(net,depth=unit_depth,
                            depth_bottleneck=unit_depth_bottleneck,
                            stride=unit_stride)
                #slim.utils.collect_named_outputs函数，输出net添加到outputs_collection
                net=slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
        return net
#残差网络的基本参数设置
def resnet_arg_scope(is_training=True,weight_decay=0.0001,batch_norm_decay=0.997,
    batch_norm_epsilon=1e-5,batch_norm_scale=True):
    batch_norm_params={
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'updates_collections':tf.GraphKeys.UPDATE_OPS

    }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:
                return arg_sc
    #定义了一块残差学习单元
#一个残差块　　
@slim.add_arg_scope #修饰器这样的函数可以设置scope
def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
    with tf.variable_scope(scope,'bottleneck',[inputs]) as sc:
            #获取输出通道数,最少四个
        depth_in=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
            #对输入进行正则化
        preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
            #如果输入的inputs的通道数是depth，则直接下采样
        if depth==depth_in:
            shortcut=subsample(inputs,stride,'shortcut')
            #如果不是，对输入进行正则化后进行卷积，改变输出通道数为depth
        else:
            shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,activation_fn=None,scope='shortcut')

        residual=slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope='conv1')
        residual=conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
        residual=slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn=None,activation_fn=None,scope='conv3')

        output=shortcut+residual

        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)
#主函数　include_root_block加上残差块前的卷积和池化  global_pool=True最后一层用全局池化
def resnet_v2(inputs,blocks,num_classes=None,global_pool=True,include_root_block=True,reuse=None,scope=None):
    with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
        end_points_collection=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense,],outputs_collections=end_points_collection):
            net = inputs
            #残差块前的卷积和池化
            if include_root_block:
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None,normalizer_fn=None):
                    #conv2d_same函数里有slim.conv2d函数，要对其进行参数设置
                    net=conv2d_same(net,64,7,stride=2,scope='conv1')
                net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1')
            #进入到残差块
            net=stack_blocks_dense(net,blocks)
            net=slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnet')
            if global_pool:
                net=tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
            if num_classes is not None:
                net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='logits')
            end_points=slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions']=slim.softmax(net,scope='predictions')
            return net,end_points


def time_run(sess,target,info_string):
    run_burn_in=10
    total_time=0
    total_time_squard=0
    for i in range(num_batch+run_burn_in):
        start_time = time.time()
        _=sess.run(target)
        time1=time.time()-start_time
        if i>=run_burn_in:
            if not i%10==0:
                print('%s:step %d,time1=%.3f' % (datetime.now(),i-run_burn_in,time1))
            total_time+=time1
            total_time_squard+=total_time*total_time
    mn=total_time/num_batch
    vr=total_time_squard/num_batch-mn*mn
    sd=math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batch, mn, sd))

def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_152(inputs,num_classes=1000,global_pool=True,reuse=None,scope='resnet_v2_152'):
    blocks=[
        Block(
            'block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block(
            'block2',bottleneck,[(512,128,1)]*7+[(512,128,2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22+ [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
        ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)




batch_size=32
height,width=224,224
inputs=tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net,end_points=resnet_v2_152(inputs,1000)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
num_batch=100
time_run(sess,net,'forward')