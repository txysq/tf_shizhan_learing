from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
sess=tf.InteractiveSession()

def weight_init(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_init(shape):
    initial=tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder(tf.float32,[None,784])
real_y=tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

w_conv1=weight_init([5,5,1,32])
b_conv1=bias_init([32])
f_covn1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=pool_2(f_covn1)

w_conv2=weight_init([5,5,32,64])
b_conv2=bias_init([64])
f_covn2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=pool_2(f_covn2)

w_fc1=weight_init([7*7*64,1024])
b_fc1=bias_init([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_init([1024,10])
b_fc2=bias_init([10])
h_fc2=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(real_y*tf.log(h_fc2),reduction_indices=[1]))
train_optimizer=tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

correct_predition=tf.equal(tf.argmax(h_fc2,1),tf.argmax(real_y,1))
accuracy=tf.reduce_mean(tf.cast(correct_predition,tf.float32))

tf.global_variables_initializer().run()

for i in range(3000):
    batch_x,batch_y=mnist.train.next_batch(50)
    train_optimizer.run(feed_dict={x:batch_x,real_y:batch_y,keep_prob:1.0})
    if i%50==0:
        train_accuracy=accuracy.eval({x:batch_x, real_y: batch_y, keep_prob: 1.0})
        print("step:%d,training accuracy%g" % (i,train_accuracy))

print("test accuracy %g"% accuracy.eval(feed_dict={
    x:mnist.test.images,real_y:mnist.test.labels,keep_prob:1.0}))
