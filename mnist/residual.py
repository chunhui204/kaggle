
# coding: utf-8

# In[2]:


import  tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D://mnist', one_hot=True)


# In[4]:


print(mnist.train.images.shape)


# In[5]:


np.random.shuffle(mnist.train.images)
train_images = mnist.train.images[:50000,:]
train_labels = mnist.train.labels[:50000,:]
validation_images = mnist.train.images[50000:,:]
validation_labels = mnist.train.labels[50000:,:]


# In[6]:


train_images = train_images.reshape(-1,28,28,1)
train_labels = train_labels.reshape(-1,10)
validation_images  =validation_images.reshape(-1,28,28,1)
validation_labels= validation_labels.reshape(-1,10)


# In[7]:


start = 0 
def _next_batch(batch_size):
    global start
    global train_images
    global train_labels
    
    start += batch_size
    if start >train_images.shape[0]:
        start = 0
        permutation = np.random.permutation(train_images.shape[0])
        train_images =train_images[permutation,:,:,:]
        train_labels = train_labels[permutation,:]
    
    return (train_images[start:start+batch_size, :], train_labels[start:start+batch_size, :])


# In[8]:


BN_DECAY = 0.9997
LOSS_COLLECTION = "loss_collection"
_BATCH_NORM_DECAY = 0.997
BN_EPSILON = 1e-5
CONV_WEIGHT_DECAY = 0.0003
FC_WEIGHT_DECAY = 0.0003
FC_DROPOUT_PROB = 0.5
#training
_LEARNING_RATE_DECAY=0.99
_LEARNING_RATE_BASE = 0.003
EPOCHES = 30000
BATCH_SIZE= 128


# In[9]:


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  trainable=True):
    v = tf.get_variable(name,
                       shape,
                       initializer=initializer,
                       trainable=trainable)
    
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)(v)
        tf.add_to_collection(LOSS_COLLECTION, regularizer)
        
    return v


# In[10]:


def _batch_norm(x, istraining):
    #定义beta, gamma，是通过训练更新的参数，trainable=True
    gamma = _get_variable('gamma', 
                          shape=(x.shape[-1]), 
                          initializer=tf.constant_initializer(1.0),
                          trainable = True)
    beta = _get_variable("beta",
                         shape=(x.shape[-1]),
                         initializer=tf.constant_initializer(0.0),
                         trainable = True)
    #定义moving_mean, moving_variance,是testing中使用的mean, var，是在训练中通过moving average更新的，note:不参与训练
    moving_mean = _get_variable('moving_mean',
                                shape=(x.shape[-1]),
                                initializer=tf.constant_initializer(0.0),
                                trainable = False)
    moving_variance = _get_variable('moving_variance',
                                shape=(x.shape[-1]),
                                initializer=tf.constant_initializer(1.0),
                                trainable = False)
    def bn_tarining():
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        #moving average具体操作，定义依赖关系，先更新moving变量再求这层的bn
        update_moving_mean = tf.assign(moving_mean, BN_DECAY*moving_mean + (1-BN_DECAY)*mean)
        update_moving_variance = tf.assign(moving_variance, BN_DECAY*moving_variance + (1-BN_DECAY)*variance)
        #training中使用的是该层实际的mean，var
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.nn.batch_normalization(x, beta, gamma, mean, variance, BN_EPSILON)
    def bn_testing():
        #testing，直接使用training阶段通过moving average更新的mean，var
        return tf.nn.batch_normalization(x, beta, gamma, moving_mean, moving_variance, BN_EPSILON)
    
    out = tf.cond(istraining, bn_tarining, bn_testing)
    
    return out


# In[11]:


def conv_relu(x, 
         filter_num, 
         filter_shape, 
         strides,
         padding='VALID'):
    
    shape=[filter_shape, filter_shape, x.get_shape()[-1], filter_num]
    w = _get_variable("weight", 
                      shape, 
                      initializer=tf.truncated_normal_initializer(stddev=1.0), 
                      weight_decay=CONV_WEIGHT_DECAY,
                      trainable=True)
    b = _get_variable("bias", 
                      shape=(filter_num), 
                      initializer=tf.constant_initializer(0.), 
                      weight_decay=0., 
                      trainable=True)
    x = tf.nn.conv2d(x, 
                    w, 
                    strides=(1,strides,strides,1), 
                    padding=padding)
    return tf.nn.relu(x + b)
    


# In[12]:


def fc(x, output_dim):
    shape = x.get_shape()
    w = _get_variable("weight", 
                      shape=(shape[-1],output_dim), 
                      initializer=tf.truncated_normal_initializer(stddev=1.0), 
                      weight_decay=FC_WEIGHT_DECAY, 
                      trainable=True)
    b = _get_variable("bias", 
                      shape=(output_dim), 
                      initializer=tf.constant_initializer(0.), 
                      weight_decay=0., 
                      trainable=True)
    return tf.matmul(x, w) + b

def fc_relu(x, output_dim, keep_prob):
    x = fc(x, output_dim)
    a = tf.nn.relu(tf.matmul(x, w) + b)
    if keep_prob < 1.:
        a = tf.nn.dropout(a, keep_prob)
    return a


# In[13]:


activation = tf.nn.relu
def flatten_pool(x):
    return tf.contrib.layers.flatten(x)

def _max_pool(x, ksize, strides, padding='SAME'):
    return tf.nn.max_pool(x, 
                          ksize=(1,ksize,ksize,1), 
                          strides=(1,strides,strides,1), 
                          padding=padding)
def _average_pool(x, ksize, strides, padding='SAME'):
    return tf.nn.avg_pool(x, 
                          ksize=(1,ksize,ksize,1),
                          strides=(1,strides,strides,1),
                          padding=padding)


# In[14]:


def residual_block(x, 
                   filters_num, 
                   filter_shape, 
                   strides,
                   is_training,
                   residual_format='identify'):
    f1,f2,f3 = filters_num
    x_origin = x
    assert residual_format=='identify' or residual_format=='projection'
    
    with tf.variable_scope('a'):
        #first layer ,bottleneck
        if residual_format == 'identify':
            x = conv_relu(x, 
                          filter_num= f1,
                          filter_shape= 1, 
                          strides= 1)
        elif residual_format == 'projection':
            x = conv_relu(x, 
                          filter_num= f1,
                          filter_shape= 1, 
                          strides= strides)
        x = _batch_norm(x, is_training)
        x = activation(x)
    
    with tf.variable_scope('b'):
        #sencond layer, main conv
        x = conv_relu(x, 
                      filter_num= f2,
                      filter_shape= filter_shape, 
                      strides= 1,
                      padding='SAME')
        x = _batch_norm(x, is_training)
        x = activation(x)
        
    with tf.variable_scope('c'):
        #third layer, bottleneck
        x = conv_relu(x, 
                      filter_num= f3,
                      filter_shape= 1, 
                      strides= 1)
        x = _batch_norm(x, is_training)
    
    with tf.variable_scope('shortcut'):
        #shortcut,
        #if use projection format, should tune x_origin dimmention to match x after convolution
        if residual_format == 'projection':
            x_origin = conv_relu(x_origin, 
                                 filter_num=f3,
                                 filter_shape=1, 
                                 strides=strides)
            x_origin = _batch_norm(x_origin, is_training)
        
    return activation(x + x_origin)


# In[29]:


def model():
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None,28,28,1), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")
    is_bn_training = tf.placeholder(tf.bool)
    
    data = train_images
    labels = train_labels
    ######################################################################################
    with tf.variable_scope('layer1'):
        x = conv_relu(X, filter_shape=5, strides=1, padding="SAME", filter_num=32)
        x = _max_pool(x,ksize=2, strides=2, padding="SAME")
    with tf.variable_scope('layer2'):
        x = conv_relu(x, filter_shape=5, strides=1, padding="SAME", filter_num=64)
        x = _max_pool(x, ksize=2, strides=2, padding="SAME")
#     with tf.variable_scope("block1"):
#         x = residual_block(x, 
#                            filter_shape=3, 
#                            filters_num=[64,64,256], 
#                            strides=2, 
#                            is_training = is_bn_training,
#                            residual_format="projection")
#     with tf.variable_scope('block2'):
#         x = residual_block(x, 
#                            filter_shape=3, 
#                            filters_num=[64,64,256], 
#                            strides=2, 
#                            is_training = is_bn_training,
#                            residual_format="identify")
#     with tf.variable_scope('block3'):
#         x = residual_block(x, 
#                            filter_shape=3, 
#                            filters_num=[64,64,256], 
#                            strides=2, 
#                            is_training = is_bn_training,
#                            residual_format="identify")
#     with tf.variable_scope('block4'):
#         x = residual_block(x, 
#                            filter_shape=3, 
#                            filters_num=[64,64,256], 
#                            strides=2, 
#                            is_training = is_bn_training,
#                            residual_format="identify")

    # output layer
    x = tf.contrib.layers.flatten(x)
    Z = fc(x, 10)
    #######################################################################################################
    #compute cost
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=Z,
                                                labels=Y))
    cost += tf.add_n(
        tf.get_collection(LOSS_COLLECTION))
    
    #trainging
    dropout_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(_LEARNING_RATE_BASE, 
                                               global_step, 
                                               int(data.shape[0]/BATCH_SIZE),
                                               _LEARNING_RATE_DECAY)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, 
                                                                            global_step=global_step)
    #prediction
    prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(x, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    
    train_cost=[]
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        for epoch in range(EPOCHES):
            mini_batch_X, mini_batch_Y = _next_batch(BATCH_SIZE)
            c,_ = sess.run(
                [cost, optimizer], feed_dict={X:mini_batch_X, Y:mini_batch_Y, dropout_prob:FC_DROPOUT_PROB, is_bn_training:True})
            
            if epoch % 10==0:
                train_cost.append(c/BATCH_SIZE)
            if epoch %1000 ==0:
                print("%d epoch, training cost is %g" %(epoch, c/BATCH_SIZE))
                print('%d epoch, validdation accuracy is %g' %(epoch, accuracy.eval(
                    {X:validation_images, Y:validation_labels, dropout_prob:1.0, is_bn_training:False})))
        plt.plot(np.squeeze(train_cost))
        plt.show()


# In[ ]:


model()

