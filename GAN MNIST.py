
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:


def discriminator(x_image, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    #first conv and pool layers
    #search for 32 different 5x5 pixel features
    #create weight and bias variables w/ tf.get_variable
    #first weight matrix will be randomly initialized and will be 5x5
    d_w1 = tf.get_variable('d_w1', [5,5,1,32], initializer =tf.truncated_normal_initializer(stdev=0.02))
    
    #tf.constant_init generates tensors with constant values
    d_b1 = tf.get_variable('d_b1', [32], initializer= tf.constant_initializer(0))
    #tf.nn.conv2d() is tf's function for a common convolution
    #it takes 4 arguments: input volume(28x28x1), weight matrix, stride, and padding
    #strides = [batch,height,width,channels]
    
    d1= tf.nn.conv2d(input=x_image, filter =d_w1, strides = [1,1,1,1], padding = 'SAME')
    d1 = d1 + d_b1
    #squash with relu
    d1= tf.nn.relu(d1)
    #a pooling layer performs down-sampling by dividing the input into
    #into rectangular pooling regions and computing the average of each region
    d1=tf.nn.avg_pool(d1, ksize=[1,2,2,1], strides =[1,2,2,1], padding = 'SAME')
    
    #Second convolutional and pool layers
    
    #these search for 64 5 x5 features
    
    d_w2 = tf.get_variable('d_w2', [5,5,32,64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2= tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input = d1, filter=d_w2, strides =[1,1,1,1], padding ='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize =[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    #now fully connected layer one
    
    d_w3 = tf.get_variable('d_w3', [7*7*64,1024], initializer=tf.truncated_normal_initializer(stddev(0.02)))
    d_b3= tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1,7*7*64])
    d3 = tf.matmul(d3, d_w3)
    d3= d3 + d_b3
    d3 = tf.nn.relu(d3)
    
    #Last fully connected layer and has output
    
    d_w4 = tf.get_variable('d_w4', [1024,1],initializer=tf.truncated_normal_initializer(stddev(0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer = tf.constant_initializer(0))
    
    #do a final matrix  multiplication and return the activation value
    
                           
    d4 = tf.matmul(d3, d_w4) + d_b4
    
    return d4

    


# In[ ]:


#The generator is like a reverse CNN, with CNNs the goal is to
#transform a 2 or 3 dimensional matrix of pixel values into a single probability
#a generator takes an n-dimensional noise vector and upsample it
#to become(in this case) a 28 x 28 image
#Relus are used to stabilize the outputs of each layer

def generator(batch_size, z_dim):
    z=tf.truncated_normal([batch_size, z_dim], mean = 0, stddev=1, name='z')
    #first deconv block
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1',[3136], initializer = tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.reshape(g1, [-1,56,56,1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)
    
    #generate 50 features
    
    g_w2 = tf.get_variable('g_w2', [3,3,1,z_dim/2], dtype=tf.float32, initiializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2',[z_dim/2], initializer =tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1,g_w2, strides=[1,2,2,1], padding = 'SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope ='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_image(g2,[56,56])
    
    #generate 25 features
    
    g_w3 = tf.get_variable('g_w3', [3,3,z_dim/2,z_dim/4], dtype=tf.float32, initializer = tf.truncated_normal_initializer(stddev(0.02)))
    g_b3 = tf.get_variable('g_b4', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1,2,2,1], padding = 'SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope = 'bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56,56])
    
    #final leyer with output
    
    g_w4 = tf.get_variable('g_w4', [1,1,z_dim/4,1] dtype =tf.float32, initializer= tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1,2,2,1], padding = 'SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)
    
    #no batch normalization and add a sigmoid function to make generated images crisper
    #dimensions of g4 = batch_size x 28 x 28 x 1


# In[ ]:


sess = tf.Session()

batch_size = 50
z_dimensions = 100

x_placeholder = tf.placeholder("float", shape = [None, 28, 28, 1], name='x_placeholder')
#the placeholder is for feeding images into the discriminator
#Loss functions for GANs are more complex than the traditional CNN(where MSE or Hinge would do the trick)
#think of a GAN as a zero sum minimax game
#the generator is seeking to generate better and more convincing images
#the discriminator seeks to become better at distinguishing real and false images

Gz = generator(batch_size, z_dimensions)
#Gz holds the generated images
#g(z)
Dx = discriminator(x_placeholder)
# Dx holds the discriminators prediction probabilities
#D(x)
Dg = discriminator(Gz, reuse=True)
#Dg holds the discriminator prediction probabilities for generated images
#D(g(z))

#The generator wants the discriminator to output a 1(positive...or fooled)
#therefore we want to compute the loss between the Dg and label of 1
#this can be done with the tf.nn.sigmoid_cross_entropy_with_logits function
#This means the cross entropy loss will be taken between two arguments. The with_logits part
#means that the function will operate on unscaled values....which means
#instead of using a softmax function to squash the output activations to probability values
#from 0 to 1 we just return the unscaled value of the matrix multiplication
#the redduce mean function takes the mean of all the components in the matrix returned
#by the cross entropy function. This is a way of reducing the loss
#to a single scalar value, instead of a vector or matrix

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

#The goal of the discriminator is to get the correct labels
#output 1 for real image and 0 for generated images. WE want to compute the loss between
#Dx and the correct label of 1 and the loss between Dg and the correct label of 0

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size,1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse =False) as scope:
    #Now we make the optimizers Adam uses learning rates and momentum
    #we call adam's minimize function and specify the variables we want it to update
    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)
    
    #Train the generator
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)


# In[ ]:


#Outputs a Summary protocol buffer containing a single scalar value.
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count', d_real_count_ph)
tf.summary.scalar('d_fake_count', d_fake_count_ph)
tf.summary.scalar('g_count', g_count_ph)

# Sanity check to see how the discriminator evaluates
# generated and real MNIST images
d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

tf.summary.scalar('d_on_generated_eval', d_on_generated)
tf.summary.scalar('d_on_real_eval', d_on_real)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 10)
merged = tf.summary.merge_all()
logdir = "tensorboard/gan/"
writer = tf.summary.FileWriter(logdir, sess.graph)
print(logdir)


# In[ ]:


saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

#During every iteration there will be two updates: one to the discriminator and one to the generator
#for the generator update we will feed in a random z vector to the generator and pass that out put to the discriminator
#to obtain a probability score(the Dg variable we specified earlier)
#from our loss function the cross entropy loss gets minimized and only the generator weights and biases get updated
#Rinse and Repeat for the discriminator.
#we will take a batch of images from the variable we created at the beginning of the program
#these will be + examples and the images in the previous section will be the negatives

gLoss = 0
dLossFake, dLossReal = 1, 1
d_real_count, d_fake_count, g_count = 0, 0, 0
for i in range(50000):
    real_image_batch = mnist.train.next_batch(batch_size)

