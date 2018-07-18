#-*- coding = utf-8
import tensorflow as tf
import numpy as np

#Get TestSet
w_real = np.float32(np.random.randint(5,size=(5,1)))
print (w_real)

x_data=np.float32(np.random.rand(100,5))
print(x_data)

y_data=np.dot(x_data,w_real)

mean_val = np.mean(y_data)
print(mean_val)

y_real = np.where(y_data > mean_val, 1, -1)
print(y_real)

#Model parameters
W= tf.Variable(np.random.randint(5,size=(5,1)), dtype = tf.float32)
b= tf.Variable(np.random.random_sample(), dtype = tf.float32)

#Moudel input and output
x = tf.placeholder(tf.float32)
perceptron_model = tf.matmul(x,W) + b
y = tf.placeholder(tf.float32)


#loss
loss = tf.reduce_sum(tf.abs(perceptron_model * tf.matrix_transpose(y)))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_data, y: y_real})
  print(sess.run(W))

# evaluate training accuracy
#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_data, y: y_real})
