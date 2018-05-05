import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import tensorflow as tf
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
Y = np.zeros((200,2))
Y[range(200),y]=1
print Y
print y
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
tf_x = tf.placeholder(dtype = tf.float32,shape=[None,2])
tf_y = tf.placeholder(dtype = tf.float32,shape=[None,2])
weights = {
    'h1': tf.Variable(tf.random_normal([2,3])),
    'out': tf.Variable(tf.random_normal([3,2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([3])),
    'out': tf.Variable(tf.random_normal([2]))
}
l1 = tf.tanh(tf.add(tf.matmul(tf_x,weights['h1']),biases['b1']))
out = tf.tanh(tf.add(tf.matmul(l1,weights['out']),biases['out']))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(tf_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show(2)

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    for step in range(500):
         sess.run(train_op,feed_dict={tf_x:X,tf_y:Y})
         if step%10 ==0:
             loss,acc = sess.run([loss_op,accuracy],feed_dict={tf_x:X,tf_y:Y})
             plot_decision_boundary(lambda x : np.argmax(sess.run(out,feed_dict={tf_x:x}), axis=1),X,y)
             print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
