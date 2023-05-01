import tensorflow as tf
from tensorflow.python.ops import math_ops
import keras.backend as K
import numpy as np

print(f'Tensorflow version: {tf.__version__}')

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Hyperparameters
batch_size = 1
epochs = 1000
optimizer = Adam(learning_rate=0.001)
weight_init = RandomNormal()

# Build model
inputs = tf.keras.Input(shape=(1,))
x = layers.Dense(512, activation='gelu', name='H1', kernel_initializer=weight_init,\
                 kernel_regularizer=None)(inputs)
x = layers.Dense(512, activation='gelu', name='H2', kernel_initializer=weight_init,
                 kernel_regularizer=None)(x)
output = layers.Dense(1, activation='linear', name='Out', kernel_initializer=weight_init)(x)
model = tf.keras.Model(inputs, output)

# function to be integrated (normal distribution PDF)
mu = 0.0
sigma = 1.0
def f_tbi(x_coloc):
    return (1 / (sigma * np.sqrt(2 * math.pi)))\
           * np.exp(-0.5 * ((x_coloc - mu ) / sigma)**2)


# create colocation points for function to be integrated
x_coloc = np.arange(-10, 10, 0.2)  # define domain
rng = np.random.default_rng()
rng.shuffle(x_coloc)
y_coloc = f_tbi(x_coloc)

# initial condition for right tail stability
x_init = np.array([10.0])
h_init = np.array([1.0])

# integral(f(x)) initial condition
x_init2 = np.array([-10.0])
h_init2 = np.array([0.0])

# training step function for each batch
def step(x_co, y_co, x_init, h_init, x_init2, h_init2):
    x_co = tf.convert_to_tensor(x_co)
    x_co = tf.reshape(x_co, [batch_size, 1])  # required by keras input
    x_co = tf.Variable(x_co, name='x_co')
    with tf.GradientTape(persistent=True) as tape:

        # model_loss1: initial condition h_init @ x_init
        pred_init = model(x_init)
        model_loss1 = math_ops.squared_difference(pred_init, h_init)

        # model_loss3: initial condition h_init2 @ x_init2
        pred_init2 = model(x_init2)
        model_loss3 = math_ops.squared_difference(pred_init2, h_init2)

        # model_loss2: collocation points
        pred_h = model(x_co)
        dfdx = tape.gradient(pred_h, x_co)  # f(x)'
        residual = dfdx - y_co
        model_loss2 = K.mean(math_ops.square(residual), axis=-1)
        model_loss2 = tf.cast(model_loss2, tf.float32)

        #total loss
        model_loss = model_loss1 + model_loss3 + model_loss2 * 10

        trainable = model.trainable_variables
        model_gradients = tape.gradient(model_loss, trainable)

        # Update model
        optimizer.apply_gradients(zip(model_gradients, trainable))
        return np.mean(model_loss)

# Training loop
bat_per_epoch = math.floor(len(x_coloc) / batch_size)
loss = np.zeros(epochs)
for epoch in range(epochs):
    print(f'epoch: {epoch}  loss: {loss[epoch-1]}')
    for i in range(bat_per_epoch):
        n = i * batch_size
        loss[epoch] = step(x_coloc[n:n + batch_size], y_coloc[n:n + batch_size],
                           x_init, h_init, x_init2, h_init2)

# save model
dir_path = '[insert path here]'
model.save(dir_path + 'normal-cdf-model.h5')

# plot CDF results
x_test = np.zeros(200)
y_test = np.zeros(200)
y_calc_p = np.zeros(200)
y_calc_c = np.zeros(200)

for i in range(200):
    x_test[i] = (i - 100) / 10
    # y_test[i] = model.predict([x_test[i]]) - model.predict([x_test[0]])  # definite
    y_test[i] = model.predict([x_test[i]])  # definite
    y_calc_p[i] = f_tbi(x_test[i])  # PDF
    y_calc_c[i] = norm.cdf(x_test[i])

plt.plot(x_test, y_test, 'r', x_test, y_calc_p, 'b', x_test[::5], y_calc_c[::5], '.g' )
plt.ylim([-0.1, 1.1])
plt.xlim([-10, 10])
plt.title('red: CDF (neural net)     green: CDF (SciPy)     blue: PDF')
plt.xlabel('x')
plt.ylabel('CDF: probability    PDF: density')
plt.show()

# plot training loss
plt.plot(loss)
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
