import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Dropout
from keras.utils.np_utils import to_categorical
from sklearn import *
import random
import math

def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    #plt.show()
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False, pair=False):
    figsize = 1.0    
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
     
    if (pair):
        A = generate_a_drawing(figsize, U, V, noise)
        B = generate_a_drawing(figsize, U, V, 0)
        return(A,B)
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_disk(noise=0.0, free_location=False, pair=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    
    if(pair):
        A = generate_a_drawing(figsize, U, V, noise)
        B = generate_a_drawing(figsize, U, V, 0)
        return(A,B)
    
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False, pair=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    
    if (pair):
        A = [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]
        imdata2 = generate_a_drawing(figsize, U, V, 0)
        B = [imdata2, [U[0], V[0], U[1], V[1], U[2], V[2]]]
        return(A,B)
    
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]


im = generate_a_rectangle(10, True)
plt.imshow(im.reshape(100,100), cmap='gray')
plt.show()

im = generate_a_disk(10)
plt.imshow(im.reshape(100,100), cmap='gray')
plt.show()

[im, v] = generate_a_triangle(20, False)
plt.imshow(im.reshape(100,100), cmap='gray')
plt.show()


def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1: 
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_dataset_denoising(nb_samples, noise=10.0, size_reduction=1, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    im_edge = int(math.sqrt(im_size))
    print(im_edge)
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples,im_size])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        category = np.random.randint(3)
        if category == 0:
            X[i], Y[i] = generate_a_rectangle(noise, free_location, True)
        elif category == 1: 
            X[i], Y[i] = generate_a_disk(noise, free_location, True)
        else:
            [X[i], V], [Y[i], V] = generate_a_triangle(noise, free_location, True)
    X = (X + noise) / (255 + 2 * noise)
    Y = Y / 255
    
    #reshape
    X = X.reshape((nb_samples, im_edge, im_edge))
    X = X[:, 0:im_edge:size_reduction, 0:im_edge:size_reduction]
    X = X.reshape((nb_samples,-1))
    Y = Y.reshape((nb_samples, im_edge, im_edge))
    Y = Y[:, 0:im_edge:size_reduction, 0:im_edge:size_reduction]
    Y = Y.reshape((nb_samples,-1))
    return [X, Y]


def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(300, 20, True)
    Y_test = to_categorical(Y_test, 3) 
    return [X_test, Y_test]

def generate_dataset_regression(nb_samples, noise=0.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples, 6])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        [X[i], Y[i]] = generate_a_triangle(noise, free_location=True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]



def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((100,100))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()

def generate_test_set_regression():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(300, 20)
    return [X_test, Y_test]

#Simple classification
[X_train, Y_train] = generate_dataset_classification(500, 20)
# Build model
model = Sequential()
model.add(Dense(3, input_dim=10000, activation="softmax"))

#opti = 'sgd' #Stochastic gradient descent optimizer.
#loss='mean_squared_error'

opti = 'adam' #adam optimizer
loss='categorical_crossentropy'

model.compile(loss=loss, optimizer=opti, metrics=['accuracy', 'categorical_accuracy'])

# Train
model.fit(X_train, to_categorical(Y_train, 3), batch_size=32, nb_epoch=100)
model.summary()

#Test
X_test = generate_a_rectangle(noise=20)
X_test = X_test.reshape(1, X_test.shape[0])
Y_test = model.predict(X_test)
print("Result rectangle",np.argmax(Y_test))
X_test = generate_a_disk()
X_test = X_test.reshape(1, X_test.shape[0])
Y_test = model.predict(X_test)
print("Result disk",np.argmax(Y_test))
[X_test, v] = generate_a_triangle(20, False)
X_test = X_test.reshape(1, X_test.shape[0])
Y_test = model.predict(X_test)
print("Result triangle", np.argmax(Y_test))

#Visualization
plt.imshow((model.get_weights()[0][:,0]).reshape(100,100))
plt.show()
plt.imshow((model.get_weights()[0][:,1]).reshape(100,100))
plt.show()
plt.imshow((model.get_weights()[0][:,2]).reshape(100,100))
plt.show()

#A more difficult classification
n_samples = 300
[X_train, Y_train] = generate_dataset_classification(n_samples, 20, True)
model.fit(X_train, to_categorical(Y_train, 3), batch_size=32, nb_epoch=100)
[X_test, Y_test] = generate_test_set_classification()
print("model evaluating")
model.evaluate(X_test, Y_test)

#covolutional model
model2 = Sequential()
model2.add(Conv1D(16, 5, input_shape=(10000, 1)))
model2.add(Conv1D(16, 5))
model2.add(MaxPooling1D())
model2.add(Flatten())
model2.add(Dense(3, activation="softmax"))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])

model2.summary()

# Train
print("model2 fitting")
model2.fit(X_train.reshape((n_samples, -1, 1)), to_categorical(Y_train, 3), batch_size=32, nb_epoch=100)
#Test
print("model2 evaluate")
model2.evaluate(X_test.reshape((X_test.shape[0], -1, 1)), Y_test)

#Regression problem
noise = 20
np.random.seed(42)
[X_train, Y_train] = generate_dataset_regression(3000, noise)
[X_test, Y_test] = generate_test_set_regression()

#Normalisation
Y_train_norm = np.zeros((Y_train.shape[0],Y_train.shape[1]))
Y_test_norm = np.zeros((Y_test.shape[0],Y_test.shape[1]))
for n in range(Y_train.shape[0]) :
    u = min(Y_train[n,0],Y_train[n,2], Y_train[n,4])
    v = min(Y_train[n,1],Y_train[n,3], Y_train[n,5])
    Y_train_norm[n,0] = Y_train[n,0] - u
    Y_train_norm[n,1] = Y_train[n,1] - v
    Y_train_norm[n,2] = Y_train[n,2] - u
    Y_train_norm[n,3] = Y_train[n,3] - v
    Y_train_norm[n,4] = Y_train[n,4] - u
    Y_train_norm[n,5] = Y_train[n,5] - v
for n in range(Y_test.shape[0]) :
    u = min(Y_test[n,0],Y_test[n,2], Y_test[n,4])
    v = min(Y_test[n,1],Y_test[n,3], Y_test[n,5])
    Y_test_norm[n,0] = Y_test[n,0] - u
    Y_test_norm[n,1] = Y_test[n,1] - v
    Y_test_norm[n,2] = Y_test[n,2] - u
    Y_test_norm[n,3] = Y_test[n,3] - v
    Y_test_norm[n,4] = Y_test[n,4] - u
    Y_test_norm[n,5] = Y_test[n,5] - v

#Model definition
model3 = Sequential()
model3.add(Conv1D(16, 5, input_shape=(10000,1), activation='relu'))
model3.add(Dropout(0.1))
model3.add(Conv1D(16, 5, activation='relu'))
model3.add(MaxPooling1D())
model3.add(Dropout(0.1))
model3.add(Conv1D(16, 5, activation='relu'))
model3.add(MaxPooling1D())
model3.add(Flatten())
model3.add(Dense(16, activation="softmax"))
model3.add(Dense(6, activation="softmax"))
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model3.summary()

model3.fit(X_train.reshape((X_train.shape[0], -1, 1)), Y_train_norm, batch_size=64, nb_epoch=100)

#Prediction and evaluation
Y = model3.predict(X_test.reshape((X_test.shape[0], -1 ,1)))
model3.evaluate(X_test.reshape((X_test.shape[0], -1 ,1)), Y_test_norm)

#Visualisation
for i in range(5) :
    sample = random.randint(0,X_test.shape[0]-1)
    
    print("Visualization of X_test[", sample,"], Y[",sample,"]")
    visualize_prediction(X_test[sample], Y[sample])
    
    print("Visualization of X_test[", sample,"], Y_test[",sample,"]")
    visualize_prediction(X_test[sample], Y_test_norm[sample])
    
#    u = min(Y_test[sample,0],Y_test[sample,2], Y_test[sample,4])
#    v = min(Y_test[sample,1],Y_test[sample,3], Y_test[sample,5])
#    Y[sample,0] = Y[sample,0] + u
#    Y[sample,1] = Y[sample,1] + v
#    Y[sample,2] = Y[sample,2] + u
#    Y[sample,3] = Y[sample,3] + v
#    Y[sample,4] = Y[sample,4] + u
#    Y[sample,5] = Y[sample,5] + v    
#    print("Visualization of X_test[", sample,"], Y[",sample,"]")
#    visualize_prediction(X_test[sample], Y[sample])

#Try to learn the translation
Y_train_norm = np.zeros((Y_train.shape[0],Y_train.shape[1]+2))
Y_test_norm = np.zeros((Y_test.shape[0],Y_test.shape[1]+2))
for n in range(Y_train.shape[0]) :
    u = min(Y_train[n,0],Y_train[n,2], Y_train[n,4])
    v = min(Y_train[n,1],Y_train[n,3], Y_train[n,5])
    Y_train_norm[n,0] = Y_train[n,0] - u
    Y_train_norm[n,1] = Y_train[n,1] - v
    Y_train_norm[n,2] = Y_train[n,2] - u
    Y_train_norm[n,3] = Y_train[n,3] - v
    Y_train_norm[n,4] = Y_train[n,4] - u
    Y_train_norm[n,5] = Y_train[n,5] - v
    Y_train_norm[n,6] = u
    Y_train_norm[n,7] = v  
for n in range(Y_test.shape[0]) :
    u = min(Y_test[n,0],Y_test[n,2], Y_test[n,4])
    v = min(Y_test[n,1],Y_test[n,3], Y_test[n,5])
    Y_test_norm[n,0] = Y_test[n,0] - u
    Y_test_norm[n,1] = Y_test[n,1] - v
    Y_test_norm[n,2] = Y_test[n,2] - u
    Y_test_norm[n,3] = Y_test[n,3] - v
    Y_test_norm[n,4] = Y_test[n,4] - u
    Y_test_norm[n,5] = Y_test[n,5] - v
    Y_test_norm[n,6] = u
    Y_test_norm[n,7] = v

model3 = Sequential()
model3.add(Conv1D(16, 5, input_shape=(10000,1), activation='relu'))
model3.add(Dropout(0.1))
model3.add(Conv1D(16, 5, activation='relu'))
model3.add(MaxPooling1D())
model3.add(Dropout(0.1))
model3.add(Conv1D(16, 5, activation='relu'))
model3.add(MaxPooling1D())
model3.add(Flatten())
model3.add(Dense(16, activation="softmax"))
model3.add(Dense(8, activation="softmax"))
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model3.summary()

model3.fit(X_train.reshape((X_train.shape[0], -1, 1)), Y_train_norm, batch_size=64, nb_epoch=100)

#Prediction and evaluation
Y = model3.predict(X_test.reshape((X_test.shape[0], -1 ,1)))
model3.evaluate(X_test.reshape((X_test.shape[0], -1 ,1)), Y_test_norm)

#Visualisation
for i in range(5) :
    sample = random.randint(0,X_test.shape[0]-1)
    
    prediction = np.zeros((1,6))
    prediction[0, 0] = Y[sample,0] + Y[sample,6]
    prediction[0, 1] = Y[sample,1] + Y[sample,7]
    prediction[0, 2] = Y[sample,2] + Y[sample,6]
    prediction[0, 3] = Y[sample,3] + Y[sample,7]
    prediction[0, 4] = Y[sample,4] + Y[sample,6]
    prediction[0, 5] = Y[sample,5] + Y[sample,7]
    
    print("Visualization of X_test[", sample,"], Y[",sample,"]")
    visualize_prediction(X_test[sample], Y[sample, 0:6])
    
    print("Visualization of X_test[", sample,"], Y[",sample,"] with translation")
    visualize_prediction(X_test[sample], prediction)
    
    print("Visualization of X_test[", sample,"], Y_test[",sample,"]")
    visualize_prediction(X_test[sample], Y_test[sample])

    
#hourglass network for denoising
reduce_factor = 4

X_train, Y_train = generate_dataset_denoising(nb_samples=3000, noise=30.0, size_reduction=reduce_factor)
X_train = X_train.reshape((3000, -1, 1))

X_test, Y_test = generate_dataset_denoising(nb_samples=100, noise=30.0, size_reduction=reduce_factor)
X_test = X_test.reshape((100, -1, 1))

model4 = Sequential()
model4.add(Conv1D(16, 5, input_shape=(int(10000/pow(reduce_factor,2)),1), activation="relu"))
model4.add(Conv1D(16, 5, activation="relu"))
model4.add(MaxPooling1D())
model4.add(Conv1D(2, 5, activation="relu"))
model4.add(Conv1D(2, 5, activation="relu"))
model4.add(MaxPooling1D())
model4.add(UpSampling1D())
model4.add(Conv1D(2, 5, activation="relu"))
model4.add(Conv1D(2, 5, activation="relu"))
model4.add(UpSampling1D())
model4.add(Conv1D(16, 5, activation="relu"))
model4.add(Conv1D(16, 5, activation="relu"))
model4.add(Flatten())
model4.add(Dense(int(10000/pow(reduce_factor,2)), activation="softmax"))
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model4.summary()

model4.fit(X_train, Y_train, batch_size=25, nb_epoch=10)

Y = model4.predict(X_test)
model4.evaluate(X_test, Y_test)

im_edge = int(100/reduce_factor)
for i in range(5):
    sample = random.randint(0,X_test.shape[0]-1)
    print("X_test[",sample,"]")
    plt.imshow(X_test[sample].reshape((im_edge, im_edge)), cmap='gray')
    plt.show()
    print("Y[",sample,"]")
    plt.imshow(Y[sample].reshape((im_edge, im_edge)), cmap='gray')
    plt.show()
    print("Y_test[",sample,"]")
    plt.imshow(Y_test[sample].reshape((im_edge, im_edge)), cmap='gray')
    plt.show()