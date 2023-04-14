
from PIL import Image as img, ImageFont as font, ImageDraw as draw
ttf = font.truetype
import numpy as np


input_size = 28

a1 = ttf("Fonts/LEMONMILK-Regular.otf",input_size);
a2 = ttf("Fonts/Roboto-Regular.ttf",input_size);
a3 = ttf("Fonts/OpenSans-Regular.ttf",input_size);
a4 = ttf("Fonts/SchibstedGrotesk-Regular.ttf",input_size);
a5 = ttf("Fonts/Anuphan-Regular.ttf",input_size)
class char:
    def __init__(self, font, letter, hot_label):
        self.img = glyph(font, letter)
        self.hot_label = hot_label
        self.letter = letter
        self.font = font
        
def glyph(font, letter):
  i = img.new("L", (input_size, input_size))
  d = draw.Draw(i)
  d.text((4,-4),letter, font=font, fill='white')
  return i
  
letters = list("GASN")
train = []
test = []
num_classes = len(letters)

for b,i in enumerate(letters):
  hot_label = np.zeros(num_classes)
  hot_label[b] = 1
  for u in [a1, a2, a3, a4]:
    train.append(char(u, i, hot_label))
  test.append(char(a5, i, hot_label))


import os
for i in train:
    J =('data/train/'+i.font.getname()[0]+'/'+i.letter+'.png')
    try:
        os.makedirs(os.path.dirname(J))
    except FileExistsError:
        pass
    i.img.save(J)

for i in test:
    J =('data/test/'+i.font.getname()[0]+'/'+i.letter+'.png')
    try:
        os.makedirs(os.path.dirname(J))
    except FileExistsError:
        pass
    i.img.save(J)
    

from random import random
from math import exp

def sigmoid(net):
    return 1.0 / (1.0 + exp(-net))

def step(value):
    return 1.0 if value >= 0 else 0.0

class Neuron:
    def __init__(self, input_count):
        weights = self.__weights = []
        self.__y = 0.0
        min_range = self.__rangeMin = - 0.0003
        max_range = self.__rangeMax = 0.0003
        diff_range = max_range - min_range        
        for i in range(input_count + 1):
            weights.append(min_range + (diff_range * random()))

    def derivative(self):
        y = self.__y
        return y * (1.0 - y)

    def calc_y(self, x):
        weights = self.__weights
        net = weights[0]
        for i in range(len(x)):
            net += x[i] * weights[i + 1]
        self.__y = sigmoid(net)

    def get_y(self):
        return self.__y

    def get_weights(self):
        return self.__weights[1:]

    def get_bias(self):
        return self.__weights[0]

    def correct_weights(self, weights_deltas):
        weights = self.__weights
        for i in range(len(weights)):
            weights[i] += weights_deltas[i]

def Vector(x, d):
    x = list(np.array(x).reshape(-1)) 
    d = list(np.array(d).reshape(-1))
    return (x, d)

from random import shuffle
shuffle(train)

from math import pow

class OneLayerPerseptron:

    def __init__(self, input_count, output_count):
        self.__input_count = input_count
        neurons = self.__neurons = []
        for j in range(output_count):
            neurons.append(Neuron(input_count))

    def train(self, vector, learning_rate):
        neurons = self.__neurons
        neurons_len = len(neurons)
        X = vector[0]
        D = vector[1]
        X_len = len(X)
        D_len = len(D)
        
        weights_deltas = [[0] * (X_len + 1)] * neurons_len
        
        for neuron in neurons:
            neuron.calc_y(X)
            
        for j, neuron in enumerate(neurons):
            weights_delta = weights_deltas[j]
            
            sigma = (D[j] - neuron.get_y()) * neuron.derivative()
            
            for i in range(len(neuron.get_weights())):
                weights_delta[i] = learning_rate * sigma * X[i]
            
            neuron.correct_weights(weights_delta)
            
        loss = 0
        
        for j, neuron in enumerate(neurons):
            loss += pow(D[j] - neuron.get_y(), 2)
        return 0.5 * loss

    def test(self, vector):
        X = vector[0]
        y = []
        for neuron in self.__neurons:
            neuron.calc_y(X)
            y.append(neuron.get_y())
        return y



from datetime import datetime
def get_max_neuron_idx(a):
    return max(range(len(a)), key=lambda i: a[i])

learning_rate = 1e-8
num_epochs = 150
shuffle(train)
one_layer_net = OneLayerPerseptron(input_size * input_size, num_classes)
train_len = len(train)
test_len = len(test)
print('Size of training set:', train_len)
print('Size of testing set:', test_len)
print('Count of epochs:', num_epochs)
d=datetime.now()
print("Begin training", d)
for epoch in range(1,num_epochs+1):
    loss = 0
    for data in train:
        loss += one_layer_net.train(Vector(data.img, data.hot_label), learning_rate)
    shuffle(train)
t=datetime.now()
print("End training", t)
t -= d
print("Training time", t)
print("Iteration time", t / num_epochs)
passed = 0
for i in test:
    x = i.img
    d = i.hot_label
    y = one_layer_net.test(Vector(x, d))
    recognized_letter = letters[get_max_neuron_idx(y)]
    original_letter = i.letter
    if recognized_letter == original_letter:
        passed += 1
    print(original_letter,"recognized as",recognized_letter)

accuracy = passed / test_len * 100.0
print("Accuracy: {:.0f}%".format(accuracy))
