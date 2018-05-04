from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from configobj import ConfigObj
from data import load_data
from main import load_model
from keras.models import model_from_json
import adv
from adv import adv

torconf = "tor.conf"
config = ConfigObj(torconf)
dnn = config['dnn']
openw = config.as_bool('openw')
test_data = config['test_data']
#get data
datapath = config['datapath']
minlen = config.as_int('minlen')
maxlen = config[dnn].as_int('maxlen')
traces = config.as_int('traces')
dnn = config['dnn']
x_test, y_test = load_data(test_data,dnn_type=dnn,minlen=1,maxlen=maxlen,openw=openw)
x_train, y_train = load_data(datapath,minlen=minlen,maxlen=maxlen,traces=traces,dnn_type=dnn)
train = np.append(x_train.reshape((7475,3000)), y_train[:,1].reshape((7475,1)), axis=1)
test = np.append(x_test.reshape((6975, 3000)), y_test[:,1].reshape((6975,1)), axis=1)
batch_size = config[dnn].as_int('batch_size')
num_classes=2
epochs = 1

model_path = config['model_path']
model = load_model(model_path)

adv(model,train,test)
