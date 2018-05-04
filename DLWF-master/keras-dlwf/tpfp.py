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

pre_cls=model.predict_classes(x_test)
cm1 = confusion_matrix(y_test[:,1],pre_cls)
TruePositive = np.diag(cm1)

FalsePositive = []
for i in range(num_classes):
    FalsePositive.append(sum(cm1[:,i]) - cm1[i,i])
FalsePositive

FalseNegative = []
for i in range(num_classes):
    FalseNegative.append(sum(cm1[i,:]) - cm1[i,i])
FalseNegative

TrueNegative = []
for i in range(num_classes):
    temp = np.delete(cm1, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))
TrueNegative

l = len(y_test)
for i in range(num_classes):
    print(TruePositive[i] + FalsePositive[i] + FalseNegative[i] + TrueNegative[i] == l)

print(TruePositive)
print(FalsePositive)
