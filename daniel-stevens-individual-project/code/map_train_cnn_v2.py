# %%
# Import modules
from glob import glob
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from sklearn import preprocessing
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Conv2D, Flatten, Reshape, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import cv2
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import pickle

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# Sets random seeds for reproducibility.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# Define hyper parameters and add to stat dictionary
LR = 1e-4
N_NEURONS = (1200, 900, 900)
N_EPOCHS = 20
BATCH_SIZE = 200
DROPOUT = 0.2
SAMPLE_COUNT = 7929
WIDTH = 100
HEIGHT = 100
COLOR = True
TEST_SIZE = .15
ACT_FUNC_1 = 'relu'
ACT_FUNC_2 = 'relu'
LOSS_FUNC = 'mean_squared_error'
runstat = {}
runstat['DTG_START'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
runstat['LR'] = LR
runstat['N_NEURONS'] = str(N_NEURONS)
runstat['N_EPOCHS'] = N_EPOCHS
runstat['BATCH_SIZE'] = BATCH_SIZE
runstat['DROPOUT'] = DROPOUT
runstat['SAMPLE_COUNT'] = SAMPLE_COUNT
runstat['HEIGHT'] = HEIGHT
runstat['WIDTH'] = WIDTH
runstat['COLOR'] = COLOR
runstat['TEST_SIZE'] = TEST_SIZE
runstat['ACT_FUNC_1'] = ACT_FUNC_1
runstat['ACT_FUNC_2'] = ACT_FUNC_2
runstat['LOSS_FUNC'] = LOSS_FUNC

# %% ----------------------------------- Data Preparation --------------------------------------------------------------
# Find and load images
print(os.getcwd())
image_files = glob('FP/train/250_*.png')
print(str(len(image_files)) + ' training images found..')
images = []
labels = []
i = 0
for image in image_files[:SAMPLE_COUNT]:
    i += 1
    label1 = float(image.split('_')[1])
    # Load and resize image
    im1 = cv2.imread(image)
    ri0 = cv2.resize(im1, dsize=(WIDTH, WIDTH), interpolation=cv2.INTER_CUBIC)
    # Rotate and flip images
    ri90 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 90, 1), (WIDTH, HEIGHT))
    ri180 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 180, 1), (WIDTH, HEIGHT))
    ri270 = cv2.warpAffine(ri0, cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 270, 1), (WIDTH, HEIGHT))
    hi0 = cv2.flip(ri0, 0)
    vi0 = cv2.flip(ri0, 1)
    hvi0 = cv2.flip(ri0, -1)
    images += [ri0, ri90, ri180, ri270, hi0, vi0, hvi0]
    labels += [label1] * 7

# %%
# Train test split
SAMPLE_COUNT = len(images)
imgarray= np.array(images)
print(imgarray.shape)
runstat['AUG_COUNT'] = SAMPLE_COUNT
print(str(SAMPLE_COUNT) + ' total images')
labels_array = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(imgarray, labels_array, test_size=TEST_SIZE, shuffle=False)
print(x_train.shape)
print(x_test.shape)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(lr=LR), loss=LOSS_FUNC, metrics=[metrics.mae, metrics.mse])


# %% -------------------------------------- Training Loop ----------------------------------------------------------
# Trains the model, while printing validation metrics at each epoch
model.fit(x_train, y_train,batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test), shuffle=True,
          callbacks=[ModelCheckpoint("cnn2_map.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
# Load best version of model and calculate final scores
model = load_model('cnn2_map.hdf5')
runstat['DTG_STOP'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
runstat['Loss'] = model.evaluate(x_test, y_test)[0]
print("Final loss on validations set:", runstat['Loss'])

# %%
# Create/update table of run results.
try:
    model_stat_dicts = pickle.load(open("model_stat_dicts.p", "rb"))
except:
    model_stat_dicts = []
model_stat_dicts += [runstat]
pickle.dump(model_stat_dicts, open("model_stat_dicts.p", "wb"))
df = pd.DataFrame(model_stat_dicts)
df.to_csv('run_stats.csv', index=False)
