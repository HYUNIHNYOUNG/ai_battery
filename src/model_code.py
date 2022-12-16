import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
import RUL_models
import ABD_models
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import datetime
def folder_generator(default_folder="../save_parameter/", battery_model='GV7280', 
                    model_name='Conv1d_regression_v1', sample_length=50000,
                    task='RUL'):

    if not os.path.exists(default_folder):
        os.mkdir(default_folder)

    if not os.path.exists(default_folder + battery_model):
        os.mkdir(default_folder + battery_model)

    if not os.path.exists(default_folder + battery_model + "/" + task):
        os.mkdir(default_folder + battery_model + "/" + task)

    if not os.path.exists(default_folder + battery_model + "/" + task + "/" + model_name):
        os.mkdir(default_folder + battery_model + "/" + task + "/" + model_name)

    if not os.path.exists(default_folder + battery_model + "/" + task + "/" + model_name + "/" + str(sample_length)):
        os.mkdir(default_folder + battery_model + "/" + task + "/" + model_name + "/" + str(sample_length))

    return default_folder + battery_model + "/" + task + "/" + model_name + "/" + str(sample_length) + "/"

def model_train(train_X, train_Y,model_list,battery_model='GV7280',
                task='RUL',sample_length=5000,seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    shuffle_idx = np.arange(train_X.shape[0])
    np.random.shuffle(shuffle_idx)
    train_X=train_X[shuffle_idx]
    train_Y=train_Y[shuffle_idx]


    for model_name in list(model_list['model_name']):
        file_prefix=folder_generator(default_folder="../save_parameter/", battery_model=battery_model, 
                        model_name=model_name, sample_length=sample_length,
                        task=task)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        with tf.device('/gpu:' + str(0)):
            tf.random.set_seed(seed)
            ADAM_opt = True
            if ADAM_opt == True:
                Myadam = tfa.optimizers.extend_with_decoupled_weight_decay(
                    optimizers.Adam)
                opt = Myadam(learning_rate=0.001, weight_decay=0.0001)
            else:
                Mysgd = tfa.optimizers.extend_with_decoupled_weight_decay(
                    optimizers.SGD)
                opt = Mysgd(learning_rate=0.001,
                            weight_decay=0.0001, momentum=0.9)
            model_inputs = keras.Input(shape=(sample_length, 3))

            
            if task=='RUL':
                AI_model = RUL_models.__dict__[model_name](model_inputs)
                AI_model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])
            else:
                AI_model = ABD_models.__dict__[model_name](model_inputs)
                AI_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            hist = AI_model.fit(train_X, train_Y,batch_size=32, epochs=200)  
            AI_model.save(file_prefix + 'model.h5')
            df=pd.DataFrame(hist.history)
            df.to_csv(file_prefix + 'hist.csv')

def model_test(x_data,battery_model='GV7280',sample_length=5000,serial_num=''):
    RUL_model_list=pd.read_csv('RUL_model_list.csv')
    ABD_model_list=pd.read_csv('ABD_model_list.csv')
    rul=0
    now = datetime.datetime.now()
    pred_log=pd.read_csv('predict_log.csv')
    pred_log=pred_log.append({'name':serial_num},ignore_index=True)
    pred_log['datetime'].iloc[-1]=now
    for model_name in list(RUL_model_list['model_name']):
        RUL_model_path=folder_generator(default_folder="../save_parameter/", battery_model=battery_model, 
                            model_name=model_name, sample_length=sample_length,
                            task='RUL')
        AI_model=load_model(RUL_model_path + 'model.h5', compile=False)
        pred=AI_model.predict(x_data)
        pred_log.iloc[-1,2:26]=np.squeeze(pred)
        rul=np.min(np.squeeze(pred))
    for model_name in list(ABD_model_list['model_name']):
        ABD_model_path=folder_generator(default_folder="../save_parameter/", battery_model=battery_model, 
                            model_name=model_name, sample_length=sample_length,
                            task='ABD')
        AI_model=load_model(ABD_model_path + 'model.h5', compile=False)
        pred=AI_model.predict(x_data)
        pred_log.iloc[-1,26:-1]=np.squeeze(pred)
        abd=np.min(np.squeeze(pred))
    pred_log.to_csv('precit_log.csv',index=False)
    return rul,abd