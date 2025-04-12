import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dropout, Activation

ruta = '/Users/adharacavazos/codes/pythonProyects/LSTM_Production/zeros/La Joya.csv'
#ruta = 'zeros/CIASA.csv'
df = pd.read_csv(ruta, index_col='Name-Year-Month')

df = df.reset_index()
df = df.drop(columns=['Name-Year-Month'])


def train_val_test_split(dataframe, trainSize=0.7, valueSize=0.15):
    N_years = int(dataframe.shape[0]/12)

    train = int(trainSize * N_years) * 12
    val = int(round(valueSize * N_years))*12
    test = dataframe.shape[0]-train-val-12

    return train, val, test

tr, vl, ts = train_val_test_split(df)
print(f'Tamaño set de entrenamiento: {tr}')
print(f'Tamaño set de validación: {vl}')
print(f'Tamaño set de prueba: {ts}')

def crear_datasets_supervisado(array, types, offset, input_length, output_length):
    X, Y = [], []
    for i in range(offset, types + offset, 12):
        X.append(array[i:i + input_length, :-2])

    for j in range(offset + 12, types + offset + 12, 12):
        temp = [np.sum(array[j:j + output_length, -2]), np.sum(array[j:j + output_length, -1])]
        Y.append(np.array(temp).reshape(-1, 1))

    X = np.array(X)
    Y = np.array(Y)
    return X, Y



INPUT_LENGTH = 12
OUTPUT_LENGTH = 7

x_tr, y_tr = crear_datasets_supervisado(df.values, tr,0,INPUT_LENGTH, OUTPUT_LENGTH)
x_vl, y_vl = crear_datasets_supervisado(df.values, vl,tr,INPUT_LENGTH, OUTPUT_LENGTH)
x_ts, y_ts = crear_datasets_supervisado(df.values, ts,tr+vl,INPUT_LENGTH, OUTPUT_LENGTH)

print('Tamaños entrada (BATCHES x INPUT_LENGTH x FEATURES) y de salida (BATCHES x OUTPUT_LENGTH x FEATURES)')
print(f'Set de entrenamiento - x_tr: {x_tr.shape}, y_tr: {y_tr.shape}')
print(f'Set de validación - x_vl: {x_vl.shape}, y_vl: {y_vl.shape}')
print(f'Set de prueba - x_ts: {x_ts.shape}, y_ts: {y_ts.shape}')

from sklearn.preprocessing import MinMaxScaler
def escalar_dataset(data_input, col_ref):
    col_ref = df.columns.get_loc(col_ref)

    NFEATS = df.shape[1]

    scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(NFEATS)]

    x_tr_s = np.zeros(data_input['x_tr'].shape)
    x_vl_s = np.zeros(data_input['x_vl'].shape)
    x_ts_s = np.zeros(data_input['x_ts'].shape)

    y_tr_s = np.zeros(data_input['y_tr'].shape)
    y_vl_s = np.zeros(data_input['y_vl'].shape)
    y_ts_s = np.zeros(data_input['y_ts'].shape)

    col_Scale_Temp = list(range(NFEATS))
    col_Scale_Temp.remove(col_ref)
    col_Scale_Temp.pop(-1)
    for i in col_Scale_Temp:
        x_tr_s[:, :, i] = scalers[i].fit_transform(x_tr[:, :, i])
        x_vl_s[:, :, i] = scalers[i].transform(x_vl[:, :, i])
        x_ts_s[:, :, i] = scalers[i].transform(x_ts[:, :, i])

    y_tr_s[:, :, 0] = scalers[col_ref].fit_transform(y_tr[:, :, 0])
    y_vl_s[:, :, 0] = scalers[col_ref].transform(y_vl[:, :, 0])
    y_ts_s[:, :, 0] = scalers[col_ref].transform(y_ts[:, :, 0])

    data_scaled = {
        'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
        'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
        'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,
    }

    return data_scaled, scalers[col_ref]


data_in = {
    'x_tr': x_tr, 'y_tr': y_tr,
    'x_vl': x_vl, 'y_vl': y_vl,
    'x_ts': x_ts, 'y_ts': y_ts,
}


data_s, scaler = escalar_dataset(data_in, col_ref='Total Cosechado')

x_tr_s, y_tr_s = data_s['x_tr_s'], data_s['y_tr_s']
x_vl_s, y_vl_s = data_s['x_vl_s'], data_s['y_vl_s']
x_ts_s, y_ts_s = data_s['x_ts_s'], data_s['y_ts_s']

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Input,Dense
from tensorflow.keras.optimizers.legacy import Adam
from keras.regularizers import l2

N_UNITS = 60    #130#55#50#60 95
INPUT_SHAPE = (x_tr_s.shape[1], x_tr_s.shape[2])
valor_de_regularizacion = 0.001  # 0.001 #0.1 0.0001 0.0003

modelo = Sequential()

modelo.add(Input(shape=INPUT_SHAPE))
modelo.add(Flatten())
modelo.add(Dense(N_UNITS, activation='relu', kernel_regularizer=l2(valor_de_regularizacion))) #sigmoid relu
modelo.add(Dropout(0.1))#.3 #.1 .5
modelo.add(Dense(N_UNITS, activation='relu', kernel_regularizer=l2(valor_de_regularizacion))) #relu softmax
modelo.add(Dropout(0.1))#.3 .4 .1
modelo.add(Dense(N_UNITS, activation='relu', kernel_regularizer=l2(valor_de_regularizacion))) # relu
modelo.add(Dropout(0.1)) #.3 .1
modelo.add(Dense(N_UNITS, activation="relu")) #relu, softmax
modelo.add(Dropout(0.1)) #.3 #.2 .1


modelo.add(Dense(1))
modelo.add(Activation('linear'))
def root_mean_squared_error(y_true, y_pred):
    rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))
    return rmse


optimizador = Adam(learning_rate=0.0001) #0.01 0.0001 0.001 0.00025
modelo.compile(
    optimizer=optimizador,
    loss=root_mean_squared_error,
)

EPOCHS = 200 #50 55
BATCH_SIZE = 75#60#55#50 75
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=(EPOCHS*0.15))
historia = modelo.fit(
    x=x_tr_s,
    y=y_tr_s,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_vl_s, y_vl_s),
    verbose=0,
    callbacks=[es]
)

plt.plot(historia.history['loss'], label='RMSE train')
plt.plot(historia.history['val_loss'], label='RMSE val')
plt.xlabel('Iteración')
plt.ylabel('RMSE')
plt.legend()
#plt.show()

rmse_tr = modelo.evaluate(x=x_tr_s, y=y_tr_s, verbose=0)
rmse_vl = modelo.evaluate(x=x_vl_s, y=y_vl_s, verbose=0)
rmse_ts = modelo.evaluate(x=x_ts_s, y=y_ts_s, verbose=0)

print('Comparativo desempeños:')
print(f'  RMSE train:\t {rmse_tr:.3f}')
print(f'  RMSE val:\t {rmse_vl:.3f}')
print(f'  RMSE test:\t {rmse_ts:.3f}')
print()

from sklearn.metrics import r2_score, mean_absolute_percentage_error
def predecir(x, model, scaler):
    y_pred_s = model.predict(x, verbose=0)
    y_pred_s = y_pred_s.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_s.reshape(1, -1))
    return y_pred

y_ts_pred = predecir(x_ts_s, modelo, scaler)
y_real = y_ts[0].flatten()
desescalar = [0,0]


mape = mean_absolute_percentage_error(y_real, y_ts_pred.flatten())
r2 = r2_score(y_real, y_ts_pred.flatten())

print(f'MAPE: {mape:.3f}')
print(f'R^2: {r2:.3f}')

print('Y real: ', y_real)
print('Y predicted: ', y_ts_pred)

for i in range(len(y_real)):
    desescalar[i] = y_real[i] - y_ts_pred[0][i]

print(desescalar)