import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2


def read_data():
    df = joblib.load('./Data/df_tfidf.joblib')
    return df


def split_data(df: pd.DataFrame):
    target = 'topic'
    y = df[target]
    x = df.drop(columns=target)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = keras.utils.to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.18, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test


def creat_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(300,),
              kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    return model


def model_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Acccuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'])
    plt.show()


def save_model(model, model_output_dir):
    model.save(model_output_dir)


def main():
    df = read_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(df)

    model = creat_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_plot(history)

    model_output_dir = '.\model_output.h5'
    save_model(model, model_output_dir=model_output_dir)


if __name__ == '__main__':
    main()
