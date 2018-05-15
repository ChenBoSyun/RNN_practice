
# coding: utf-8

# In[4]:


from keras.models import Sequential
from keras import layers
from keras.callbacks import History
import numpy as np
import pickle
import char_table

history = History()

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

TRAINING_SIZE = 160000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+- '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

def RNN_model():
    print('Build model...')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(layers.RepeatVector(DIGITS + 1))
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def train(model , x_train , y_train , x_val, y_val):
    ctable = char_table.characterTable(chars)
    
    val_loss = []
    val_acc = []
    for iteration in range(150):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        history = model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
        val_loss.append(history.history['val_loss'][0])
        val_acc.append(history.history['val_acc'][0])
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if REVERSE else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)
    return model , val_loss , val_acc

if __name__ == "__main__":
    
    #use save corpus
    with open('corpus.pickle', 'rb') as handle:
        corpus = pickle.load(handle)
    
    x_train = corpus['x_train']
    y_train = corpus['y_train']
    x_val = corpus['x_val']
    y_val = corpus['y_val']
    
    model = RNN_model()
    model , val_loss , val_acc = train(model , x_train , y_train , x_val , y_val)
    #save trained model
    model.save('RNN_model.h5')

