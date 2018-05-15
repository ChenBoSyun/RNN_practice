
# coding: utf-8

# In[8]:


from keras.models import load_model
import pickle
import char_table
import numpy as np


# In[10]:


def predict(rowx):
    q = ctable.decode(rowx[0])
    preds = model.predict_classes(rowx, verbose=0)
    guess = ctable.decode(preds[0], calc_argmax=False)
    return guess


# In[11]:


if __name__ == "__main__":
    chars = '0123456789+- '
    ctable = char_table.characterTable(chars)
    model = load_model('RNN_model.h5')
    
    with open('corpus.pickle', 'rb') as handle:
        corpus = pickle.load(handle)
        
    x_test = corpus['test_x']
    y_test = corpus['test_y']    
        
    num_of_truth = 0
    for i in range(len(x_test)):
        rowx, rowy = x_test[np.array([i])], y_test[np.array([i])]
        guess = predict(rowx)
        correct = ctable.decode(rowy[0])
        if guess == correct:
            num_of_truth = num_of_truth + 1
    print("accuracy:  %3f"%(num_of_truth/len(x_test)))

