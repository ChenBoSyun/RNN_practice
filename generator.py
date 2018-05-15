
# coding: utf-8

# In[2]:


import char_table
import pickle
import numpy as np

TRAINING_SIZE = 160000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS #預設question長度 不足的地方補空白
chars = '0123456789+- '

#隨機產生160000筆string data 包含加法和減法 example  ,  question:'140+62 ' , label:'202 '
def data_generate():
    questions = []
    answers = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int( ''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))) )
        a , b = f() , f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        q_plus = '{}+{}'.format(a,b)
        q_plus = q_plus + ' ' * (MAXLEN - len(q_plus))
        ans_plus = str(a+b)
        ans_plus = ans_plus + ' ' * (DIGITS + 1 - len(ans_plus))

        questions.append(q_plus)
        answers.append(ans_plus)

        if b > a:
            tmp = a
            a = b
            b = tmp
        q_minus = '{}-{}'.format(a,b)
        q_minus = q_minus + ' ' * (MAXLEN - len(q_minus))
        ans_minus = str(a-b)
        ans_minus = ans_minus + ' ' * (DIGITS + 1 - len(ans_minus))

        questions.append(q_minus)
        answers.append(ans_minus)

    print('Total questions:', len(questions))
    return questions,answers

#將string轉成vector，並區分成traing , validation , testing
def data_preprocess(questions,answers):
    ctable = char_table.characterTable(chars)
    
    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN, len(chars)) )
    y = np.zeros((len(answers), DIGITS + 1, len(chars)) )
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(answers):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    # train_test_split
    train_x = x[:int(TRAINING_SIZE/2)]
    train_y = y[:int(TRAINING_SIZE/2)]
    test_x = x[int(TRAINING_SIZE/2):]
    test_y = y[int(TRAINING_SIZE/2):]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)
    
    return x_train , x_val , y_train , y_val

#使用pickle 將產生的data存起來
def corpus_save(x_train , x_val , y_train , y_val):
    corpus = {}
    corpus['x_train'] = x_train
    corpus['x_val'] = x_val
    corpus['y_train'] = y_train
    corpus['y_val'] = y_val
    with open('corpus.pickle', 'wb') as handle:
        pickle.dump(corpus , handle , protocol=pickle.HIGHEST_PROTOCOL)

        
if __name__ == "__main__":
    
    questions,answers = data_generate()
    x_train , x_val , y_train , y_val = data_preprocess(questions,answers)
    corpus_save(x_train , x_val , y_train , y_val)
    print("data generation complete")

