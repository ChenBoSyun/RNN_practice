
# coding: utf-8

# In[3]:


###執行one-got encoding的class
###將 0-9 + - ' ' 建成size為13的index

import numpy as np
class characterTable():
    def __init__(self,chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
    #encode將string轉換成one-hot vector
    #example: 4 -> [0,0,0,0,0,0,0,1,0,0,0,0,0]
    def encode(self , C , num_rows):
        x = np.zeros((num_rows , len(self.chars)))
        for i , c in enumerate(C):
            x[i , self.char_indices[c]] = 1
        return x
    
    #encode將one-hot vector轉換成string
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

