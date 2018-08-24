#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec 
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import jieba,gensim
import os
import tensorflow as tf

cat_to_id = {'投诉（含抱怨）':0, '办理':1, '咨询（含查询）':2, '其他':3,'非来电':3,'表扬及建议':3}
id_to_cat = {0:'投诉（含抱怨）', 1:'办理', 2:'咨询（含查询）', 3:'其他'}
maxlen = 1000
class_num = 4
train_path = 'data/train'
test_path = 'data/test'
train_seg_path = 'data/train_seg'
test_seg_path = 'data/test_seg'
model_path = 'model/cnn_model.h5'
w2v_path = 'data/baike.bigram-char'
result_path = 'result/test_out'
batch_size = 128
epochs = 20


#读入文件
def readfile(path):
    with open(path) as f:
        seg_list,labels = [],[]
        seg_file = open(path+'.seg','w')
        for each in f:
            try:
                label,text = each.strip().split('\t')
                label = label.strip()
                text = text.strip()
                if text:
                   seg = ' '.join(jieba.cut(text))
                   seg_list.append(seg)
                   labels.append(cat_to_id[label])
                   #保存分词后的文件
                   seg_file.write('%d\t%s\n'%(cat_to_id[label],seg))
            except:
                pass
    
    seg_file.close()
    print('read file done')
    return seg_list,labels

def loadfile(path):
    seg_list,labels = [],[]
    
    with open(path) as f:
        for each in f:
            label,text = each.strip().split('\t')
            seg_list.append(text)
            labels.append(label)
    print('load file done')
    return seg_list,labels        

#数据预处理
def preprocessing(train_texts, train_labels, test_texts, test_labels,maxlen=maxlen,num_words=20000,embed_dim=300,w2v=False,w2v_path=None):
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_texts)

    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    
    word_index = tokenizer.word_index
    
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    
    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)
    if w2v:
        #加载词向量 
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,binary=False)
        embedding_matrix = np.zeros((len(word_index)+1,embed_dim))
        for word, i in word_index.items():
            try:
                vec = model[word]
                embedding_matrix[i]=vec
            except:
                   pass
        print('data processing done')
        return x_train, y_train, x_test, y_test,embedding_matrix
        
    else:
        print('data processing done')
        return x_train, y_train, x_test, y_test


def cnn_w2v(y,embedding_matrix,maxlen=maxlen,filters=128,embed_size=300):
    news_input = Input(shape=[maxlen],name='x_seg')
    
    embedding_layer = Embedding(len(embedding_matrix),embed_size,weights=[embedding_matrix],input_length=maxlen)
    emb_news = embedding_layer(news_input)
    
    #卷积层
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(emb_news)
        l_conv = Dropout(0.5)(l_conv)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=y.shape[1], activation='softmax')(output)

    model = Model([news_input], output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    print('build model done')
    return model


#测试
def output(x_test,y_test,model_path):
    model = load_model(model_path)
    output = np.argmax(model.predict(x_test),axis=1)
    with open(result_path,'w') as f:
        for out in output:
            f.write(id_to_cat[out]+'\n')
    
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    

if __name__ == '__main__':
    train_texts, train_labels = readfile(train_file_path)
    test_texts, test_labels = readfile(test_file_path)
    
    train_texts, train_labels = loadfile(train_seg_path)
    test_texts, test_labels = loadfile(test_seg_path)
    x_train, y_train, x_test, y_test ,embedding_matrix= preprocessing(train_texts, train_labels, test_texts, test_labels,w2v=True,w2v_path=w2v_path)

    if sys.argv[1]=='train':
        model = cnn_w2v(y_train,embedding_matrix)
        batch_size = batch_size
        epochs = epochs
        print('start training model')
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
    
        filepath="model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
        model.fit(x_train, y_train,
                  validation_split=0.1,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[checkpoint])
        model.save(model_path)
        print('model saved')
    
        scores = model.evaluate(x_test, y_test)
        print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    elif sys.argv[1] == 'test':
        pre = output(x_test,y_test,model_path)



