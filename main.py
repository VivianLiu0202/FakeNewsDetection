import numpy as np
import pandas as pd
import os
import jieba
import csv
import jieba.analyse
from keras.utils import pad_sequences
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout1D
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,precision_recall_curve,average_precision_score
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import Sequential
from keras import losses
from keras.layers import (GRU,LSTM,
                          Embedding,
                          Dense,
                          Dropout,
                          Bidirectional)
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
from string import punctuation

train_df=pd.read_csv("train.news.csv")
print('Shape of dataset',train_df.shape)
print(train_df.columns)
print(train_df.head())
print(train_df.isnull().sum())
train_df=train_df.dropna()
print('Shape of dataset',train_df.shape)
train_df=shuffle(train_df)

#Preparing thr text data
stem=PorterStemmer()
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
def stopwordslist(filepath):
    stop_words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stop_words
stopwords=stopwordslist('chinesestopwords.txt')
def cleaning(text):
    cutwords=list(jieba.lcut_for_search(text))
    final_cutwords=''
    for word in cutwords:
        if word not in stopwords and punc:
            final_cutwords+=word+" "
    return final_cutwords

train_df["Report Content"]=train_df["Report Content"].apply(lambda x: x.split("##"))
t=pd.DataFrame(train_df.astype(str))
train_df["data"]=t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
t = pd.DataFrame(train_df.astype(str))
train_df["data"]=t["data"].apply(cleaning)
train_data=train_df["data"]
print(train_data)
texts=train_data
targets=np.asarray(train_df["label"])

MAX_NB_WORDS = 28455
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index))
MAX_SEQUENCE_LENGTH = 300
text_data = pad_sequences(sequences,
                          maxlen = MAX_SEQUENCE_LENGTH,
                          padding = 'post',
                          truncating = 'post')
print(text_data)
EMBEDDING_DIM = 128
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM))
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 1)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
cp=ModelCheckpoint('model_Rnn.hdf5',
                   monitor='val_accuracy',
                   verbose=1,
                   save_best_only=True,
                   mode='auto'
                   )
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
print(model.summary())
#
# model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])

#训练的总轮数为5轮
# print(text_data.shape)
# print(targets.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(text_data,
#                                                     targets,
#                                                     test_size=0.2,
#                                                     train_size=0.8,
#                                                     #0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练
#                                                     #并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
#                                                     random_state=18,#random_state结果复现。如果想要完全随机可以去掉这个参数
#                                                     stratify=None,
#                                                     shuffle=True)


VALIDATION_SPLIT = 0.1
EPOCHS = 5
history = model.fit(text_data,
                    targets,
                    batch_size = 32,
                    validation_split = VALIDATION_SPLIT,
                    epochs = EPOCHS,
                    shuffle=True,
                    callbacks=[cp],
                    validation_freq=1,
                    )

#绘图
plt.title('accuracy')
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.legend()
plt.show()

plt.title('loss')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()

# plot train and validation loss
# plt.figure(figsize=(8,8),dpi=200)
# plt.plot(history.history['loss'][500:])
# plt.plot(history.history['val_loss'][500:])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','validation'], loc='upper right')
# plt.show()


# loss, accuracy = model.evaluate(text_data, targets, verbose=True)
# print("Training Accuracy: {:.4f}".format(accuracy))
# print("Training Loss: {:.4f}".format(loss))

# loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
# print("Testing Accuracy: {:.4f}".format(accuracy))
# print("Testing Loss: {:.4f}".format(loss))
#
#
# y_pred=model.predict(X_test, batch_size=200, verbose=1)
# report = classification_report(y_test, y_pred.round())
# print(report)


#TEST

test_df=pd.read_csv("test.news.csv",encoding='utf-8')
test_df=shuffle(test_df)
print('Shape of dataset ',test_df.shape)
print(test_df.columns)
print('missing data are:')
print(test_df.isnull().sum())

test_df["Report Content"]=test_df["Report Content"].apply(lambda x: x.split("##"))
t=pd.DataFrame(test_df.astype(str))
test_df["data"]=t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
test_label=np.asarray(test_df["label"])
test_df.fillna(method='bfill', inplace=True)
test_df["data"] = test_df["data"].apply(cleaning)
text_test=test_df["data"]
print(text_test)

test_sequences = tokenizer.texts_to_sequences(text_test)
test_data = pad_sequences(test_sequences,
                          maxlen = MAX_SEQUENCE_LENGTH,
                          padding = 'post',
                          truncating = 'post')

test_data_y=model.predict(test_data)

preds = test_data_y
for i in range(len(preds)):
    if preds[i] >= 0.5:
        preds[i] = 1
    else:
        preds[i] = 0

print(preds)
print('分类报告： \n',classification_report(test_label,preds))
print('AUC: ',roc_auc_score(test_label,preds))

precision,recall,thresholds =precision_recall_curve(test_label,model.predict(test_data))
PRC=average_precision_score(test_label,model.predict(test_data))
print('PRC: ',PRC)

plt.figure(figsize=(6,6))
plt.title('PR curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.step(recall, precision, color='b', label=' (PRC={:.4f})'.format(PRC))
plt.plot([0, 1], [1, 0], color='m', linestyle='--')
plt.legend(loc='lower right')
plt.show()


predictions =[]
for i in preds:
    predictions.append(i)
print(len(predictions))

submission = pd.DataFrame({'label':predictions})
submission.to_csv('submit.csv',index=False)

