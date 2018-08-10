#-*- coding:utf-8 -*-
import json
import sys
import os
import mxnet as mx
import sys
'''
sys.path.append('/Users/hujiaxin/gluon-tutorials/')
'''

#路径仍然需检查问题
for i in ['', 'C:\\ProgramData\\Miniconda3\\python36.zip', 'C:\\ProgramData\\Miniconda3\\DLLs', 'C:\\ProgramData\\Miniconda3\\lib', 'C:\\ProgramData\\Miniconda3', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\win32', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\win32\\lib', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\Pythonwin', '.', 'C:\\Program Files\\Python36\\python36.zip', 'C:\\Program Files\\Python36\\DLLs', 'C:\\Program Files\\Python36\\lib', 'C:\\Program Files\\Python36', 'C:\\Program Files\\Python36\\lib\\site-packages', 'C:\\Program Files\\Python36\\lib\\site-packages\\IPython\\extensions', '', 'C:\\ProgramData\\Miniconda3\\python36.zip', 'C:\\ProgramData\\Miniconda3\\DLLs', 'C:\\ProgramData\\Miniconda3\\lib', 'C:\\ProgramData\\Miniconda3', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\win32', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\win32\\lib', 'C:\\ProgramData\\Miniconda3\\lib\\site-packages\\Pythonwin']:
    sys.path.append(i)

import utils
import re
import jieba
#import jieba.analyse as jieba_tag
import jieba.analyse as jieba_tag



with open('C:/Users/QTC I7-1060/Desktop/ml_hu/data/result_clean.txt','r',encoding='utf-8') as f:
    wenda = f.readlines()
    f.close()

with open('C:/Users/QTC I7-1060/Desktop/ml_hu/data/lchen_result_clean.txt','r',encoding='utf-8') as f:
    wenda.extend(f.readlines())
    f.close()

print(len(wenda))


def que_clean(que_set):
    q_fine = []
    for q in que_set:
        q_delta = q.strip('好大夫在线网上咨询').strip('_').strip('\n')
        if not '...' in q_delta:
            q_fine.append(q_delta)
    return q_fine



ans_fined = que_clean(wenda)

fined_ques = [a for a in ans_fined if len(a)>15]

answer=[i.replace(' ','').replace('-','').replace('？','').replace('?','').replace('\n','').replace('_',',').strip('.').strip(',') for i in fined_ques]

import mxnet 
from mxnet import gluon


#initialize the paramate

import mxnet as mx

#print(answer)
print('we have',len(answer),'questions!')

#we want to filter the doc of useless
def sen_filter(doc_set,tag_num = 1,ask_term = True):
    keep_set = []
    for sen in doc_set:
        sen_seg = jieba_tag.extract_tags(sen,topK=10000, \
withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        if len(sen_seg) > tag_num:
            keep_set.append(sen)
        for i in sen:
            if i in ['办','呢','吗','。','？','样','!','怎','么','如','何','应','该']:
                keep_set.append(sen)
            break
    return list(set(keep_set))



corpus_fil = sen_filter(answer,tag_num=3) #number = 9657 
print(len(corpus_fil))


corpus_chars = ' '.join(corpus_fil)
#corpus_chars = corpus_chars[0:20000]


#char to idx
idx_to_char = list(set(corpus_chars))

char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)

#print(idx_to_char[:10])
#print(char_to_idx)
#print('vocab size:', vocab_size)


corpus_indices = [char_to_idx[char] for char in corpus_chars]

sample = corpus_indices[:40]


import random
from mxnet import nd

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为label的索引是相应data的索引加一
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个数据
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


my_seq = list(range(30))
#random sampling
import random
from mxnet import nd

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为label的索引是相应data的索引加一
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个数据
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label

#seq sampling 
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    # 减一是因为label的索引是相应data的索引加一
    epoch_size = (batch_len - 1) // num_steps
    
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label

my_seq = list(range(30))

for data, label in data_iter_consecutive(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')

#one-hot vectorization similar to dummy
nd.one_hot(nd.array([0, 2]), vocab_size)

def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]

inputs = get_inputs(data)

print('input length: ', len(inputs))
print('input[0] shape: ', inputs[0].shape)



ctx = utils.try_gpu()
#ctx = mx.cpu()
print('Will use', ctx)
    

input_dim = vocab_size

# 隐含状态长度
#hidden_dim = 256      #该参数是否设置正确？
#try to tunnig the hidd_dim parameter!
hidden_dim=500

output_dim = vocab_size
std = .01

def get_params():
    # 隐含层
    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params

#define the circulation neuronetwork!
def rnn(inputs, state, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
    # H: 尺寸为 batch_size * hidden_dim 矩阵。
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)

#测试
state = nd.zeros(shape=(data.shape[0], hidden_dim), ctx=ctx)

params = get_params()
outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state, *params)

print('output length: ',len(outputs))
print('output[0] shape: ', outputs[0].shape)
print('state shape: ', state_new.shape)


def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):
    # 预测以 prefix 开始的接下来的 num_chars 个字符。
    #prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        # 当RNN使用LSTM时才会用到，这里可以忽略。
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        # 在序列中循环迭代隐含变量。
        if is_lstm:
            # 当RNN使用LSTM时才会用到，这里可以忽略。
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix)-1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


#梯度裁剪
def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm
        else:
            for p in params:
                p.grad[:] *= 1


from mxnet import autograd
from mxnet import gluon
from math import exp


#res=[]

#ans_collect=[]  
 
def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim, 
                          learning_rate, clipping_theta, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):    
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    ans_collect=[]

    for e in range(1, epochs + 1):
        # 如使用相邻批量采样，在同一个epoch中，隐含变量只需要在该epoch开始的时候初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                # 当RNN使用LSTM时才会用到，这里可以忽略。
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps, 
                                     ctx):
            # 如使用随机批量采样，处理每个随机小批量前都需要初始化隐含变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs 尺寸：(batch_size, vocab_size)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                    state_c, *params) 
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # 设t_ib_j为i时间批量中的j元素
                # label 尺寸：（batch_size * num_steps）
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                label = label.T.reshape((-1,))
                # 拼接outputs，尺寸：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                # 经上述操作，outputs和label已对齐。
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()
            grad_clipping(params, clipping_theta, ctx)
            utils.SGD(params, learning_rate)
            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            try:
                ans = exp(train_loss/num_examples)
                print("Epoch %d. Perplexity %f" % (e, 
     ans))
            except OverflowError:
                print("overflow!", train_loss,';',num_examples)
                print('still we are on our way---batch')
                ans = float('inf')
            
            
            #for seq in seqs:
            for i in range(len(seqs)):
                seq=seqs[i]
                try:
                    ansres=predict_rnn(rnn, seq, pred_len, params,
                      hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs, is_lstm)
                    print('-',seq,ansres)
                    #ans_collect.append(ansres.split(' ')[0])
                    ans_collect.append(re.split(r'[;,\s]\s*',ansres)[0])
                    with open('C:/Users/QTC I7-1060/Desktop/ml_hu/data/que_generate.txt','w') as f:
                        f.write(ansres,'\n')
                        f.close()
                except:
                    print('theme--',seq,'unmatched!')
               #ans_collect.setdefault(seq,ansres)
            print()
    return ans_collect

   


#set the parameter to run the model
epochs = 200
num_steps = 35
learning_rate = 0.1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#def loadFont(file,selected_value):
f = open('C:/Users/QTC I7-1060/Desktop/ml_hu/data/tags_conditions_lung.json', encoding='utf-8')  #//设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
setting = json.load(f)
#family = setting['tags'] 
theme_collect=[]
for d in setting:
	if len(d['tags']):
		theme_collect.extend(d['tags'].split(','))





#seqs = [seq1]*100+[seq2]*100+[seq3]*100+[seq4]*100+[seq5]*100
#seqs=[theme for theme in theme_collect if theme in char_to_idx.keys()][:99]
#print(seqs)
for theme in theme_collect:
    for i in theme:
        if not i in char_to_idx.keys():
            theme_collect.remove(theme)
            break

random.shuffle(theme_collect)
seqs=theme_collect[:99]


#随机采样
'''

train_and_predict_rnn(rnn=rnn, is_random_iter=True, epochs=200, num_steps=35,
                      hidden_dim=hidden_dim, learning_rate=0.2,
                      clipping_theta=5, batch_size=32, pred_period=20,
                      pred_len=100, seqs=seqs, get_params=get_params,
                      get_inputs=get_inputs, ctx=ctx,
                      corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                      char_to_idx=char_to_idx)

'''

#相邻采样
res=train_and_predict_rnn(rnn=rnn, is_random_iter=False, epochs=200, num_steps=10,
                      hidden_dim=hidden_dim, learning_rate=0.2,
                      clipping_theta=0.01, batch_size=32, pred_period=20,
                      pred_len=20, seqs=seqs, get_params=get_params,
                      get_inputs=get_inputs, ctx=ctx,
                      corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                      char_to_idx=char_to_idx)
print(res)

#with open('/Users/hujiaxin/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/81a308ec0769a1ff3cb7565d648feceb/Message/MessageTemp/ab86407be8e4e5104936dd02c6cb255b/File/历史语料.txt') as f:
#    ans_rev = f.read().replace('?','').split('\n')



def getavage_recall(target):
	count=0
	for i in range(len(target)):
		count+= target[i] in answer
	print(count/len(target))

print('we try to rate the unique questions:','\n',getavage_recall(res))


with open('C:/Users/QTC I7-1060/Desktop/ml_hu/data/que_generate.txt','w') as f:
    for q in res:
        f.write(q,'\n')
    f.close()
