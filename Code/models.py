from hypers import *

import numpy
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from tensorflow.compat.v1.keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.compat.v1.keras import backend as K

from tensorflow.compat.v1.keras.layers import Layer, InputSpec
from keras import initializers #keras2
from keras.utils.vis_utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *



# 关系编码器
def get_relation_encoder(num):
    relation_embedding = np.load("/home/mist/datasets/KGData/relation_embedding.npy")

    relation_input = Input(shape=(max_neibor_num,), dtype='int32')
    relation_embedding_layer = Embedding(1238, num, weights=[relation_embedding], trainable=True)
    relation_vecs = relation_embedding_layer(relation_input)
    relation_vecs = Dropout(0.2)(relation_vecs)
    relationEncoder = Model(relation_input, relation_vecs)

    return relationEncoder



class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        # Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        # K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        # V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        # A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        KT_seq = K.permute_dimensions(K_seq, (0, 1, 3, 2))
        A = tf.matmul(Q_seq, KT_seq) / self.size_per_head ** 0.5


        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        # O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = tf.matmul(A, V_seq)

        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        # O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = K.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)




def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)


    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model


def GraphCoAttNet15(num):
    entity_input = Input(shape=(num+max_neibor_num,100))
    entity_emb = keras.layers.Lambda(lambda x:x[:,:max_neibor_num,:])(entity_input)
    candidate_emb = keras.layers.Lambda(lambda x:x[:,max_neibor_num:,:])(entity_input)
    
    entity_vecs = Attention(5,20)([entity_emb]*3)
    
    entity_co_att = Dense(100)(entity_vecs)
    candidate_co_att = Dense(100)(candidate_emb)
    
    S = keras.layers.Dot(axes=-1)([entity_co_att,candidate_co_att])
    
    entity_self_att = Dense(100)(entity_vecs)
    
    candidate_co_att = Dense(100)(candidate_emb)
    entity_co_att = keras.layers.Dot(axes=[-1,-2])([S,candidate_emb,])
    entity_att = keras.layers.Add()([entity_self_att,entity_co_att])
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_att = keras.layers.Reshape((max_neibor_num,))(Dense(1)(entity_att))
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_vec = keras.layers.Dot(axes=[-1,-2])([entity_att,entity_vecs])
    
    model = Model(entity_input,entity_vec)
    
    return model


def GraphCoAttNet20(num):
    entity_input = Input(shape=(num * 2, 100))
    entity_emb = keras.layers.Lambda(lambda x: x[:, :num, :])(entity_input)
    candidate_emb = keras.layers.Lambda(lambda x: x[:, num:, :])(entity_input)

    entity_vecs = Attention(5, 20)([entity_emb] * 3)  # [bz,10,100]

    entity_co_att = Dense(100)(entity_vecs)  # [bz,10,100]
    candidate_co_att = Dense(100)(candidate_emb)  # [bz,10,100]

    S = keras.layers.Dot(axes=-1)([entity_co_att, candidate_co_att])  # 【bz,10,10]

    entity_self_att = Dense(100)(entity_vecs)  # [bz,10,100]

    candidate_co_att = Dense(100)(candidate_emb)  # [bz,10,100]
    entity_co_att = keras.layers.Dot(axes=[-1, -2])([S, candidate_emb, ])  # 【bz,10,100]
    entity_att = keras.layers.Add()([entity_self_att, entity_co_att])  # [BZ,10,100]
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_att = keras.layers.Reshape((num,))(Dense(1)(entity_att))  # [bz,10]
    entity_att = keras.layers.Activation('tanh')(entity_att)
    entity_vec = keras.layers.Dot(axes=[-1, -2])([entity_att, entity_vecs])  # [bz,100]

    model = Model(entity_input, entity_vec)

    return model

def get_contexts_encoder(title_word_embedding_matrix):

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], title_word_embedding_matrix.shape[1], weights=[title_word_embedding_matrix],trainable=True)
    
    word_vecs = title_word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep1 = Conv1D(400,kernel_size=3,activation='relu',padding='same')(droped_vecs)
    word_rep2 = Attention(20,20,)([droped_vecs]*3)
    word_rep2 = keras.layers.Activation('relu')(word_rep2)
    
    word_rep = keras.layers.Add()([word_rep1,word_rep2])
    
    word_rep = Dropout(0.2)(word_rep)

    
    sentEncodert = Model(sentence_input, word_rep)
    return sentEncodert

def get_context_aggergator():
    vec_input = Input(shape=(MAX_SENT_LENGTH,400+1))
    vecs = keras.layers.Lambda(lambda x:x[:,:,:400])(vec_input) #(bz,30,400)
    att = keras.layers.Lambda(lambda x:x[:,:,400])(vec_input) #(bz,30,1)
    att = keras.layers.Reshape((MAX_SENT_LENGTH,))(att)
    vec = keras.layers.Dot(axes=[-1,-2])([att,vecs])
    
    return Model(vec_input,vec)

def get_agg():
    vec_input = Input(shape=(MAX_SENT_LENGTH+1,MAX_SENT_LENGTH))
    vecs1 = keras.layers.Lambda(lambda x:x[:,:MAX_SENT_LENGTH,:])(vec_input) #(bz,30,30)
    vec2 = keras.layers.Lambda(lambda x:x[:,MAX_SENT_LENGTH,:])(vec_input) #(bz,1,30)
    vec2 = Reshape((MAX_SENT_LENGTH,))(vec2) #(bz,30)
    cross_att = Dot(axes=-1)([vecs1,vec2]) #(bz,30,1)
    cross_att = Reshape((MAX_SENT_LENGTH,))(cross_att) #(bz,30,)

    return Model(vec_input,cross_att)


def get_entity_encoder(max_neibor_num,):
    entity_input = Input(shape=(max_neibor_num,100), dtype='float32')
    entity_vecs = Attention(2,50)([entity_input,entity_input,entity_input])
    entity_vecs = Add()([entity_vecs,entity_input])
    entity_vecs = entity_input   # [bz , 5, 100]
    droped_rep = Dropout(0.2)(entity_vecs)  # [bz , 5, 100]
    entity_vec = AttentivePooling(max_neibor_num,100)(droped_rep)
    sentEncodert = Model(entity_input, entity_vec)
    return sentEncodert

def create_pair_pair(max_entity_num,):
    num = max_entity_num*(max_neibor_num+2)   #

    entity_input = Input(shape=(num+num,100))  # [bz, 140,100]
    
    user_entity_input = keras.layers.Lambda(lambda x:x[:,:num,:])(entity_input) # [bz, 70,100]
    news_entity_input = keras.layers.Lambda(lambda x:x[:,num:,:])(entity_input)#  [bz, 70,100]
    
    user_entity_zerohop = keras.layers.Lambda(lambda x: x[:, max_entity_num * max_neibor_num:60, :])(user_entity_input)  # [BZ ,10 ,100]
    user_entity_onehop = keras.layers.Lambda(lambda x:x[:,:max_entity_num*max_neibor_num,:])(user_entity_input)   # [BZ, 50,100]
    user_entity_onehop = keras.layers.Reshape((max_entity_num,max_neibor_num,100))(user_entity_onehop)  #[BZ , 10 , 5 ,100]


    user_can = keras.layers.Lambda(lambda x: x[:, 60:, :])(user_entity_input)  # [BZ ,10 ,100]

    user_can = keras.layers.Reshape((max_entity_num * 100,))(user_can)  # [BZ , 1000]
    user_can = keras.layers.RepeatVector(max_entity_num)(user_can)  # [bz , 10 ,1000]
    user_can = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(user_can)  # [bz ,10 ,10 ,100]

    news_entity_zerohop = keras.layers.Lambda(lambda x: x[:, max_entity_num * max_neibor_num:60, :])(news_entity_input)  # [BZ ,10 ,100]
    news_entity_onehop = keras.layers.Lambda(lambda x: x[:, :max_entity_num * max_neibor_num, :])(news_entity_input)  # [BZ, 50,100]
    news_entity_onehop = keras.layers.Reshape((max_entity_num, max_neibor_num, 100))(news_entity_onehop)  # [BZ , 10 , 5 ,100]

    news_can = keras.layers.Lambda(lambda x: x[:, 60:, :])(news_entity_input)  # [BZ ,10 ,100]

    news_can = keras.layers.Reshape((max_entity_num * 100,))(news_can)  # [BZ , 1000]
    news_can = keras.layers.RepeatVector(max_entity_num)(news_can)  # [bz , 10 ,1000]
    news_can = keras.layers.Reshape((max_entity_num, max_entity_num, 100))(news_can)  # [bz ,10 ,10 ,100]

    # —————————————————————————————————————————————————————————————————————————————————————————————————————


    user_entity_onehop = keras.layers.Concatenate(axis=-2)([user_entity_onehop,news_can]) #  [bz , 10 ,15, 100]
    news_entity_onehop = keras.layers.Concatenate(axis=-2)([news_entity_onehop,user_can]) #  [bz , 10 ,15, 100]
    
    gcat = GraphCoAttNet15(max_entity_num)   #[bz,15,100] - > [BZ,100]
    
    user_entity_onehop = TimeDistributed(gcat)(user_entity_onehop) # [bz , 10 ,100]
    news_entity_onehop = TimeDistributed(gcat)(news_entity_onehop) # [bz , 10 ,100]
    
    user_entity_vecs = keras.layers.Concatenate(axis=-1)([user_entity_zerohop,user_entity_onehop])# [bz,10,200]
    news_entity_vecs = keras.layers.Concatenate(axis=-1)([news_entity_zerohop,news_entity_onehop])# [bz,10,200]
    Merge = Dense(100)
    user_entity_vecs = Merge(user_entity_vecs)# [bz , 10 ,100]
    news_entity_vecs = Merge(news_entity_vecs)# [bz , 10 ,100]

    user_entity_vecs = keras.layers.Concatenate(axis=-2)([user_entity_vecs,news_entity_zerohop])
    news_entity_vecs = keras.layers.Concatenate(axis=-2)([news_entity_vecs,user_entity_zerohop])
    
    gcat0 = GraphCoAttNet20(max_entity_num) #entity co-att

    user_entity_vec = gcat0(user_entity_vecs)
    news_entity_vec = gcat0(news_entity_vecs)
    
    vec = keras.layers.Concatenate(axis=-1)([user_entity_vec,news_entity_vec])# 【bz，200】
    
    model = Model(entity_input,vec)

    
    return model

def create_pair_model(title_word_embedding_matrix): #with title
    MergeLayer = Dense(400)
    
    
    contexts_encoder = get_contexts_encoder(title_word_embedding_matrix)
    contexts_agg = get_context_aggergator()

    
    clicked_title_input = Input(shape=(MAX_SENTS,MAX_SENTENCE), dtype='int32')
    clicked_entity_input = Input(shape=(MAX_SENTS,max_entity_num,100))
    clicked_one_hop_input = Input(shape=(MAX_SENTS,max_entity_num,max_neibor_num,100))

    clicked_relation_input = Input(shape=(MAX_SENTS, max_entity_num, max_neibor_num), dtype='int32')


    title_inputs = Input(shape=(MAX_SENTENCE,),dtype='int32')
    entity_inputs = Input(shape=(max_entity_num,100),dtype='float32')
    one_hop_inputs = Input(shape=(max_entity_num,max_neibor_num,100),dtype='float32')

    relation_inputs = Input(shape=(max_entity_num, max_neibor_num), dtype='int32')
    
    
    clicked_title_word_vecs = TimeDistributed(contexts_encoder)(clicked_title_input)
    candi_title_word_vecs = contexts_encoder(title_inputs)
        
    att_layer1 = Dense(200,activation='tanh')
    att_layer2 = Dense(1)
    
    clicked_title_att_vecs = TimeDistributed(att_layer1)(clicked_title_word_vecs) #(bz,50,30,128)
    clicked_title_att = TimeDistributed(att_layer2)(clicked_title_att_vecs) #(bz,50,30,1)
    clicked_title_att = keras.layers.Reshape((MAX_SENTS,MAX_SENTENCE,))(clicked_title_att) #(bz,50,30)
    
    candi_title_att_vecs= att_layer1(candi_title_word_vecs) #(bz,30,128)
    candi_title_att = att_layer2(candi_title_att_vecs) #(bz,30,1)
    candi_title_att0 = keras.layers.Reshape((MAX_SENTENCE,))(candi_title_att) #(bz,30)
    candi_title_att = keras.layers.RepeatVector(MAX_SENTS)(candi_title_att0) #(bz,50,30)
    
    clicked_title_att_vecs = keras.layers.Reshape((MAX_SENTS*MAX_SENTENCE,-1))(clicked_title_att_vecs) #(bz,50*30,128) 
    candi_title_att_vecs = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))(candi_title_att_vecs) #(bz,128,30)
    cross_att = keras.layers.Dot(axes=[-1,-2])([clicked_title_att_vecs,candi_title_att_vecs]) #(bz,50*30,30)
    cross_att_candi = keras.layers.Softmax()(cross_att) #(bz,50*30,30)
    cross_att_candi = keras.layers.Dot(axes=-1)([cross_att_candi,candi_title_att0]) #(bz,50*30,)
    cross_att_candi = Reshape((MAX_SENTS,MAX_SENTENCE))(cross_att_candi)
    cross_att_candi = Lambda(lambda x:0.001*x)(cross_att_candi)
    
    clicked_title_att = Add()([clicked_title_att,cross_att_candi]) #(bz,)
    clicked_title_att = keras.layers.Softmax()(clicked_title_att) #(bz,50,30)
    
    cross_att_click = keras.layers.Reshape((MAX_SENTS,MAX_SENTENCE,MAX_SENTENCE))(cross_att) #(bz,#click,#click_word,#candi_word)
    cross_att_click = Lambda(lambda x:K.permute_dimensions(x,(0,1,3,2)))(cross_att_click) #(bz,#click,#candi_word,#click_word,)
    clicked_title_att_re = Reshape((MAX_SENTS,1,MAX_SENTENCE))(clicked_title_att) #(bz,#click,1,#click_word,)
    cross_att_click_vecs = Concatenate(axis=-2)([cross_att_click,clicked_title_att_re]) #(bz,#click,#candi_word+1,#click_word,)
    cross_att_click = TimeDistributed(get_agg())(cross_att_click_vecs) #(bz,#click,#candi_word,)
    cross_att_click = Lambda(lambda x:0.001*x)(cross_att_click)
 
    candi_title_att = Add()([candi_title_att,cross_att_click])
    candi_title_att  = keras.layers.Softmax()(candi_title_att)

    
    candi_title_vecs = keras.layers.Dot(axes=[-1,-2])([candi_title_att,candi_title_word_vecs])

    clicked_title_att = keras.layers.Reshape((MAX_SENTS,MAX_SENTENCE,1))(clicked_title_att)
    clicked_title_word_vecs_att = keras.layers.Concatenate(axis=-1)([clicked_title_word_vecs,clicked_title_att])
    clicked_title_vecs = TimeDistributed(get_context_aggergator())(clicked_title_word_vecs_att)


    relation_encoder = get_relation_encoder(100)



    clicked_relation_input_reshape = keras.layers.Reshape((MAX_SENTS*max_entity_num,max_neibor_num))(clicked_relation_input)    #  [bz,500,5]

    clicked_relation_vec = TimeDistributed(relation_encoder)(clicked_relation_input_reshape) # [bz,500,5,100]
    clicked_relation_vec = keras.layers.Reshape((MAX_SENTS, max_entity_num, max_neibor_num, 100))(clicked_relation_vec) # [bz,50,10,5,100]

    candi_relation_vecs = TimeDistributed(relation_encoder)(relation_inputs)  # 【bz,10,5,100]


    clicked_entity_embedding = tf.expand_dims(clicked_entity_input, axis=3) # [bz,50,10,1,100]
    entity_embedding = tf.expand_dims(entity_inputs, axis=2) # [bz,10,1,100]

    clicked_entity_embedding = K.tile(clicked_entity_embedding, [1, 1, 1, 5, 1])
    entity_embedding = K.tile(entity_embedding, [1, 1, 5, 1])

    clicked_concat = keras.layers.Concatenate(axis=-1)([clicked_entity_embedding, clicked_relation_vec, clicked_one_hop_input])
    news_concat = keras.layers.Concatenate(axis=-1)([entity_embedding, candi_relation_vecs, one_hop_inputs])

    KGAT_dense1 = Dense(128)
    KGAT_dense2 = Dense(1)

    clicked_concat = KGAT_dense1(clicked_concat)
    news_concat = KGAT_dense1(news_concat)

    clicked_concat = keras.layers.Activation('relu')(clicked_concat)  # Relu
    news_concat = keras.layers.Activation('relu')(news_concat)  # Relu


    clicked_concat = KGAT_dense2(clicked_concat)  # [bz,50,10,5,1]
    news_concat = KGAT_dense2(news_concat) # [bz,10,5,1]

    clicked_concat = keras.layers.Reshape((50, 10, 5))(clicked_concat)  # (bz,50,10,5)
    news_concat = keras.layers.Reshape((10, 5))(news_concat)  # (bz,10,5)

    clicked_neibor_weight = keras.layers.Softmax()(clicked_concat)# (bz,50,10,5)
    news_neibor_weight = keras.layers.Softmax()(news_concat) # (bz,10,5)


    clicked_neibor_weight = keras.layers.Reshape((50, 10, 1, 5))(clicked_neibor_weight)  # (bz,50,1,5)
    news_neibor_weight = keras.layers.Reshape((10, 1, 5))(news_neibor_weight)  # (bz,10,1,5)

    clicked_neibor_vec = tf.matmul(clicked_neibor_weight, clicked_one_hop_input)
    clicked_neibor_vec = keras.layers.Reshape((50, 10, 100))(clicked_neibor_vec)

    news_neibor_vec = tf.matmul(news_neibor_weight, one_hop_inputs)
    news_neibor_vec = keras.layers.Reshape((10, 100))(news_neibor_vec)


    clicked_entity_vec = keras.layers.Concatenate(axis=-1)([clicked_neibor_vec, clicked_entity_input])
    news_entity_vec = keras.layers.Concatenate(axis=-1)([news_neibor_vec, entity_inputs])

    KGAT_fuse = Dense(100)

    clicked_kgat = KGAT_fuse(clicked_entity_vec)
    news_kgat = KGAT_fuse(news_entity_vec)





    clicked_onehop = keras.layers.Reshape((MAX_SENTS,max_entity_num*max_neibor_num,100))(clicked_one_hop_input)
    clicked_entity = keras.layers.Concatenate(axis=-2)([clicked_onehop,clicked_entity_input,clicked_kgat])
    

    news_onehop = keras.layers.Reshape((max_entity_num*max_neibor_num,100))(one_hop_inputs)

    news_entity = keras.layers.Concatenate(axis=-2)([news_onehop,entity_inputs,news_kgat])

    news_entity = keras.layers.Reshape((max_entity_num*(max_neibor_num+2)*100,))(news_entity)

    news_entity = keras.layers.RepeatVector(MAX_SENTS)(news_entity)
    news_entity = keras.layers.Reshape((MAX_SENTS,max_entity_num*(max_neibor_num+2),100))(news_entity)
    
    entity_emb = keras.layers.Concatenate(axis=-2)([clicked_entity,news_entity])
    
    
    pair_graph = create_pair_pair(max_entity_num)
    
    entity_vecs = TimeDistributed(pair_graph)(entity_emb)
    
    user_entity_vecs = keras.layers.Lambda(lambda x:x[:,:,:100])(entity_vecs)
    news_entity_vecs = keras.layers.Lambda(lambda x:x[:,:,100:])(entity_vecs)
    
    user_vecs = keras.layers.Concatenate(axis=-1)([clicked_title_vecs,user_entity_vecs])
    user_vecs = MergeLayer(user_vecs)
    
    news_vecs = keras.layers.Concatenate(axis=-1)([candi_title_vecs,news_entity_vecs])
    news_vecs = MergeLayer(news_vecs)
    
    
    match_att_layer1 = Dense(100,activation='tanh')
    match_att_layer2 = Dense(1)
    match_reduce_layer = Dense(100)
    
    user_att1 = match_att_layer1(user_vecs)
    user_att1 = match_att_layer2(user_att1)
    user_att = keras.layers.Reshape((MAX_SENTS,))(user_att1)
    
    news_att1 = match_att_layer1(news_vecs)
    news_att1 = match_att_layer2(news_att1)
    news_att = keras.layers.Reshape((MAX_SENTS,))(news_att1)
    
    cross_user_vecs = match_reduce_layer(user_vecs)
    cross_news_vecs = match_reduce_layer(news_vecs)
    cross_news_vecs = keras.layers.Lambda(lambda x:K.permute_dimensions(x,(0,2,1)))(cross_news_vecs)
    cross_att = keras.layers.Dot(axes=(-1,-2))([cross_user_vecs,cross_news_vecs])
    
    cross_user_att = keras.layers.Softmax()(cross_att)
    cross_user_att = keras.layers.Dot(axes=(-1,-2))([cross_user_att,news_att1])
    cross_user_att = Reshape((MAX_SENTS,))(cross_user_att)
    cross_user_att = Lambda(lambda x:0.01*x)(cross_user_att)
    user_att = Add()([user_att,cross_user_att])
    user_att = Softmax()(user_att)
    
    cross_news_att = Lambda(lambda x : K.permute_dimensions(x,(0,2,1)))(cross_att)
    cross_news_att = keras.layers.Softmax()(cross_news_att)
    cross_news_att = keras.layers.Dot(axes=(-1,-2))([cross_news_att,user_att1])
    cross_news_att = Reshape((MAX_SENTS,))(cross_news_att)
    cross_news_att = Lambda(lambda x:0.01*x)(cross_news_att)
    news_att = Add()([news_att,cross_news_att])
    news_att = Softmax()(news_att)

    
    
    user_vec = keras.layers.Dot(axes=[-1,-2])([user_att,user_vecs])
    news_vec = keras.layers.Dot(axes=[-1,-2])([news_att,news_vecs])
    

    score = keras.layers.Dot(axes=-1)([user_vec,news_vec])
    
    
    model = Model([title_inputs,entity_inputs,one_hop_inputs,relation_inputs,clicked_title_input,clicked_entity_input,clicked_one_hop_input,clicked_relation_input],score)
    
    return model

def create_model(title_word_embedding_matrix):
    clicked_title_input = Input(shape=(MAX_SENTS,MAX_SENTENCE), dtype='int32')
    clicked_entity_input = Input(shape=(MAX_SENTS,max_entity_num,100))
    clicked_one_hop_input = Input(shape=(MAX_SENTS,max_entity_num,max_neibor_num,100))


    clicked_relation_input = Input(shape=(MAX_SENTS,max_entity_num,max_neibor_num), dtype='int32')

    
    
    title_inputs = Input(shape=(1+npratio,MAX_SENTENCE),dtype='int32')
    entity_inputs = Input(shape=(1+npratio,max_entity_num,100),dtype='float32')
    one_hop_inputs = Input(shape=(1+npratio,max_entity_num,max_neibor_num,100),dtype='float32')


    relation_inputs = Input(shape=(1+npratio,max_entity_num,max_neibor_num), dtype='int32')

    
    pair_model = create_pair_model(title_word_embedding_matrix)
    
    doc_score = []
    for i in range(1+npratio):
        ti = keras.layers.Lambda(lambda x:x[:,i,:,])(title_inputs)
        ei = keras.layers.Lambda(lambda x:x[:,i,:,:])(entity_inputs)
        eo = keras.layers.Lambda(lambda x:x[:,i,:,:,:])(one_hop_inputs)

        ri = keras.layers.Lambda(lambda x:x[:,i,:,:])(relation_inputs)

        score = pair_model([ti,ei,eo,ri,clicked_title_input,clicked_entity_input,clicked_one_hop_input,clicked_relation_input,])
        score = keras.layers.Reshape((1,1,))(score)
        doc_score.append(score)
    doc_score = keras.layers.Concatenate(axis=-2)(doc_score)
    doc_score = keras.layers.Reshape((1+npratio,))(doc_score)
    logit = keras.layers.Activation('softmax')(doc_score)
    
    model = Model([title_inputs,entity_inputs,one_hop_inputs,relation_inputs,clicked_title_input,clicked_entity_input,clicked_one_hop_input,clicked_relation_input],logit) #  KGAT
    
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=tf.keras.optimizers.Adam(lr=0.00005),
                  metrics=['acc'])
    
    return model,pair_model