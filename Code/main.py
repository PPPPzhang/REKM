import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tfv1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras.backend as KTF

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

KTF.set_session(session)


tf.compat.v1.disable_eager_execution()




from hypers import *
from utils import *
from generator import *
from models import *
from preprocessing import *

data_root_path = "/home/mist/datasets/MINDout"
embedding_path = "/home/mist/datasets/embedding"
KG_root_path = "/home/mist/datasets/KGData"


news,news_index,category_dict,subcategory_dict,word_dict = read_news(data_root_path,'docs.tsv')

news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)

title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)



new_graph, EntityId2Index, EntityIndex2Id, entity_embedding, LinksId2Index, LinksIndex2Id = new_load_entity_metadata(KG_root_path)

news_entity = load_news_entity(news,EntityId2Index,data_root_path)

news_entity_index = parse_zero_hop_entity(EntityId2Index,news_entity,news_index,max_entity_num)



one_hop_entity,one_hop_relation = new_parse_one_hop_entity(LinksId2Index,EntityId2Index,EntityIndex2Id,news_entity_index,new_graph,news_index,max_entity_num)

train_session = read_clickhistory(data_root_path,news_index,'train.tsv')

train_user = parse_user(news_index,train_session)

train_sess, train_user_id, train_label = get_train_input(news_index,train_session)



train_generator = get_hir_train_generator(news_title,news_entity_index,one_hop_entity,one_hop_relation,entity_embedding,train_user['click'],train_user_id,train_sess,train_label,16)


test_session = read_clickhistory(data_root_path,news_index,'test.tsv')

test_user = parse_user(news_index,test_session)

test_docids, test_userids, test_labels, test_bound = get_test_input(news_index,test_session)


test_generator = get_test_generator(test_docids,test_userids,news_title,news_entity_index,one_hop_entity,one_hop_relation,entity_embedding,test_user['click'],64)




valid_session = read_clickhistory(data_root_path,news_index,'valid.tsv')
valid_user = parse_user(news_index,valid_session)
valid_sess, valid_user_id, valid_label = get_train_input(news_index,valid_session)
valid_generator = get_hir_train_generator(news_title,news_entity_index,one_hop_entity,one_hop_relation,entity_embedding,valid_user['click'],valid_user_id,valid_sess,valid_label,16)





best_val_loss = 1000
def SaveBetter(history):
    global best_val_loss
    valid_loss = history.history['val_loss'][-1]

    if (valid_loss <= best_val_loss):
        best_val_loss = valid_loss

        model_8 = model.get_layer(name='model_8')
        model_weights = model_8.get_weights()
        np.savez("/home/mist/datasets/Model_Weights/weights_REKM.npz", *model_weights)


EPOCH = 6
model,inter_model = create_model(title_word_embedding_matrix)



history6 = model.fit_generator(train_generator, epochs=EPOCH, validation_data=valid_generator)
SaveBetter(history6)

for i in range(3):
    history_turn = model.fit_generator(train_generator, epochs=1, validation_data=valid_generator)
    SaveBetter(history_turn)



model_8 = model.get_layer(name='model_8')
data = np.load("/home/mist/datasets/Model_Weights/weights_REKM.npz")
trained_weights = []
for arr in data.files:
    trained_weights.append(data[arr])
model_8.set_weights(trained_weights)



predicted_label = inter_model.predict_generator(test_generator,verbose=1)
result = evaluate(predicted_label,test_labels,test_bound)
print("预测结果：",result)
