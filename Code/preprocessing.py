from hypers import *
from datetime import datetime
import time
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import numpy
import os
import json
import random
from tqdm import tqdm
import heapq


def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def shuffle(pn,labeler,pos):
    index=np.arange(pn.shape[0])
    pn=pn[index]
    labeler=labeler[index]
    pos=pos[index]
    
    for i in range(pn.shape[0]):
        index=np.arange(npratio+1)
        pn[i,:]=pn[i,index]
        labeler[i,:]=labeler[i,index]
    return pn,labeler,pos

def read_news(path,filenames):
    news={}
    category=[]
    subcategory=[]
    news_index={}
    index=1
    word_dict={}
    word_index=1
    with open(os.path.join(path,filenames),encoding='utf-8') as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id,vert,subvert,title= splited[0:4]
        news_index[doc_id]=index
        index+=1
        category.append(vert)
        subcategory.append(subvert)
        title = title.lower()
        title=word_tokenize(title)
        news[doc_id]=[vert,subvert,title]
        for word in title:
            word = word.lower()
            if not(word in word_dict):
                word_dict[word]=word_index
                word_index+=1
    category=list(set(category))
    subcategory=list(set(subcategory))
    category_dict={}
    index=0
    for c in category:
        category_dict[c]=index
        index+=1
    subcategory_dict={}
    index=0
    for c in subcategory:
        subcategory_dict[c]=index
        index+=1
    return news,news_index,category_dict,subcategory_dict,word_dict

def get_doc_input(news,news_index,category,subcategory,word_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_SENTENCE),dtype='int32')
    news_vert=np.zeros((news_num,),dtype='int32')
    news_subvert=np.zeros((news_num,),dtype='int32')
    for key in news:    
        vert,subvert,title=news[key]
        doc_index=news_index[key]
        news_vert[doc_index]=category[vert]
        news_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id].lower()]
        
    return news_title,news_vert,news_subvert


def new_load_entity_metadata(KG_root_path):
    with open(os.path.join(KG_root_path, 'entity2id.txt')) as f:
        lines = f.readlines()

    with open(os.path.join(KG_root_path, 'link2id.txt')) as f2:
        links = f2.readlines()

    EntityId2Index = {}
    EntityIndex2Id = {}
    LinksId2Index = {}
    LinksIndex2Id = {}
    for i in range(1, len(lines)):
        eid, eindex = lines[i].strip('\n').split('\t')
        EntityId2Index[eid] = int(eindex)
        EntityIndex2Id[int(eindex)] = eid

    for i in range(1, len(links)):
        lid, lindex = links[i].strip('\n').split('\t')
        LinksId2Index[lid] = int(lindex)
        LinksIndex2Id[int(lindex)] = lid


    entity_embedding = np.load(os.path.join(KG_root_path, 'entity_embedding.npy'))
    entity_embedding = np.concatenate([entity_embedding, np.zeros((1, 100))], axis=0)


    with open(os.path.join(KG_root_path, 'KG_Neighbors.json')) as f:
        s = f.read()
    graph = json.loads(s)

    return graph, EntityId2Index, EntityIndex2Id, entity_embedding, LinksId2Index, LinksIndex2Id

def load_entity_metadata(KG_root_path):
    #Entity Table
    with open(os.path.join(KG_root_path,'entity2id.txt')) as f:
        lines = f.readlines()
        
    EntityId2Index = {}
    EntityIndex2Id = {}
    for i in range(1,len(lines)):
        eid, eindex = lines[i].strip('\n').split('\t')
        EntityId2Index[eid] = int(eindex)
        EntityIndex2Id[int(eindex)] = eid
        
        
    entity_embedding = np.load(os.path.join(KG_root_path,'entity_embedding.npy'))
    entity_embedding = np.concatenate([entity_embedding,np.zeros((1,100))],axis=0)
    
    
    with open(os.path.join(KG_root_path,'KGGraph.json')) as f:
        s = f.read()
    graph = json.loads(s)

    return graph, EntityId2Index, EntityIndex2Id, entity_embedding




def load_news_entity(news,EntityId2Index,data_root_path):
    with open(os.path.join(data_root_path,'docs.tsv'),encoding='utf-8') as f:
        lines = f.readlines()

    news_entity = {}
    g = []
    for i in range(len(lines)):
        docid,_,_,_,_,_,entities,_ = lines[i].strip('\n').split('\t')
        entities = json.loads(entities)
        news_entity[docid] = []
        for j in range(len(entities)):
            e = entities[j]['Label']
            eid = entities[j]['WikidataId']
            if not eid in EntityId2Index:
                continue
            news_entity[docid].append([e,eid,EntityId2Index[eid]])

    return news_entity

def parse_zero_hop_entity(EntityId2Index,news_entity,news_index,max_entity_num = 5):
    news_entity_index = np.zeros((len(news_index)+1,max_entity_num),dtype='int32')+len(EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity[newsid]
        ri = np.random.permutation(len(entities))
        for j in range(min(len(entities),max_entity_num)):
            e = entities[ri[j]][-1]
            news_entity_index[index,j] = e
    return news_entity_index

def parse_one_hop_entity(EntityId2Index,EntityIndex2Id,news_entity_index,graph,news_index,max_entity_num=5):
    one_hop_entity = np.zeros((len(news_index)+1,max_entity_num,max_neibor_num),dtype='int32')+len(EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity_index[index]
        for j in range(max_entity_num):
            eindex = news_entity_index[index,j]
            if eindex==len(EntityId2Index):
                continue
            eid = EntityIndex2Id[eindex]
            neighbors = graph[eid]
            rindex = np.random.permutation(len(neighbors))
            for k in range(min(max_neibor_num,len(neighbors))):
                nindex = rindex[k]
                neig_id = neighbors[nindex]
                neig_index = EntityId2Index[neig_id]
                one_hop_entity[index,j,k] = neig_index
    return one_hop_entity





def new_parse_one_hop_entity(LinksId2Index, EntityId2Index, EntityIndex2Id, news_entity_index, graph, news_index,max_entity_num=5):
    one_hop_entity = np.zeros((len(news_index) + 1, max_entity_num, max_neibor_num), dtype='int32') + len(EntityId2Index)
    one_hop_link = np.zeros((len(news_index) + 1, max_entity_num, max_neibor_num), dtype='int32') + len(LinksId2Index)

    if (os.path.exists("/home/mist/datasets/KGData/relation_weight.txt")):
        links_weight = numpy.loadtxt("/home/mist/datasets/KGData/relation_weight.txt", delimiter=',')
        weight_exist = 1
    else:
        weight_exist = 0

    for newsid in tqdm(news_index):
        index = news_index[newsid]
        entities = news_entity_index[index]
        for j in range(max_entity_num):
            eindex = news_entity_index[index, j]
            if eindex == len(EntityId2Index):
                continue
            eid = EntityIndex2Id[eindex]
            neighbors = graph[eid]



            if (len(neighbors) <= max_neibor_num):
                for k in range(len(neighbors)):
                    neig_id, link_id = neighbors[k]
                    neig_index = EntityId2Index[neig_id]
                    link_index = LinksId2Index[link_id]

                    one_hop_entity[index, j, k] = neig_index
                    one_hop_link[index, j, k] = link_index


            else:


                if(weight_exist==1):
                    links_weight_list = []
                    for k in range(len(neighbors)):
                        neig_id, link_id = neighbors[k]

                        neig_index = EntityId2Index[neig_id]
                        link_index = LinksId2Index[link_id]
                        link_weight = links_weight[link_index]
                        links_weight_list.append([link_weight, neig_index, link_index])


                    re2 = heapq.nlargest(max_neibor_num, links_weight_list)


                    for k in range(max_neibor_num):
                        one_hop_entity[index, j, k] = re2[k][1]
                        one_hop_link[index, j, k] = re2[k][2]
                else:
                    rindex = np.random.permutation(len(neighbors))
                    for k in range(max_neibor_num):
                        nindex = rindex[k]
                        neig_id, link_id = neighbors[nindex]
                        neig_index = EntityId2Index[neig_id]
                        link_index = LinksId2Index[link_id]

                        one_hop_entity[index, j, k] = neig_index
                        one_hop_link[index, j, k] = link_index



    return one_hop_entity, one_hop_link

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word

def read_clickhistory(data_root_path,news_index,filename):
    
    lines = []
    userids = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user

def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)

    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        #index = np.random.randint(1+npratio)
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label


def get_test_input(news_index,session):
    
    DocIds = []
    UserIds = []
    Labels = []
    Bound = []
    count = 0
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        start = count
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            DocIds.append(docid)
            Labels.append(1)
            UserIds.append(sess_id)
            count += 1
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            DocIds.append(docid)
            Labels.append(0)
            UserIds.append(sess_id)
            count+=1
        Bound.append([start,count])
        
    DocIds = np.array(DocIds,dtype='int32')
    UserIds = np.array(UserIds,dtype='int32')
    Labels = np.array(Labels,dtype='float32')

    return DocIds, UserIds, Labels, Bound