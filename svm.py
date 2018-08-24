#!/usr/bin/env python
from sklearn import svm
import random
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import tree
import numpy as np
import xgboost as xgb

cat_to_id = {'投诉（含抱怨）':0, '办理':1, '咨询（含查询）':2, '其他':3,'非来电':3,'表扬及建议':3}
id_to_label = {0:'投诉（含抱怨）', 1:'办理',2:'咨询（含查询）',3:'其他'}

comma_tokenizer = lambda x: jieba.cut(x,cut_all=True)
sel = VarianceThreshold(threshold=(0.01*(1-0.01)))

#读取文件
def readfile(filename):
    f = open(filename,'r')
    lines = f.readlines()
    sents,labels = [],[]
    for i in lines:
        label , sent = i.strip().split('\t')
        sents.append(sent.strip())
        labels.append(cat_to_id[label.strip()])
    return sents , labels

#提取特征
def covectorize(train_words,test_words,re_low=False,sel_k=False):
    v = CountVectorizer(tokenizer=comma_tokenizer)
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    #移除低方差特征
    if re_low:
        train_data = sel.fit_transform(train_data)
        test_data = sel.transform(test_data)
        print(train_data.shape)
    #选择前k个特征
    elif sel_k:
        ch2 = SelectKBest(chi2,k=1000)
        train_data = ch2.fit_transform(train_data,train_labels)
        test_data = ch2.transform(test_data)
        
    return train_data,test_data

def tfVectorize(train_words,test_words,re_low=False,sel_k=False):
    v = TfidfVectorizer(tokenizer=comma_tokenizer)
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    print(train_data.shape)
    #移除低方差特征
    if re_low:
        train_data = sel.fit_transform(train_data)
        test_data = sel.transform(test_data)
        print(train_data.shape)
    #选择前k个特征
    elif sel_k:
        ch2 = SelectKBest(chi2,k=1000)
        train_data = ch2.fit_transform(train_data,train_labels)
        test_data = ch2.transform(test_data)
    
    return train_data, test_data

#创建svm分类器
def train_clf(train_data, train_labels,use_svm=False,use_nb=False,use_rnd_forest=False):
    if use_svm:
        clf=svm.SVC(C=15)
        #clf=svm.LinearSVC()
        #clf=svm.LinearSVC(C=10,penalty='l2')
    #朴素贝叶斯
    elif use_nb:
        #clf = GaussianNB()
        #clf = MultinomialNB()
        clf = BernoulliNB(alpha=0.5)
    #随机森林
    elif use_rnd_forest:
        clf = RandomForestClassifier(n_estimators=200)
    
    
    clf.fit(train_data,np.asarray(train_labels))
    return clf

#xgboost
def xgbt(train_data,train_labels,test_data):
    params={

    'booster':'gbtree',
    #'objective': 'multi:softprob', #多分类的问题
    'objective': 'multi:softmax', #多分类的问题
    'num_class':4, # 类别数，与 multisoftmax 并用
    #'gamma':0.5,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    #'max_depth':6, # 构建树的深度，越大越容易过拟合
    'lambda':5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #'subsample':1, # 随机采样训练样本
    #'colsample_bytree':0.7, # 生成树时进行的列采样
    #'min_child_weight':2, # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
    'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
    #'eta': 0.6, # 如同学习率
    #'seed':1000,
    #'nthread':7,# cpu 线程数
    #'eval_metric': 'auc'
    }
    dtrain = xgb.DMatrix(train_data,label = np.asarray(train_labels))
    dtest = xgb.DMatrix(test_data,missing=-999.0)
    num_rounds=1000
    plst = list(params.items())
    bst = xgb.train(plst,dtrain,num_rounds)
#保存模型
    #bst.save_model('test.model')
    #bst.dump_model('dump.raw.txt')
    #bst.dump_model('dump.raw.txt','featmap.txt')
    return bst, dtrain, dtest

#测试
def evaluate(actual, pred):
        m_precision = metrics.accuracy_score(actual, pred)
        #m_recall = metrics.recall_score(actual,pred,average='macro')
        print ('precision:{0:.3f}'.format(m_precision))
        #print ('recall:{0:0.3f}'.format(m_recall))


        

if __name__ == '__main__':
    train_sents, train_labels = readfile(train_path)
    test_sents, test_labels = readfile(test_path)
    train_data , test_data = covectorize(train_sents, test_sents) 
#训练
    clf = train_clf(train_data,train_labels,use_svm=True)
    #clf,train_data,test_data = xgbt(train_data,train_labels,test_data)

#预测
    tre = clf.predict(train_data)
    re = clf.predict(test_data)
    print('train score')
    evaluate(np.asarray(train_labels),tre)
    print('test score')
    evaluate(np.asarray(test_labels),re)
