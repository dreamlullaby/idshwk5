import math
import datetime
import sys
import numpy as np
import os
from sklearn import preprocessing
from sklearn.utils import shuffle
class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 900
        self.rate = 0.25
        self.domains=[]
        self.feats = []
        self.labels = []
        self.domains_test=[]
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []

    def loadDataSet(self, file_name ,label_existed):
        domains = []
        labels = []
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                temp = []
                token = line.strip().split(',')
                domains.append(token[0])
                if label_existed == 1:
                    if token[-1]=='notdga':
                        labels.append(0)
                    else:
                        labels.append(1)          
        domains = np.array(domains)
        labels = np.array(labels)
        return domains, labels

    def loadTrainData(self):
        self.domains, self.labels = self.loadDataSet(self.train_file,1)

    def loadTestData(self):
        self.domains_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    def savePredictResult(self,flag=0):
        if flag==0:
            print(self.labels_predict)
            with open(self.predict_result_file, 'w') as f:
                for i in range(len(self.labels_predict)):
                    f.write(str(self.labels_predict[i])+"\n")
        else:    
            print(self.labels_predict)
            with open(self.predict_result_file, 'w') as f:
                for i in range(len(self.labels_predict)):
                    if self.labels_predict[i]==0:
                        label='notdga'
                    else:
                        label='dga'
                    f.write(str(self.domains_test[i])+','+label+"\n")    

    def divideSet(self):
        self.loadTrainData()
        num=int(0.8*len(self.domains))
        with open(r"data\test2.txt",'w') as f1:
            with open(r"data\ans.txt",'w') as f2:
                print(num,len(self.domains))
                for i in range(num,len(self.domains)):
                    f1.write(str(self.domains[i])+"\n")
                    f2.write(str(self.labels[i])+"\n")

    def calLen(self,domain)->int:
        #prefix=domain.split('.')[0]
        prefix=domain
        return len(prefix)
    
    def calEntropy(self,domain):
	    tmp_dict={}
	    domain_len = len(domain)
	    for i in range(0,domain_len):
	    	if domain[i] in tmp_dict.keys():
	    		tmp_dict[domain[i]] = tmp_dict[domain[i]] + 1
	    	else:
	    		tmp_dict[domain[i]] = 1
	    shannon = 0
	    for i in tmp_dict.keys():
	    	p = float(tmp_dict[i]) / domain_len
	    	shannon = shannon - p * math.log(p,2)
	    return shannon

    def calNumratio(self,domain):
        _len=len(domain)
        count=0
        for i in range(len(domain)):
            if domain[i].isdigit():
                count+=1
        return count/_len

    def genFeats(self):
        self.loadTestData()
        self.loadTrainData()
        feats=[]
        feats_test=[]
        for domain in self.domains:
            _len=self.calLen(domain)
            detropy=self.calEntropy(domain)
            ratio=self.calNumratio(domain)
            feats.append([_len,detropy,ratio])
            #可多线程
        for domain in self.domains_test:
            _len=self.calLen(domain)
            detropy=self.calEntropy(domain)
            ratio=self.calNumratio(domain)
            feats_test.append([_len,detropy,ratio])
        self.feats=preprocessing.scale(np.array(feats))
        self.feats_test=preprocessing.scale(np.array(feats_test))


    def sigmod(self, x):
        return 1/(1+np.exp(-x))

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        self.weight = np.ones((self.param_num,), dtype=np.float)

    def compute(self, recNum, param_num, feats, w):
        return self.sigmod(np.dot(feats, w))

    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self,flag):
        preval = self.compute(len(self.feats_test),
                              self.param_num, self.feats_test, self.weight)
        self.labels_predict = (preval+0.5).astype(np.int)
        self.savePredictResult(flag)

    def train(self):
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        #ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for i in range(self.max_iters):
            preval = self.compute(recNum, self.param_num,
                                  self.feats, self.weight)
            sum_err = self.error_rate(recNum, self.labels, preval)
            #if i%30 == 0:
                #print("Iters:" + str(i) + " error:" + str(sum_err))
                #theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
                #print(theTime)
            err = self.labels - preval
            delt_w = np.dot(self.feats.T, err)
            delt_w /= recNum
            self.weight += self.rate*delt_w


if __name__=="__main__":
    train_file =  r"train.txt"
    test_file = r"test.txt"
    predict_file = r"result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.genFeats()
    lr.train()
    lr.predict(1)

