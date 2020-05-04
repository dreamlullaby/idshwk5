from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []
testlist=[]
def calEntropy(domain):
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
    	shannon = shannon - p * np.log(p,2)
    return shannon

def calNumratio(domain):
        _len=len(domain)
        count=0
        for i in range(len(domain)):
            if domain[i].isdigit():
                count+=1
        return count/_len

class Domain:
	def __init__(self,_domain,_label):
		self.domain = _domain
		self.label = _label
		self.len,self.numratio,self.entropy=len(_domain),calNumratio(_domain),calEntropy(_domain)

	def returnData(self):
		return [self.len, self.numratio, self.entropy]

	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1
		
def initData(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			domain = tokens[0]
			label = tokens[1]
			domainlist.append(Domain(domain,label))

def initTest(filename):
	with open(filename) as f:
		for line in f:
			if line.startswith("#") or line =="":
				continue
			testlist.append(line)

def main():
	initData(r"train.txt")
	featureMatrix = []
	labelList = []
	for item in domainlist:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())

	initTest(r"test.txt")
	testfeats=[]
	for domain in testlist:
		testfeats.append([len(domain),calNumratio(domain),calEntropy(domain)])
	
	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	testlabels=clf.predict(testfeats)
	output=list(zip(testlist,testlabels))
	with open(r"result.txt",'w') as f:
		for domain,label in output:
			line=domain+','
			if label==0:
				line+='notdga\n'
			else:
				line+='dga\n'
			f.write(line)

if __name__ == '__main__':
	main()

