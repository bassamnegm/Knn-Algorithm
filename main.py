import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math 
import operator
def plotfigre(dataname):
    fig=plt.figure()
    ax=fig.add_subplot()
    data=pd.read_csv(dataname, names=['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
    ind=0
    label1=0
    label2=0
    #loop to get index of class 1
    for v in data['target']:
        if v=="1":
            label1=ind
            break
        ind+=1
    ind=0
    #loop to get index of class 0
    for v in data['target']:
        if v=='0':
            label2=ind
            break
        ind+=1
            
    fig, ax = plt.subplots()
    data1=data.iloc[label1:label1+10]
    data2=data.iloc[label2:label2+10]
    
   
    ax.scatter(data2['age'], data2['trestbps'],color='m',marker="*",label="0")
    ax.scatter(data1['age'], data1['trestbps'],color='r',marker="p",label="1")
    plt.legend(loc='upper left')
    ax.set_title('plot dataset')
    ax.set_xlabel('age')
    ax.set_ylabel('trestbps')
    plt.show()




filedata='datasets.csv'
plotfigre(filedata)
data = pd.read_csv(filedata)


def euclideanDistance(point1, point2, length):
    distance = 0
    for x in range(length):
        distance += np.square(point1[x] - point2[x])
    return np.sqrt(distance)

def knn(training, test, k):
    freqancy = {}
    distances = {}
    sort = {}
    length = test.shape[1]
    neighbors = []
    for x in range(len(training)):
        dist = euclideanDistance(test, training.iloc[x], length)
        distances[x] = dist[0]
    sort = sorted(distances.items(), key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(sort[x][0])

    for x in range(len(neighbors)):
        label = training.iloc[neighbors[x]][-1]
 
        if label in freqancy:
            freqancy[label] += 1
        else:
            freqancy[label] = 1
  
    freqancysort = sorted(freqancy.items(), key=operator.itemgetter(1), reverse=True)
    return freqancysort[0][0]



train = data.sample(frac=0.60, random_state=0) #spilt to train data


k = 3 #k of neighbors

test = data.drop(train.index)


accurcy=0
rsme=0
total=0
num=test.shape[0]

for i in range(num):
    testonerow = pd.DataFrame(np.array(test.iloc[i:i+1,:-1]).reshape(-1,len(test.iloc[i:i+1,:-1])))
    getlabel=list(np.array(test.iloc[i:i+1]))
    predction=getlabel[0][13]
    result=knn(train,testonerow,k)
    total+=(float(result)-float(predction))**2
    if result==predction:
        accurcy=1+accurcy
    
    print('\n',i,':','the predction of class to this point is  = ', result)
    print('\n',i,':','the real label of class to this object is = ', predction)
    print('--------------------------------------------------------------------')
rsme=math.sqrt(total/num)
print("accuracy of my model :",(accurcy/num)*100,"%")
print("the root mean square error of my model :",rsme)
print("the total numbers of them model can classify corecct :",accurcy)
print("the total numbers of them model can classify not corecct :",num-accurcy)









