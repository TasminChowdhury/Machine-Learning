
# coding: utf-8

# In[1]:

import numpy as np
# simple dataset
y= [[[1,0,0,1],[2,0,1,-1]],[[1,0,0,2],[2,0,1,1]]]
print(y[0][0])
print(y[0][1])
print(y[1][0])
print(y[1][1])
lamda0 = []


# In[2]:

def diff(k):
    j= []
    for i in range(len(k)-1):
        temp = k[i+1] - k[i]
        j.append(temp)
    return j


# In[4]:

i=0
j=0
first = []
second = []
temp = []
dely = []
for i in range (2):
    for j in range (2):
        #feature vector
        first= diff(y[i][j])
        second= diff(first)
        #print(first)
        #print(second)
        temp.append(second)
        #print(temp)
    dely.append(temp)
    temp = []

print(dely)

i=0
j=0
Vu = []
summ = np.array([0,0])
mean = []
for i in range (2):
    for j in range (2):
        # temp = dely[i]
        # temp = np.array(temp).reshape(2,1) + np.array(dely[i][j]).T[np.newaxis]
        #print (temp)
        #print(np.array(dely[i][j]).T)
        summ = summ + np.array(dely[i][j]).T
        #summ = summ/2
        
        
        
        # print (temp)
    summ = summ/2
    mean.append(summ)
    summ2d = np.array(summ,ndmin=2)
    summ2d =np.matmul(summ2d.T,summ2d)
    Vu.append(np.matrix(summ2d))
    #print (summ)
    summ = np.array([0,0])
    #Vu = temp * temp.T


lamda0.append(Vu)
print(lamda0)



# In[5]:

#np.matmul(np.array([2,-1],ndmin=2).T,np.array([2,-1],ndmin=2))
#diffmean = []
diffmean = np.array([0,0])
m=4
Ve = np.array([0,0])
for i in range (2):
    for j in range (2):
        #print (np.array(dely[i][j]).T)
        #print (np.array(mean[i]))
        diffmean= dely[i][j] - mean[i]
        print(dely[i][j])
        print(mean[i])
        diffmean = np.array(diffmean,ndmin=2)
        diffmean2d =np.matmul(diffmean.T,diffmean)
        print(diffmean2d) 
        Ve = Ve + diffmean2d
Ve= np.matrix(Ve/4)
print (Ve)
lamda0.append(Ve)


# In[ ]:

print("Ve",Ve)
A= np.matrix(Ve).I
print("A" , A)
B = []
U = []
#print(Vu)
Vetemp = np.array([0,0])
for i in range (2):
    Btemp1 = 2 * Vu[i] + Ve
    #print(Vu[i])
    #print(Btemp1)
    #print(np.matrix(Btemp1).I)
    Btemp2 = -(np.matrix(Btemp1).I  * Vu[i] * A)
    B.append(Btemp2)
    Btemp1 = []
    Btemp = []
print("B",B)
#print( Vu)
vediff = 1
Venew = []

while (vediff >.0001):
    for i in range (2) :
        d = np.add(dely[i][0], dely[i][1])
        print(d)
        E1 =np.matmul(d.T , Vu[i] * (A + 2* B[i])) 
        print(E1)
        E1E1T= np.matmul(E1.T,E1)
        print(E1E1T)
        U.append(E1E1T)
        for j in range (2):
            E2 = np.matmul(d.T, Ve * B[i] )
            #print(E2)
            E3 = np.matmul(np.array(dely[i][j]).T, Ve * A )
            #print(E3)
            E2 = E2 + E3
            #print (E2)
            E2E2T = np.matmul(E2.T,E2)
            Vetemp = Vetemp + E2E2T
    Venew = Vetemp/4 
    #print(Venew)
    #print(Ve)
    vediff = Venew - Ve
    #print(vediff)
    vediff = np.sqrt(np.trace(np.matmul(vediff.T,vediff)))
    print ("vediff", vediff)
    Ve= Venew 
    A= np.matrix(Ve).I
    B = []
    for i in range (2):
        Btemp1 = 2 * U[i] + Ve
        print(Vu[i])
        print(Btemp1)
        #print(np.matrix(Btemp1).I)
        Btemp2 = -(np.matrix(Btemp1).I  * U[i] * A)
        B.append(Btemp2)
        Btemp1 = []
        Btemp = []
    print(B) 
  
    


# In[156]:

sigma = np.array([0,0])
for i in range (2) :
    sigma = Vu[i] + Ve
print(sigma)


