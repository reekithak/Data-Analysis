#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os 
import warnings
warnings.simplefilter('ignore')


# In[46]:


import numpy as np
import pandas as pd


# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[49]:


rajini=os.listdir(r"C:\Users\admin\Desktop\Images_srm\Images_srm\rajinikanth")


# In[50]:


limit=100
raj_im=[None]*limit
j=0
for i in rajini:
    if(j<limit):
        raj_im[j]=imread("C:/Users/admin/Desktop/Images_srm/Images_srm/rajinikanth/"+i)
        j+=1
    else:
        break


# In[51]:


dhanush=os.listdir(r"C:\Users\admin\Desktop\Images_srm\Images_srm\dhanush")
limit=100
dan_im=[None]*limit
j=0
for i in dhanush:
    if(j<limit):
        dan_im[j]=imread("C:/Users/admin/Desktop/Images_srm/Images_srm/dhanush/"+i)
        j+=1
    else:
        break


# In[52]:


vijay=os.listdir(r"C:\Users\admin\Desktop\Images_srm\Images_srm\vijay")
limit=100
vij_im=[None]*limit
j=0
for i in vijay:
    if(j<limit):
        vij_im[j]=imread("C:/Users/admin/Desktop/Images_srm/Images_srm/vijay/"+i)
        j+=1
    else:
        break


# In[53]:


raj_gray=[None]*limit
j=0
for i in rajini:
    if(j<limit):
        raj_gray[j]=rgb2gray(raj_im[j])
        j+=1
    else:
        break


# In[54]:


imshow(raj_gray[1])


# In[55]:


dan_gray=[None]*limit
j=0
for i in dhanush:
    if(j<limit):
        dan_gray[j]=rgb2gray(dan_im[j])
        j+=1
    else:
        break


# In[56]:


imshow(dan_gray[1])


# In[57]:


vij_gray=[None]*limit
j=0
for i in vijay:
    if(j<limit):
        vij_gray[j]=rgb2gray(vij_im[j])
        j+=1
    else:
        break


# In[58]:


imshow(vij_gray[1])


# In[59]:


for j in range(100):
    rk=raj_gray[j]
    raj_gray[j]=resize(rk,(64,64))
imshow(raj_gray[1])


# In[60]:


for j in range(100):
    dn=dan_gray[j]
    dan_gray[j]=resize(dn,(64,64))
imshow(dan_gray[1])


# In[61]:


for j in range(100):
    vj=vij_gray[j]
    vij_gray[j]=resize(vj,(64,64))
imshow(vij_gray[1])


# In[62]:


len_of_raj=len(raj_gray)
im_size_raj=raj_gray[1].shape
flat_size_raj=im_size_raj[0] * im_size_raj[1]


# In[63]:


for i in range(len_of_raj):
    raj_gray[i]=np.ndarray.flatten(raj_gray[i]).reshape(flat_size_raj,1)
raj_gray=np.dstack(raj_gray)
raj_gray.shape


# In[64]:


raj_gray=np.rollaxis(raj_gray,axis=2,start=0)


# In[65]:


raj_gray=raj_gray.reshape(len_of_raj,flat_size_raj)


# In[66]:


raj_gray.shape

PANDAS { DATA FRAME }
# In[67]:


raj_data=pd.DataFrame(raj_gray)


# In[68]:


raj_data


# In[69]:


raj_data['Label']='Rajinikanth'


# In[70]:


len_of_dan=len(dan_gray)
im_size_dan=dan_gray[1].shape
flat_size_dan=im_size_dan[0] * im_size_dan[1]


# In[71]:


for i in range(len_of_dan):
    dan_gray[i]=np.ndarray.flatten(dan_gray[i]).reshape(flat_size_dan,1)
dan_gray=np.dstack(dan_gray)
dan_gray.shape


# In[72]:


dan_gray=np.rollaxis(dan_gray,axis=2,start=0)


# In[73]:


dan_gray.shape


# In[74]:


dan_gray=dan_gray.reshape(len_of_dan,flat_size_dan)


# In[75]:


dan_data=pd.DataFrame(dan_gray)


# In[76]:


dan_data


# In[77]:


dan_data['Label']='Dhanush'


# In[78]:


len_of_vij=len(vij_gray)
im_size_vij=vij_gray[1].shape
flat_size_vij=im_size_vij[0] * im_size_vij[1]


# In[79]:


for i in range(len_of_vij):
    vij_gray[i]=np.ndarray.flatten(vij_gray[i]).reshape(flat_size_vij,1)
vij_gray=np.dstack(vij_gray)
vij_gray.shape


# In[80]:


vij_gray=np.rollaxis(vij_gray,axis=2,start=0)
vij_gray.shape


# In[81]:


vij_gray=vij_gray.reshape(len_of_vij,flat_size_vij)


# In[82]:


vij_gray.shape


# In[83]:


vij_data=pd.DataFrame(vij_gray)


# In[84]:


vij_data['Label']='Vijay'


# In[85]:


vij_data


# In[86]:


actor_1=pd.concat([raj_data,dan_data])
actors=pd.concat([actor_1,vij_data])


# In[87]:


actors


# In[88]:


from sklearn.utils import shuffle 


# In[89]:


kollywood_actors=shuffle(actors).reset_index()


# In[90]:


kollywood_actors


# In[91]:


kollywood_actors=kollywood_actors.drop(['index'],axis=1)


# In[92]:


kollywood_actors


# In[93]:


kollywood_actors.to_csv(r"C:\Users\admin\Desktop\VAC_IMAGE\actors.csv")


# In[106]:


from sklearn import *
from sklearn.model_selection import train_test_split


# In[107]:


x=kollywood_actors.values[:,:-1]
y=kollywood_actors.values[:,-1]


# In[113]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=True)


# In[ ]:




