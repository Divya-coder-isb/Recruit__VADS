#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv (r"D:\1 ISB\Term 2\FP\FP project\Resume data.csv")


# In[2]:


df ['Experience'] = np.random.randint (1, 31, size=len (df))


# In[3]:


certifications = ["CSPO", "PMP", "Google Project Management", "PfMP", "Agile certified", "Certified scrum master", "PgMP", "CAPM", "Six sigma", "C++", "Java", "Python", "BPA", "AIPMM", "CBAP"]
df ['Certification'] = df.apply (lambda x: ", ".join (np.random.choice (certifications, size=np.random.randint (1, 6), replace=False)), axis=1)


# In[4]:


# Create a new column "Email ID" by combining candidate name with @gmail.com
df['Email ID'] = df['Candidate Name'].str.replace(' ', '').str.lower() + '@gmail.com'


# In[5]:


print (df)


# In[6]:


df.to_csv("Modifiedresumedata_data.csv", header=True, index=False)

