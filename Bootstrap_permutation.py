#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[2]:


stimulus_type = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
responses = np.array([-1.64, -0.79, -0.31, -0.3 ,  1.78,  0.39,  0.56,  0.76, -0.27, -1.  ,  1.87, -1.51,  
                      0.37, -0.51,  1.56,  0.8 ])


# In[3]:


# as before, separate the data into responses_1 and responses_2
responses_1 = responses[stimulus_type == 1]
responses_2 = responses[stimulus_type == 2]


# # bootstrap

# In[4]:


# compute the standard error of the mean (SEM) for each set of responses
sem_1 = responses_1.std() / np.sqrt(len(responses_1))
sem_2 = responses_2.std() / np.sqrt(len(responses_1))


# In[5]:


# create empty arrays to store boostrap means
nboots = 10000
bs_means_1 = np.zeros(nboots)
bs_means_2 = np.zeros(nboots)


# In[6]:


# 10000 bootstraps!
for ii in range(nboots):
    # sample (with replacement) the same number of items as in original dataset
    # then save the bootstrap sample mean into the empty array
    
    bs_responses_1 = np.random.choice(responses_1, len(responses_1))
    bs_means_1[ii] = bs_responses_1.mean()
    
    bs_responses_2 = np.random.choice(responses_2, len(responses_2))
    bs_means_2[ii] = bs_responses_2.mean()


# In[8]:


# the SEM of our data is the std of the bootstrap means
bs_sem_1 = bs_means_1.std()
bs_sem_2 = bs_means_2.std()


# In[9]:


# very accurate!
print(sem_1, bs_sem_1)
print(sem_2, bs_sem_2)


# # permutation test

# In[6]:


# t-test (assuming a Gaussian distribution for our population)
# should get p = 0.02
from scipy.stats import ttest_ind

ttest_ind(responses_1, responses_2)


# In[7]:


# create empty array to store permutation results
nperms = 10000
perm_mean_diffs = np.zeros(nperms)


# In[20]:


# for a single permutation test, mix the oiginal responses...
perm_data = np.random.permutation(responses)
# ...and split into 2 parts for responses 1 and 2 with np.split 
# (came up during practice today, not required to know)
perm_responses_1, perm_responses_2 = np.split(perm_data, 2)

# OR, just use the simple version:
perm_responses_1 = perm_data[:8]
perm_responses_2 = perm_data[8:]


# In[12]:


# just showing the np.split result here so yall know what it does
print(perm_data)
print(perm_responses_1, perm_responses_2)


# In[13]:


# 10000 perm tests!
for ii in range(nperms):
    perm_data = np.random.permutation(responses)
    perm_responses_1, perm_responses_2 = np.split(perm_data, 2)
    
    # store the difference between the two means into the empty array
    perm_mean_diffs[ii] = perm_responses_1.mean() - perm_responses_2.mean()


# In[18]:


# plot the histogram of our empirical null distribution from bootstrap

# bins=50: use 50 bins to see the more fine grained counts
plt.hist(perm_mean_diffs, bins=50);
# plot the mean difference from our own data
actual_diff = responses_1.mean() - responses_2.mean()
plt.vlines(x=actual_diff, ymin=0, ymax=500, color='k')

# plot the -1 * actual diff, for calculating p-values below
plt.vlines(x=-actual_diff, ymin=0, ymax=500, color='grey')


# In[19]:


# to calculating a 2-tailed p-value, we look at the fraction of permutations that 
# give mean differences more extreme than the actual value
# (smaller than the black line and larger than the grey line)

# fraction of tests smaller than actual value (to the left of black line)
frac_smaller = (perm_mean_diffs < actual_diff).mean()

# fraction of tests bigger than -1 * actual value (to the right of grey line)
frac_bigger = (perm_mean_diffs > (-actual_diff)).mean()

# add these together to get the p-value
perm_pvalue = frac_smaller + frac_bigger
print(perm_pvalue)

