#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[129]:


NAME = "Sonu Yadav"
COLLABORATORS = "Kevin Sheth", "Jared Lim"


# # Programming and Data Analysis in Modern Neuroscience: Problem Set 4

# In[130]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import stats
from statistics import median
import pandas as pd


# ---
# # Problem 1.
# Solve each part.

# In[131]:


# run this cell to create some data
my_data_1 = np.random.randn(100)
my_data_2 = my_data_1 + np.random.randn(100)


# In[132]:


# compute the mean of my_data_1 (1 pt)
### YOUR CODE HERE ###
mean=statistics.mean(my_data_1)
mean


# In[133]:


# compute the median of my_data_1 (1 pt)
### YOUR CODE HERE ###
median = statistics.median(my_data_1)
median


# In[134]:


# compute the variance (not sample variance) of my_data_1 (1 pt)
### YOUR CODE HERE ###
var = statistics.pvariance(my_data_1)
var


# In[135]:


# compute the SAMPLE variance of my_data_1 (1 pt)
### YOUR CODE HERE ###
import statistics
variance = statistics.variance(my_data_1)
variance


# In[136]:


# compute the standard deviation (not sample standard deviation) of my_data_1 (1 pt)
### YOUR CODE HERE ###
sd = statistics.pstdev(my_data_1)
sd


# In[137]:


# compute the SAMPLE standard deviation of my_data_1 (1 pt)
### YOUR CODE HERE ###
sam_sd = statistics.stdev(my_data_1)
sam_sd


# In[138]:


# compute the correlation of my_data_1 and my_data_2 (1 pt)
# (this should only be a single number, not a matrix!)
### YOUR CODE HERE ###
cor = np.corrcoef(my_data_1,my_data_2)[0][1]
cor


# In[139]:


# compute the covariance of my_data_1 and my_data_2 (1 pt)
# (this should also just be a single number)
### YOUR CODE HERE ###
cov = np.cov(my_data_1,my_data_2)[0][1]
cov


# In[140]:


# compute the 10th and 90th percentiles of my_data_1 (1 pt)
### YOUR CODE HERE ###
print(np.percentile(my_data_1,10))
print(np.percentile(my_data_1,90))


# ***
# # Problem 2.
# Suppose that we have run an experiment to see whether a new training strategy changes subjects' performance on a task. We first test 200 subjects on the task without training them, giving us the scores `d1`. Then we get 200 more subjects, train them using the new strategy, and then test how well they perform the task, giving the scores `d2`. We want to know whether the performance is different between these two groups (but we don't care which direction the difference lies).
# 
# In this problem you'll test whether these two datasets have significantly different means using a few different techniques.

# In[141]:


# Load the data
datafile = np.load('ps4_dataset_2.npz')
d1 = datafile['data1']
d2 = datafile['data2']

# Let's see the size of each array
print(d1.shape)
print(d2.shape)

# and print the mean of each
print('d1 mean:', d1.mean())
print('d2 mean:', d2.mean())


# ## (a) Use a t-test to see whether the means are significantly different
# Use the function `ttest_ind` to test whether the means of the two dataset are significantly different. Print the p-value. Note that this is a 2-sided test because it is testing whether the two are _different_ (i.e. either d1's mean is greater than d2, or vice versa).
# 
# Print the p-value.

# In[142]:


### YOUR CODE HERE ###
from scipy.stats import ttest_ind
ttest_ind(d1,d2, alternative='two-sided')


# ## (b) Use a bootstrap test to see whether the means are different
# Next, design and use a bootstrap test to tell whether the two datasets have significantly different means.

# In[143]:


# create new datasets d1_2 and d2_2
d1_2 = d1-np.mean(d1)+np.mean(np.hstack([d1,d2]))### YOUR CODE HERE ###
d2_2 = d2-np.mean(d2)+np.mean(np.hstack([d1,d2]) ### YOUR CODE HERE ###

# create array to hold bootstrap mean differences
nbootstraps = 10000
bs_mean_diffs = np.zeros(nbootstraps) ### YOUR CODE HERE ###

# take bootstrap samples many times
for ii in range(nbootstraps):
    # choose which indices will be used from d1_2 and d2_2
    inds1[ii] = d1_2.mean() ### YOUR CODE HERE ###
    inds2[ii] = d2_2.mean() ### YOUR CODE HERE ###
    
    # create your bootstrap samples
    bs_d1 = np.random.choice(d1_2, len(d1_2)) ### YOUR CODE HERE ###
    bs_d2 = np.random.choice(d2_2, len(d2_2)) ### YOUR CODE HERE
    
    # measure their difference and store it
    ### YOUR CODE HERE ###
    bs_mean_diff = np.mean(bs_d1) - np.mean(bs_d2)

# take the absolute value of each bootstrap difference, and find the fraction that are 
# larger than the absolute mean difference between d1 and d2. this is the bootstrap p-value
bs_pval = (np.abs(mean_diffs) > np.abs((np.mean(d1) - np.mean(d2)))).mean() ### YOUR CODE HERE ###

print('bootstrap p-value:', bs_pval)


# ## (c) Use a permutation test to see whether the means are different
# Next, design and use a permutation test to tell whether the two datasets have significantly different means.
# 
# The permutation p-value you get at the end should be about the same as what you got for the t-test and bootstrap test.

# In[144]:


# lump both datasets together
d_lump = np.hstack([d1, d2]) ### YOUR CODE HERE ###

# create array to hold permutation differences
npermutations = 10000
p_mean_diffs = np.zeros(npermutations) ### YOUR CODE HERE ###

# permute & compute many times
for ii in range(npermutations):
    # permute d_lump
    perm_d_lump = np.random.permutation(d_lump) ### YOUR CODE HERE ###
    
    # split it into two parts and find the difference of their means
    perm_d1 = perm_d_lump[:200] ### YOUR CODE HERE ###
    perm_d2 = perm_d_lump[200:] ### YOUR CODE HERE ###
    
    # compute the difference of their means and store it
    ### YOUR CODE HERE ###
    p_mean_diffs[ii] = np.mean(perm_d1) - np.mean(perm_d2)


# take the absolute value of each permutation difference, and find the fraction that are 
# larger than the absolute mean difference between d1 and d2. this is the permutation p-value
p_pval = ((np.abs(p_mean_diffs)) > (np.abs(np.mean(d1) - np.mean(d2)))).mean() ### YOUR CODE HERE ###

print('permutation p-value:', p_pval)


# ---
# # Problem 3.
# Now suppose that we've realized our initial experimental design was a nightmare. (400 subjects, who has the time!) So for the next experiment (and new training strategy) we've gone with a different design. This time, we test each of the 200 subjects (as before, yielding the dataset `e1`), _then_ train them using our new training strategy, then test each subject _again_, yielding the dataset `e2`.
# 
# This time the datasets are _paired_, meaning that `e1[0]` and `e2[0]` are from the same subject. We have to account for this in our analyses, because what we really care about is how much each subject _changed_.

# In[145]:


# Load the data
datafile = np.load('ps4_dataset_3.npz')
e1 = datafile['data1']
e2 = datafile['data2']

# Let's see the size of each array
print(e1.shape)
print(e2.shape)

# and print the mean of each
print('e1 mean:', e1.mean())
print('e2 mean:', e2.mean())


# ## (a) Use a t-test and paired t-test to compare means
# First, use a normal t-test (as before) to compare the means of `e1` and `e2`. Then, use a paired t-test (`ttest_rel`). Print the p-values for both.

# In[146]:


### YOUR CODE HERE ###
from scipy.stats import ttest_rel
print(ttest_ind(e1, e2, alternative = 'two-sided'))
print(ttest_rel(e1, e2, alternative = 'two-sided'))


# ## (b) Use a paired bootstrap test to compare the means
# Design and run a paired bootstrap test to compare the means of `e1` and `e2`. This should look almost identical to the simple bootstrap test above, but with one critical difference. Think hard about what the bootstrap samples represent, and how to make it paired. As before, use 10,000 bootstrap samples.
# 
# Print the bootstrap p-value at the end.

# In[147]:


### YOUR CODE HERE ###
e1_2 = e1 - np.mean(e1) + np.mean(np.hstack([e1,e2]))
e2_2 = e2 - np.mean(e2) + np.mean(np.hstack([e1,e2]))

nbootstrapse = 10000
bs_mean_diffs = np.zeros(nbootstrapse)

for ii in range(nbootstrapse):
    random = np.random.randint(0, len(e1), size = len(e1))
    bs_e1 =  e1_2[random]
    bs_e2 =  e2_2[random]
    bs_mean_diffs[ii] = np.mean(bs_e1) - np.mean(bs_e2)

    
bs_pval = (np.abs(bs_mean_diffs) > np.abs((np.mean(e1) - np.mean(e2)))).mean()
print('bootstrap p-value:', bs_pval)


# ## (c) Use a paired permutation test to compare the means
# Design and run a paired permutation test to compare the means of `e1` and `e2`. Unlike for the bootstrap, the paired permutation test is actually quite different from the normal permutation test. Remember that you need to keep the pairs of datapoints together. Think hard about what the null hypothesis (i.e. that the training had no effect) would mean when you do permutation. As before, do 10,000 permutations.
# 
# Print the permutation p-value at the end.

# In[148]:


### YOUR CODE HERE ###

npermutations = 10000
p_mean_diffs = np.zeros(npermutations)

for ii in range(npermutations):
    rand_perm = np.random.permutation(len(e1))
    ind1 = rand_perm[0:100]
    arr1 = np.array(e1)
    arr2 = np.array(e2)
    arr1[ind1] = e2[ind1]
    arr2[ind1] = e1[ind1]
    p_mean_diffs[ii] = np.mean(arr1) - np.mean(arr2)


p_pval = ((np.abs(p_mean_diffs)) > (np.abs(np.mean(e1) - np.mean(e2)))).mean()
print('permutation p-value:', p_pval)

