#!/usr/bin/env python
# coding: utf-8

# # Final Project

# In[2]:


# Run this cell first to load the stuff you'll need

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import signal


# # Problem 1
# 
# Read the description of what each piece of code should do, and fix the code so that it does what the description says. Many of these changes will be quite small—you don't need to write lots of code! In most cases the description tells you what the output should be (or what shape it should be). Verify that your change does the right thing!

# In[3]:


# RUN THIS CELL FIRST TO CREATE DATA
a = np.random.randn(10)
b = np.random.randn(10)
c = np.random.randn(3,5)
d = np.array([True, True, False, True])


# In[4]:


# compute the correlation between a and b (should give a 2x2 matrix) (1 pt)
np.corrcoef(a, b)


# In[5]:


# compute the mean of each column of c (should give a length-3 array) (1 pt)
np.mean(c,axis=0)


# In[6]:


# concatenate arrays a and b into a single length-20 vector (1 pt)
np.concatenate([a,b])


# In[43]:


# negate each element in d (should give [False False True False]) (1 pt)
d = np.array([True, True, False, True])
[not elem for elem in d]


# In[9]:


# create a length-10 array with all zeros (1 pt)
np.zeros(10)


# In[10]:


# get the last 3 elements of a (1 pt)
# (you must use a negative index to get credit!)
a[-3:]


# In[11]:


# square each element in a using a list comprehension (1 pt)
# (please be sure to not re-use the variable names a, b, c, or d)
print([x**2 for x in a])


# In[12]:


# square each element in a using numpy (1 pt)
np.square(a)


# In[13]:


# reverse the order of b (1 pt)
b[::-1]


# In[44]:


# reshape array b to a 2x5 matrix (1 pt)
np.reshape(b, (2,5))


# # Problem 2
# 
# Use numpy array operations to solve each problem. You may not use a for loop!

# In[45]:


# RUN THIS FIRST TO CREATE VARIABLES

vec1 = np.array([-2, 14, 12, 212, 5280])
mat1 = np.arange(30).reshape(5,6) - 14.5


# In[46]:


# extract all elements of vec1 that are greater than 100 (1 pt)
# (this should give the array [212, 5280])
vec1[vec1 > 100]


# In[47]:


# find the minimum element of vec1 (1 pt)
np.min(vec1)


# In[48]:


# find the INDEX of the smallest element of vec1 (1 pt)
np.argmin(vec1)


# In[49]:


# find the biggest element in each row of mat1 (1 pt)
np.max(mat1, axis=1)


# In[50]:


# get the last column of mat1 using a negative index (1 pt)
mat1[:, -1]


# In[51]:


# get the second-to-last row of mat1 using a negative index (1 pt)
mat1[-2,:]


# # Problem 3
# Use matplotlib to make these plots

# In[52]:


# RUN THIS FIRST TO CREATE VARIABLES
aa = np.sin(np.linspace(0, 1e9, 100))
bb = 0.5 * aa + 0.5 * np.sin(np.linspace(0, 1e7, 100))

saa = np.sort(aa)
sbb = np.sort(bb)

cc = np.sin(np.linspace(0, 1e5, 100)).reshape(10,10)


# In[56]:


plt.plot(aa,bb,'ro', label='red data points')
plt.xlabel('aa')
plt.ylabel('bb')
plt.legend(loc='lower right')


# In[68]:


plt.figure(figsize=(5.05,5.05))
plt.plot(saa,sbb,'--', )
plt.xlabel('saa')
plt.ylabel('sbb')
plt.grid()


# In[77]:


fig = plt.figure(figsize=(5,4))
ax = plt.axes()
plt.imshow(cc,cmap='gray')
cbar=plt.colorbar()
ax.set_title('The awesome title')


# In[84]:


fig = plt.figure(figsize=(5,4))
plt.matshow(cc,vmin=0)
plt.colorbar()
plt.xlabel('Neato x-label')


# # Problem 4
# 
# Solve each statistics problem.

# In[22]:


# compute the median of the_sample (1 pt)
the_sample = np.array([0, 14, 2, 3.6, 6.6])

sample_median = np.median(the_sample)
print(sample_median)


# In[23]:


# compute the mean of each column of big_sample (1 pt)
big_sample = np.array([[1, 4, 7], [2, 3, 9], [9, 3, -5], [1, -20, 44]])

col_means = np.mean(big_sample, axis=0)
print(col_means)


# In[24]:


# compute the standard deviation of each column of big_sample (1 pt)
col_stds = np.std(big_sample, axis=0)
print(col_stds)


# In[25]:


# z-score the_sample by subtracting its mean and dividing it by its standard deviation (1 pt)
# (result should still be a length-5 array)
the_zscored_sample = (the_sample - np.mean(the_sample)) / np.std(the_sample)
print(the_zscored_sample)


# In[86]:


# you flip a coin 200 times and it came up heads 117 times
# use the binomial test to test whether this coin is biased (i.e. whether Prob(heads)=0.5)
# at a p<0.05 level (2 pts)
from scipy.stats import binom_test
n = 200
p = 0.5
k = 117
bin_p_value = stats.binom_test(k, n, p)
print(bin_p_value)


# In[31]:


# use a (independent) t-test to compare one_sample with another_sample
# are their means significantly different at a p<0.05 level? (2 pts)
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
one_sample = np.array([-0.44, 0.16, 0.63, -0.46, 1.31])
another_sample = np.array([1.2, 1.37, 0.74, 1.58, 3.05])

t_value, p_value = ttest_ind(one_sample, another_sample)
print(p_value)
print("Yes, the means of the samples are significantly different at a p<0.05 level.")


# In[32]:


# use a paired t-test to compare before_sample with after_sample
# print the p-value
# are their means significantly different at a p<0.05 level? (2 pts)
before_sample = np.array([2.17, -10.55, 1.79, -1.31, -2.71])
after_sample = np.array([0.05, -11.16, -0.62, -2.91, -2.86])


t_value, p_value = ttest_rel(before_sample, after_sample)
print(p_value)
print("Yes, the means of the samples are significantly different at a p<0.05 level.")


# # Problem 5
# 
# Use bootstrapping to estimate the standard error for each weight in a linear regression.

# In[33]:


# Run this cell to set up variables for the problem
from sklearn.metrics import r2_score
from sklearn import datasets
diabetes = datasets.load_diabetes()

# these are the actual names of the 10 features, which are more descriptive than the ones given by sklearn
features = "age sex bmi map tc ldl hdl tch ltg glu".split()

diabetes_X = diabetes.data
diabetes_y = diabetes.target - diabetes.target.mean()

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Make sure these shapes are correct!
print("diabetes_X_train.shape:", diabetes_X_train.shape)
print("diabetes_X_test.shape:", diabetes_X_test.shape)

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Make sure these shapes are correct too!
print("diabetes_y_train.shape:", diabetes_y_train.shape)
print("diabetes_y_test.shape:", diabetes_y_test.shape)


# In[34]:


# Run this cell to do the linear regression

# first we fit a model with all the training data to get baseline weights
wt, res, rank, sing = np.linalg.lstsq(diabetes_X_train, diabetes_y_train)

# predict the test set
pred = diabetes_X_test.dot(wt)

# and find R^2
R2 = r2_score(diabetes_y_test, pred)
print(R2)


# ## (a) Use bootstrap to find confidence intervals for weights and $R^2$
# Next we're going to find 95% confidence intervals for (1) each of the weights and (2) the model goodness of fit ($R^2$). We'll do this by bootstrapping the training dataset 10,000 times and re-fitting the model each time. For each bootstrap sample, we'll choose randomly among the 422 training datapoints (with replacement, so each datapoint can show up more than once).
# 
# Note that you don't need to resample the test set! Just use it exactly as above on each iteration of the bootstrap.

# In[107]:


nboots = 10000

# create matrices to hold the bootstrapped weights and R^2
bootstrap_wt = np.zeros((nboots, 10))
bootstrap_R2 = np.zeros(nboots)

# bootstrap the training dataset 10,000 times
# each time you need to:
# (1) select 422 random indices from 0..421 (with replacement)
# (2) use the indices to select rows from `diabetes_X_train` (and `diabetes_y_train`)
# (3) use the newly sampled X and y to fit model weights
# (4) save the new weights into the appropriate row of `bootstrap_wt`
# (5) use the new weights to make predictions
# (6) compute R^2 using the new predictions
# (7) save the new R^2 in the appropriate location in `bootstrap_R2`

for ii in range(nboots):
    inds = np.random.choice(diabetes_X_train.shape[0], size=diabetes_X_train.shape[0], replace=True)
    sampled_X = diabetes_X_train[inds, :]
    sampled_y = diabetes_y_train[inds]
    bwt, res, rank, sing = np.linalg.lstsq(sampled_X, sampled_y)
    bootstrap_wt[ii] = bwt
    pred = diabetes_X_test.dot(bwt)
    bR2 = r2_score(diabetes_y_test, pred)
    bootstrap_R2[ii] = bR2


# ## (b) Plot the results

# In[108]:


# plot a histogram of `bootstrap_R2` with 20 bins (1 pt)
# set an appropriate x-axis label
plt.hist(bootstrap_R2, bins=20)
plt.xlabel("R^2 value")


# In[110]:


# find and print the 95% confidence interval for R^2 using the bootstrap samples (1 pt)
R2_high = np.percentile(bootstrap_R2, 95)


# In[118]:


# plot the weights with errorbars indicating 95% confidence intervals (3 pts)
# use the function plt.bar to create a bar plot of the "true" weights
# (i.e. the weights you got from regression on the whole dataset)
# then use plt.errorbar to add errorbars (make sure to turn off lines connecting the points, etc.)
# add labels on the x-axis using plt.xticks
# give it xlocs for the `locs` parameter, and the feature names (`features`) for the labels
plt.figure()
plt.bar(np.arange(10),wt)
plt.errorbar(np.arange(10),wt,yerr=[R2_high-wt],fmt='None')
plt.xticks()


# ### Bonus
# Why is the procedure used here probably a bad way to compute a confidence interval for $R^2$?

# # why?
# The procedure used here to compute a confidence interval for 𝑅2 is probably a bad way to do it because 
# - the bootstrap procedure is used to estimate the confidence intervals for the model weights, but not for the 𝑅2 statistic itself. 
# - This means that the confidence intervals for the model weights are correct, but the confidence interval for 𝑅2 is not correct.
# - It also uses the entire training set to fit the model weights on each bootstrap sample, which means that the estimated confidence intervals for the model weights are likely to be too narrow, since the bootstrap procedure artificially increases the amount of data available to fit the model weights. 
# - This can lead to overconfidence in the estimated model weights.

# # Problem 6
# 
# You have a signal called `the_signal`, which is sampled at 10 kHz. High-pass filter `the_signal` at 2 kHz. First, you'll need to design a filter with a 2 kHz cutoff frequency using `signal.firwin`. Second, you'll use that filter to _low_-pass filter your signal, obtaining `the_lowpass_signal`. Finally, you'll subtract `the_lowpass_signal` from `the_signal` to get `the_highpass_signal`.

# In[35]:


# first let's create the signal! it's random!
the_signal = np.random.randn(25000)


# In[10]:


# plot the spectrum of `the_signal` using plt.psd (1 pt)
# make sure you set Fs correctly!

# plot the spectrum of `the_signal` using plt.psd
Fs = 10000  # set the correct sampling frequency
plt.psd(the_signal, Fs=Fs)
plt.show()


# In[99]:


# create a 2 kHz low-pass filter using signal.firwin (1 pt)
# let's use 151 taps
Fs = 10000
num_taps = 151
the_filter = signal.firwin(num_taps, cutoff=2000, fs=10000)


# In[101]:


# low-pass filter `the_signal` with `the_filter` using np.convolve (2 pt)
# to create `lowpass_signal`
# make sure you set mode="same" in your call to np.convolve!
lowpass_signal = np.convolve(the_signal, the_filter, mode="same")

# then use plt.psd to plot the spectrum (set Fs correctly!)
plt.psd(lowpass_signal, Fs=10000)
plt.show()


# In[106]:


# subtract `lowpass_signal` from `the_signal` to get `highpass_signal` (3 pts)
highpass_signal = the_signal-lowpass_signal
Fs=10000
# then plot the psd of `highpass_signal` (set Fs correctly!)
plt.psd(highpass_signal, Fs=Fs)


# and on a separate figure, plot the first 50 timepoints of:
# (1) the_signal (as a black line)
# (2) lowpass_signal (as a blue line)
# (3) highpass_signal (as a red line)
plt.figure()
plt.plot(the_signal[:50], color="black")
plt.plot(lowpass_signal[:50], color="blue")
plt.plot(highpass_signal[:50], color="red")
plt.show()

