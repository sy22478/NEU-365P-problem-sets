#!/usr/bin/env python
# coding: utf-8

# In[5]:


NAME = "Sonu Yadav"
COLLABORATORS = "Kevin Sheth, Raiyan Osman"


# # Programming and Data Analysis in Modern Neuroscience: Problem Set 3

# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
from matplotlib import pyplot as plt


# # Problem 1.
# Solve each plotting problem.

# In[79]:


# run this cell to create the dataset you'll be plotting
a = np.linspace(0, 4 * np.pi, 50)
b = np.cos(a)


# In[81]:


# create a line plot with a on the x-axis and b on the y-axis (1 pt)
### YOUR CODE HERE ###
plt.plot(a,b)


# In[83]:


# again plot a vs. b, but this time use small dots (.) connected by lines (1 pt)

### YOUR CODE HERE ###
plt.plot(a,b,'.', linestyle='solid')


# In[85]:


# plot a vs. b using red dots (and no connecting line) (1 pt)

### YOUR CODE HERE ###
plt.plot(a,b,'.', color='red')


# In[87]:


# use plt.bar to create a bar plot of the values in b, where the center of each bar is at a (1 pt)

### YOUR CODE HERE ###
plt.bar(a,b)


# In[89]:


# fix the bar plot by setting the width of each bar to 0.2 (1 pt)

### YOUR CODE HERE ###
plt.bar(a,b, width=0.2)


# In[93]:


c = np.sin(a[:,np.newaxis] * a)
print(c.shape)

# c is a 50 x 50 matrix
# use plt.matshow to make a raster plot of c (1 pt)
### YOUR CODE HERE ###
plt.matshow(c)
# and add a colorbar
### YOUR CODE HERE ###
plt.colorbar()


# In[95]:


# create a raster plot using the colormap plt.cm.RdBu (1 pt)
# include a colorbar!

### YOUR CODE HERE ###
plt.matshow(c, cmap='RdBu')
plt.colorbar()


# # Problem 2.
# For this problem you're going to be _analyzing some real neural data. Specifically, you're going to be analyzing data from an auditory electrophysiology experiment done in rodents in [this paper by Hamilton et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3841078/).
# 
# For this problem you'll be doing data munging using numpy, visualization, and some statistics. This is the whole data analysis stack!

# In[21]:


# load the neurophysiology data
data_file = np.load('ps3_neural_data.npz')
stimulus = data_file['stim'].astype(float)
response = data_file['resp'].astype(float)


# `stimulus` is a 23 x 104000 matrix that tells you which sound stimuli were presented and when. Each of the 23 rows corresponds to a different stimulus (the first one is a noise burst, the next 21 are tone pips at different frequencies, and the last one is another noise burst). The 104000 columns each correspond to one time bin. Each time bin is 5 milliseconds long.

# In[23]:


print(stimulus.shape)


# `response` is a 16 x 104000 matrix that tells you whether a spike was recorded from each recording site during each 5 millisecond bin. Neural responses were recorded simultaneously from 16 sites.

# In[25]:


print(response.shape)


# ## a.
# Use `plt.matshow` to show the first 500 timepoints (i.e. first 500 columns) of the `stimulus` matrix. Use the `aspect` parameter to control how tall or wide the plot should be (using an aspect of around 2 works well).

# In[27]:


### YOUR CODE HERE ###
plt.matshow(stimulus[:,0:500], aspect = 2)


# ## b.
# Next, you're going to compute the tuning curve for each recording site by finding the average spiking rate while each stimulus is being played. For responses from one recording site (one row of `response`) and one of the stimuli (one row of `stimulus`), find the response timepoints where that `stimulus` is equal to 1, then compute their mean. Do this for every response and every stimulus, putting the values into a 23 x 16 matrix called `tuning_curves`.

# In[29]:


tuning_curves = np.zeros((len(stimulus), len(response)))
### YOUR CODE HERE ###
for si in range(len(stimulus)):
    for ri in range(len(response)):
        x = stimulus[si] 
        tuning_curves[si,ri] = np.mean(response[ri,stimulus[si]==1])
tuning_curves.shape


# **Bonus:** compute `tuning_curves` in one expression using the dot product `np.dot`.

# In[31]:


### YOUR (BONUS) CODE HERE ###
print(np.dot(len(stimulus),len(response)))


# In[32]:


dot = np.dot(23,16)
print(dot)


# In[33]:


print(tuning_curves.shape)


# ## c.
# Use `plt.matshow` to make a raster plot of the tuning curves. Use `plt.colorbar` to show a colorbar as well.

# In[35]:


### YOUR CODE HERE ###
plt.matshow(tuning_curves)
plt.colorbar()


# ## d. (1 pt)
# Use `plt.plot` to plot the tuning curve for the last recording site (column 15 from `tuning_curves`). Trim off the responses to the first and last stimuli before plotting.

# In[37]:


### YOUR CODE HERE ###
plt.plot(tuning_curves[1:-1,15])


# ## e.
# Use `plt.subplot` to make a 4x4 grid of plots showing the tuning curve for each of the 16 recording sites (each column of `tuning_curves`). Read the documentation for `plt.subplot` (remember the question mark!) to figure out how to call it. Set the y-axis limits to the same (0 to 0.25) for all the subplots.

# In[39]:


for ii in range(16):
    plt.subplot(4,4,ii+1)
    ### YOUR CODE HERE ###
    plt.plot(tuning_curves[:,ii])
    plt.ylim([0,0.25])


# # Problem 3.
# 
# Next you're going to use this data to make a [peri-stimulus time histogram](https://en.wikipedia.org/wiki/Peristimulus_time_histogram) for each recording site.

# In[41]:


# first we need to find the actual stimulus start times. just run this bit
stim_start_freqs, stim_start_times = np.nonzero(np.diff(stimulus, axis=1) > 0)

# unique_start_times has the index for each time a sound started playing
unique_start_times = np.unique(stim_start_times) - 3
print(unique_start_times.shape)
print(unique_start_times[:10])


# ## a.
# Now you're going to compute the PSTH for one recording site (the last one).

# In[43]:


psth_before = 40
psth_after = 40

total_psth = np.zeros(80)
for s in unique_start_times:
    ### YOUR CODE HERE ###
    each = response[-1][s-psth_before:s+psth_after]
    total_psth+=each

total_psth /= len(unique_start_times)
print(total_psth)


# ## b.
# Next create a bar plot from `total_psth`.
# 
# Use `plt.vlines` to add a black vertical line at $t=0$. Use `plt.xlabel` to add a meaningful label to the x-axis, and use `plt.ylabel` to add a meaningful label to the y-axis. Use `plt.ylim` and `plt.xlim` to set nice-looking limits.

# In[45]:


### YOUR CODE HERE ###
x_location = np.arange(-40,40)*5
plt.bar(x_location, total_psth, width = 5, color='orchid')
plt.vlines(0,0,0.4, color='black')
plt.xlabel('Time relative to sound onset(ms)')
plt.ylabel('# Spikes')
plt.ylim(0, 0.4)
plt.xlim(-210,210)


# ## c.
# Now compute the PSTH for each of the 16 recording sites using the same technique as above. First create a matrix `all_psths` that is (number of response recording sites) x (length of PSTH). Then compute each PSTH, storing the results in that matrix.

# In[47]:


psth_before = 40
psth_after = 40

all_psths = np.zeros((16,80))
### YOUR CODE HERE ###
for s in unique_start_times:
    for r in range(len(response)):
        all_psths[r] += response[r,s-40:s+40]
    
# again, normalize by the number of unique stimulus start times
all_psths /= (len(unique_start_times))
print(all_psths)
all_psths.shape


# ## d.
# Create a 4x4 grid of PSTH plots using `plt.subplot`.

# In[49]:


plt.figure(figsize=(12,12))
### YOUR CODE HERE ###
grid = np.arange(-40,40)*5
plt.figure(figsize=(12,12))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.bar(grid,all_psths[i,:], width=5)
    if i in range(12,16):
        plt.xlabel('Times relative to sound onset(ms)')
    if i in [0,4,8,12]:
        plt.ylabel('# Spikes')


# # Problem 4.
# The last thing we're going to do with this dataset is use statistical tests to see which recording sites are responding signifincantly to each stimulus. These binary neural responses work nicely with a binomial model, so we'll use the binomial test. We will be using `binom_test` from `scipy.stats`. 
# 
# But before we can work with the neural data, we first need to address an important issue: [correction for multiple comparisons](https://en.wikipedia.org/wiki/Multiple_comparisons_problem). Here we're going to be using the [Bonferroni method](https://en.wikipedia.org/wiki/Bonferroni_correction), which is not the best method that's out there, but it's easy to understand and easy to implement.

# In[ ]:


from scipy.stats import binom_test


# ## a.
# First let's generate a fake dataset with 1000 different binary response timecoureses with 360 time points each, all with response rates exactly equal to 10%. Set `fake_q` to the response rate you want, then use `np.random.rand` to create a 360 x 1000 matrix, and then binarize it using your `fake_q`, storing the result in `fake_data`.

# In[ ]:


fake_q = 0.1## YOUR CODE HERE ###
fake_data = np.random.rand(360,1000) < fake_q### YOUR CODE HERE ###


# ## b.
# Next, find the total number of spikes for each of the 1000 response timecourses. It should be around 36 for most of them.

# In[ ]:


fake_totals = fake_data.sum(axis=0)### YOUR CODE HERE ###
print(fake_totals.shape)
print(fake_totals[:10])


# ## c.
# Do a binomial test on each one to get a p-value for the binomial test that the rate is different from 10% (either lower or higher). Use the function `binom_test` with proper settings for parameters `x`, `n`, and `p` (what we call $q$) for each of the 1000 response timecourses. (You should use a list comprehension surrounded by `np.array(...)`.)

# In[ ]:


fake_pvals = np.array([binom_test(x, len(fake_data),0.1) for x in fake_totals])### YOUR CODE HERE ###
print(fake_pvals[:10])


# ## d.
# Find all the "significant" tests according to traditional criteria (i.e. where the p-value is less than 0.05). `fake_signif` should be `True` for each responses where `fake_pvals` is smaller than 0.05, and `False` for each where it's greater.

# In[ ]:


fake_signif = fake_pvals < 0.05### YOUR CODE HERE ###

# how many are there?
print(fake_signif.sum())


# Many of these tests come out significant, even though we _know_ they shouldn't be. We tested whether the response rate for each of our 1000 fake experiments was "significantly" different from 10%, but they were all generated using a rate of 10%! What gives?
# 
# It comes down to the $p$-value and how we define "significance". Remember that the $p$-value is how often an experiment with _no actual effect_ would yield a result that is at least as extreme as what is observed. Since we defined a $p$-value threshold of 0.05 (standard in the field), we should expect that about 5% of experiments should be "significant" at the 0.05 level. What do we do about this?
# 
# ## Correcting for multiple statistical tests
# If you run many statistical tests you shouldn't use the same significance threshold that you would use on a single test. The simplest solution to this problem is [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction). With Bonferroni you change the significance threshold so that the probability of seeing _any_ significant result across all of your tests is less than some value. The probability of any test being significant by chance is called the "family-wise error rate" (FWER). 
# 
# ## e. 
# Bonferroni correction is accomplished by simply dividing your significance threshold by the number of tests. Do that here.

# In[ ]:


bonferroni_threshold = 0.05/1000### YOUR CODE HERE ###
print(bonferroni_threshold)


# ## f.
# The number of significant tests should probably be zero now. Running this whole analysis there should only be a 5% chance of having one or more significant results. Compute the significance of each test using your new threshold.

# In[ ]:


fake_signif_bonferroni = fake_pvals < bonferroni_threshold### YOUR CODE HERE ###
print(fake_signif_bonferroni.sum())


# # Problem 5.
# 
# Use a binomial test to determine whether each neural recording channel responded significantly to each of the stimuli.
# 
# ## a.
# Compute the background spiking rate for each of the 16 recording channels (i.e. the average response while there is no stimulus). Do this by first finding all the timepoints where none of the stimuli are on and storing in `no_stim` (here it can be nice to use the negation operator `~`, which flips `True` and `False`, e.g. `~my_arr`). Then select those `no_stim` timepoints from the responses, and find the mean response in each channel.

# In[ ]:


no_stim = stimulus == 0 ### YOUR CODE HERE ###
no_stim_avg_response = np.mean(response[:, np.all(no_stim, axis = 0)], axis = 1)### YOUR CODE HERE ###
print(no_stim_avg_response)


# ## b.
# Next, use binomial tests to compare the spiking rate of each recording site during each stimulus with the background rate. This will look similar to your tuning curve calculation from problem 2, but will also involve `binom_test`. For each stimulus and each recording site, you will need to find (1) the total number of timepoints when that stimulus was playing, and (2) the total number of spikes when that stimulus was playing, and then use `binom_test` to compare those values to the background spiking probability for that recording site (in `no_stim_avg_response`).

# In[ ]:


stim_response_pvals = np.zeros((len(stimulus), len(response)))

for si in range(len(stimulus)):
    for ri in range(len(response)):
        stim_response_pvals[si][ri] = binom_test(np.sum(response[ri, stimulus[si] == 1]), sum(stimulus[si]), no_stim_avg_response[ri])
        ### YOUR CODE HERE ###
print(stim_response_pvals)


# ## c.
# Create a raster plot showing which stimuli elicited significant responses from each of the recording sites. Here define significance as having a p-value less than 0.05 in the binomial test. Use `plt.matshow` on the resulting binary matrix.

# In[ ]:


### YOUR CODE HERE ###
plt.matshow(stim_response_pvals<0.05)


# ## d.
# From problem 4 you know that the significance threshold of p < 0.05 is problematic when we're doing lots of tests: some of them will probably come out positive just by chance! Let's use the Bonferroni method to adjust the significance threshold.
# 
# First compute the total number of tests that we're doing (the number of stimuli multiplied by the number of responses).

# In[ ]:


total_tests = len(stimulus)*len(response)### YOUR CODE HERE ###
print(total_tests)


# Then compute the Bonferroni-corrected significance threshold.

# In[ ]:


bonferroni_threshold = 0.05/total_tests### YOUR CODE HERE ###
print(bonferroni_threshold)


# Finally, make a raster plot (like part **(c)**) showing which tests are significant using this more stringent threshold.

# In[ ]:


plt.matshow(stim_response_pvals<bonferroni_threshold)### YOUR CODE HERE ###


# ## e.
# Finally let's put it all together in a nice-looking plot. Here, use `plt.matshow` to make a raster plot of the `tuning_curves` from problem 2. Then use `plt.spy` to add markers showing which responses are significant (you'll need to use the `marker` parameter, and probably `color` and `markersize`). Then add a colorbar and label your axes. Make it look nice!

# In[ ]:


### YOUR CODE HERE ###
plt.matshow(tuning_curves)
plt.spy(stim_response_pvals<bonferroni_threshold, marker='*', color='white', markersize=3)
plt.xlabel('Response')
plt.ylabel('Stimulus')
plt.colorbar()

