#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[127]:


NAME = "Sonu Yadav"
COLLABORATORS = "Kevin Sheth, Jared Lim"


# # Programming & Data Analysis in Modern Neuroscience: Problem Set 5

# In[128]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import IPython.display as ipd


# ---
# # Problem 1.
# You're being given a timeseries (audio) dataset, and your goal in this problem is to analyze it to find what it actually contains.

# In[129]:


# Load the data
data = np.load('ps5_dataset_1.npz')['signal']

# The SAMPLING RATE for this data is 44100 Hz (or 44.1 kHz)
# this is typical for audio data
Fs = 44100


# In[130]:


# evaluate this cell to create an audio object. hit the play button to hear the raw data
# (this should not sound good)
ipd.Audio(data, rate=Fs)


# ## (a) Plot the power spectral density
# Plot the power spectral density of `data`. Set the sampling rate correctly in your call to the power spectral density function so that the x-axis has the correct range (from zero to the Nyquist frequency). (And remember that you can put a semicolon `;` at the end of a line in jupyter to make it not print the output from that line. This hides all the junk when you plot a power spectrum.)

# In[131]:


### YOUR CODE HERE ###
plt.psd(data, Fs=44100);


# ## (b) Plot the spectrogram
# Now plot the spectrogram of `data`. Set the sampling rate so that the y-axis is scaled correctly. Add labels for the x- and y-axes.

# In[132]:


### YOUR CODE HERE ###
plt.specgram(data, Fs=44100);
plt.xlabel("Time")
plt.ylabel("Frequency")


# ## (c) Filter the signal to remove the noise
# From looking at the PSD and spectrogram, you might conclude that `data` contains a lot of noise in the higher frequencies, but something that looks like signal in the low frequencies.
# 
# Use `signal.firwin` to create a low-pass filter, and then use `np.convolve` to apply it to `data`. You'll need to select the cutoff frequency and number of taps (the length of the filter), and also set the sampling rate in your call to `firwin`. Then, play the resulting audio and see if you can hear what it says.
# 
# Try a few different values for the cutoff frequency and the number of taps. Try large numbers of taps and see what happens.

# In[133]:


my_filter = signal.firwin(10000, cutoff = 2400, fs = 44100) ### YOUR CODE HERE ###

filtered_data = np.convolve(data, my_filter, 'same') ### YOUR CODE HERE ###

ipd.Audio(filtered_data, rate=44100)


# So what does the audio say? Write the answer down here:
# Congratulations! This is the right answer!

# ---
# # Problem 2.
# 
# In this problem we'll be plotting and analyzing some EEG data.

# In[135]:


# first, load the data
datafile = np.load("ps5_EEG_data.npz")

# what objects are inside this data file?
print(datafile.files)

# load the eeg_data
# this dataset is an EEG recording with 8 channels and 101638 timepoints
eeg_data = datafile["eeg_data"]
print("eeg_data shape:", eeg_data.shape)

# get the sampling rate (Fs) in Hz
eeg_Fs = datafile["Fs"]
print("sampling rate:", eeg_Fs)


# ## (a) Plot some of the EEG data timeseries (4 pts)
# 
# Make 4 plots of the EEG data timeseries:
# * One plot showing half a second of data (how many samples is this?)
# * One plot showing two seconds of data
# * One plot showing 10 seconds of data
# * One plot showing 100 seconds data
# 
# You can start with just plotting one channel for each, but your finished plot should show all 8 channels on the same axis.
# 
# For each plot you need to figure out how many samples to include. You know that the sampling rate (the variable `Fs` that we loaded from the datafile) is 128 Hz, or 128 samples per second.
# 
# Please label _at least_ the x-axis of each plot.
# 
# **Bonus:** Make the x-axis ticks show units of seconds instead of samples.

# In[136]:


# plot half a second
half_sec = 128/2
for i in range(len(eeg_data)):
    new_data = eeg_data[i, :]
    plt.plot(new_data);
    plt.xlim([0,half_sec])


# In[137]:


# plot 2 seconds
two_sec = 128*2
for i in range(len(eeg_data)):
    new_data2 = eeg_data[i, :]
    plt.plot(new_data2);
    plt.xlim([0,two_sec])


# In[138]:


# plot 10 seconds
ten_sec = 128*10
for i in range(len(eeg_data)):
    new_data3 = eeg_data[i, :]
    plt.plot(new_data3);
    plt.xlim([0,ten_sec])


# In[139]:


# plot 1000 seconds
thousand_sec = 128*100
for i in range(len(eeg_data)):
    new_data4 = eeg_data[i, :]
    plt.plot(new_data4);
    plt.xlim([0,thousand_sec])


# ## (b) Plot the power spectrum (psd) of one channel of the EEG data
# 
# Use the function `plt.psd` to plot the power spectrum of one EEG channel. Set the sampling rate `Fs` correctly so that you get the correct units of frequency.
# 
# Then plot the power spectra for all 8 EEG channels in the same axis.

# In[140]:


# plot one power spectrum
eeg_channel = eeg_data[0]
plt.psd(eeg_channel, Fs = 128);


# In[141]:


# plot the power spectra from each of the 8 channels on the same axis
for i in range(len(eeg_data)):
    plt.psd(eeg_data[i, :], Fs = 128);


# ## (c) Plot a spectrogram of the EEG data
# Use the `plt.specgram` function to plot a spectrogram of the first 60 seconds of the EEG data from one channel. You'll need to set the parameter `Fs` appropriately. Label the x- and y-axes appropriately (with units).
# 
# You can also try playing with the `NFFT` and `noverlap` parameters to `plt.specgram`. Some settings of these parameters are illegal and will make `specgram` error--specifically, `noverlap` needs to be smaller than `NFFT`.

# In[142]:


# plot a spectrogram
eeg_channel2 = eeg_data[1]
plt.specgram(eeg_channel2, Fs = 128);
plt.xlim([0,60])
plt.xlabel('time (sec)')
plt.ylabel('Frequency (Hz)')


# ## (d) Filter the EEG data to remove noise
# 
# The big spike at 60 Hz is _definitely_ noise. Let's filter the EEG signal to remove it.
# 
# The simplest thing to do would be to low-pass filter just below 60 Hz (since there probably isn't much interesting signal in the 60-64 Hz range anyway, and 64 Hz is the highest frequency we can see here -- Nyquist!!).
# 
# **First,** design a low-pass filter using `signal.firwin`. You should set the `cutoff` frequency to something like 55 Hz, and make sure to set the sampling rate `fs` so that `firwin` knows how to handle the cutoff frequency you give it. Look at the docs for `signal.firwin` and check out the demos and notes for lecture 27 to see a demo of how to use this function. You'll also need to choose the number of taps in the filter--remember that fewer taps means a "softer" filter, while more taps means a "sharper" filter. You can play with this parameter to get a result that looks good.
# 
# **Second,** plot your filter using `plt.plot` to see what it looks like. Label the x-axis, with units.
# 
# **Third,** use `signal.freqz` to get the frequency response of your filter, and plot it. Make sure to use `np.abs` to get the magnitude of the complex-valued numbers that `freqz` gives you. (Make sure to tell `freqz` the sampling rate (`fs`) of your signal so that what it returns will be in actual units of Hertz (Hz).)
# 
# **Fourth,** apply the filter to the EEG data (use channel 0) using `np.convolve`. Plot the first 10 seconds of the resulting filtered timeseries as well as the first 10 seconds of the original timeseries on the same axis. How do they compare?
# 
# **Fifth,** plot the power spectrum of the filtered EEG signal. Make sure the units are correct and labeled.

# In[143]:


# design a low-pass filter
low_pass_filter = signal.firwin(101, cutoff=55, fs=128)


# In[144]:


# plot the filter
plt.plot(low_pass_filter)
plt.xlabel('number of taps')


# In[145]:


# plot the frequency response of the filter
x, y = signal.freqz(low_pass_filter, fs=128)
a = np.abs(y)
plt.plot(x, a);


# In[146]:


# filter the signal from one EEG channel
eeg_one = eeg_data[0]
job = np.convolve(eeg_one, low_pass_filter, 'same');


# In[147]:


# plot filtered & original data in same axis to compare
ten_sec = 128*10
eeg1 = eeg_data[0, :]
plt.plot(eeg1, label = 'original')
plt.plot(job, label = 'filtered data')
plt.legend()
plt.xlim([0,ten_sec])
plt.xlabel('samples')


# In[148]:


# plot power spectrum of the filtered EEG data
plt.psd(job, Fs = 128);


# # Problem 3.
# 
# Solve these small regression problems.

# In[149]:


# Run this first to set up variables
b_true = np.array([1, 0.25, 0.5, 1.25, 0])
X = np.random.randn(100, 5)
y = X.dot(b_true) + 0.25*np.random.randn(100)


# In[150]:


# use ordinary least squares regression (np.linalg.lstsq)
# to fit a regression model that predicts y from X
# store the weights, residuals, rank, and singular values in
# separate variables (2 pts)
b, residuals, rank, singular = np.linalg.lstsq(X, y)


# In[151]:


# compute (in-set) R^2 for the fit model using X and y (2 pts)
# (it should be close to 1.0)
residual_sum = ((X @ b - y) ** 2).sum()
total_sum = ((y - y.mean()) ** 2).sum()
R_2 = 1 - (residual_sum/total_sum)
print(R_2)


# In[152]:


# create a scatter plot showing the true weights (b_true)
# on the x-axis and the estimated weights on the y-axis
# use the format string 'o'
# label the x- and y-axes (2 pts)
plt.plot(b_true, b, 'o')
plt.xlabel('true weights')
plt.ylabel('estimated weights')


# # Problem 4.
# 
# Here you'll be using regression to fit models to fMRI data! In the next few problem, you will learn about how to use linear regression to fit _filters_ that you can use to model timeseries. You'll be using linear regression techniques to model fMRI responses to video stimuli (the ones that we've talked about in class) based on the presence of two categories in the videos: people and buildings.
# 
# This problem will just involve loading and doing basic visualizations of the data.

# In[153]:


# Load the fMRI dataset
datafile = np.load('ps5_fmri_data.npz')

# list all the variables in the file
print(datafile.files)


# In[154]:


# Load all the data
# (you don't need to worry about subtracting the mean 
# later because that's done here)

# the features (X) say whether people (column zero) or buildings 
# (column one) are present in each video clip
X_trn = datafile['X_trn'] # time x features
X_trn -= X_trn.mean(0) # subtract the mean over time
X_test = datafile['X_test'] # time x features
X_test -= X_test.mean(0) # ditto

# y_trn and y_test have the fMRI response of one voxel
y_trn = datafile['y_trn'] # time
y_trn -= y_trn.mean(0)
y_test = datafile['y_test'] # time
y_test -= y_test.mean(0)

# ybig_trn and ybig_test have the response of 10,000 voxels
ybig_trn = datafile['ybig_trn']
ybig_trn -= ybig_trn.mean(0)
ybig_test = datafile['ybig_test']
ybig_test -= ybig_test.mean(0)


# ## (a) Print the shape of each array

# In[155]:


# print the shapes of X_trn, X_test, y_trn, y_test, ybig_trn, ybig_test
print(np.shape(X_trn))
print(np.shape(X_test))
print(np.shape(y_trn))
print(np.shape(y_test))
print(np.shape(ybig_trn))
print(np.shape(ybig_test))


# ## (b) Simple plots
# Step 1 in any analysis should _always_ be to **LOOK AT YOUR DATA**. Let's plot it and see what it looks like: what's the range, does it look uniform, etc. Plot the regression target (`y_trn`) and both regression features (`X_trn[:,0]` and `X_trn[:,1]`) on the same axis. Label the x-axis. Assign labels to each line (using the `label=` keyword in `plt.plot`) and then add a legend using `plt.legend()` so we know which line is which.
# 
# Then plot a histogram of the values in `y_trn`.

# In[156]:


# plot responses & features over time (the first 100 timepoints)
plt.plot(X_trn[:,0], label = 'X_trn1')
plt.plot(X_trn[:,1], label = 'X_trn2')
plt.plot(y_trn, label = 'fmri')
plt.xlim(0,100)
plt.xlabel('time')
plt.legend()
# histogram y_trn
plt.figure()
plt.hist(y_trn);
plt.xlabel('fmri')


# ## (c) Simple correlation
# Compute and print the correlation between `y_trn` and `X_trn[:,0]`, and the correlation between `y_trn` and `X_trn[:,1]`.

# In[157]:


# compute & print correlations
corr1 = np.corrcoef(X_trn[:,0], y_trn)
corr2 = np.corrcoef(X_trn[:,1], y_trn)
print(corr1[0,1])
print(corr2[0,1])


# # Problem 5.
# 
# One of the key things you should know about fMRI is that it measures the Blood-Oxygen-Level Dependent (BOLD) signal, which tells you about blood flow in each area of the brain. This signal doesn't directly track neural activity! After there is a burst of neural activity in an area, nearby blood vessels slowly respond and recruit more blood over the next 2-8 seconds. So we can't directly model the fMRI response with our stimuli! We need to account for the slow [hemodynamic response](https://en.wikipedia.org/wiki/Haemodynamic_response).
# 
# We can think of the hemodynamic response as (to first approximation) a _filter_ on the underlying neural activity. So if the stimulus features (here, the presence of people or buildings in a video) are correlated with the neural activity, we can try to find a filter that we can apply to the stimulus features in order to make them look like the BOLD response.
# 
# We'll do this by creating new features that are _lagged_ versions of the stimulus features. If we use lags (0,1,2,3,4,5) then it's like we're modeling the response $y_t$ as a function of the stimulus features $x_t, x_{t-1}, x_{t-2}, x_{t-3}, x_{t-4}, x_{t-5}$. This is just like fitting a filter that is (effectively) convolved with the stimulus feature to get the response!
# 
# This approach is called a **Finite Impulse Response (FIR) model**.
# 
# ## (a) Create FIR regressors

# In[158]:


# let's start with just one feature, feature zero
# create a matrix that will hold 6 lagged versions of the feature vector
# this matrix should have T rows (same as X_trn) and 6 columns
X0_trn_lagged = np.zeros((len(X_trn), 6))


# In[159]:


# now we'll copy the feature X_trn[:,0] into each of the 6 columns of your new matrix
# we want the first column to be exactly X_trn[:,0] (because it's lag=0),
# the second column to be lagged by 1 timepoint relative to the original,
# the third to be lagged lagged 2, etc.

# The for loop is already set up so that:
# `ii` is the index of the column in `X0_trn_lagged` you want to copy into
# `lag` is how much this column should be lagged

# you don't need to change anything, just run it!

for ii,lag in enumerate([0,1,2,3,4,5]):
    # `frm` should be the timepoint that you start copying from in X_trn
    frm = 0
    # and `to` is the last timepoint
    to = len(X_trn) - lag
    X0_trn_lagged[lag:,ii] = X_trn[frm:to,0]


# In[160]:


# (2 pts)
# plot the first 100 timepoints for each of these 6 feature vectors
# on the same axis
# they should look identical, but shifted in time
vec = X0_trn_lagged[0:100]
plt.plot(vec);
plt.xlabel('time')


# In[161]:


# (2 pts)
# now use the same logic to create X0_test_lagged from X_test
data1 = np.zeros((len(X_test), 6))
for ii,test in enumerate([0,1,2,3,4,5]):
    frm = 0
    to = len(X_test) - test
    data1[test:,ii] = X_test[frm:to,0]
vec = data1[0:100]
plt.plot(vec);


# ## (b) Fit the FIR model
# Next, use `np.linalg.lstsq` to fit a linear regression model that predicts `y_trn` from `X0_trn_lagged`. Save the `wt, rank, res, sing` as always.

# In[162]:


# (2 pts)
# do regression here
wt, rank, res, sing = np.linalg.lstsq(X0_trn_lagged, y_trn)


# In[163]:


# (2 pts)
# now plot the weights as dots connected by lines
# make the x-axis values equal to the lags (0..5)
plt.plot(wt, '.-.')
plt.xlabel('lags')
plt.ylabel('weight value')


# In[164]:


# (2 pts)
# write, in 1 sentence:
# * What do these weights mean?
#The weights help us predict how much the model with our 
#datapoints are closer to the actual model with actual data. 
#It gives us the lowest squared error for each data point.
# * Why do they look like that?
#They look like that because there is a delay in response from initial spiking activity due to lag.


# ## (c) Test the FIR model
# Use the dot product between `wt` and `X0_test_lagged` to predict responses in the test dataset, then compute the correlation between predicted and actual responses (`y_test`).

# In[165]:


# (2 pts)
# create predicted test timecourse
y_test_pred = np.dot(data1, wt) ## SOMETHING ##


# In[166]:


# (1 pt)
# compute correlation
model1_corr = np.corrcoef(y_test_pred, y_test)[0,1] ## SOMETHING ##
print(model1_corr)


# ## (d) Repeat procedure in big dataset
# The model above was fit to just one voxel. Let's repeat this process for each of the 10,000 voxels in the bigger dataset. Ideally we would do this by reshaping the `ybig_*` datasets into a more sensible shape, but it might be easier to do using a for loop.
# 
# **IMPORTANT NOTE:** If you run into an error doing this regression (e.g. `SVD did not converge`), it's likely because `ybig_trn` contains weird non-numeric values called `NaN`s, and you need to remove these. To do this, you need to apply the function `np.nan_to_num` to your `ybig_trn`. That should look like this: `np.nan_to_num(ybig_trn[ ... ])` where the `...` is some kind of indexing expression that you need to use to pull out the responses for voxel (ii,jj).

# In[167]:


# (4 pts)
wt_big = np.zeros((6, 100, 100)) # this matrix will hold the weights
for ii in range(100):
    for jj in range(100):
        ytrn_data = np.nan_to_num(ybig_trn[:, ii, jj])
        wt, rank, res, sing = np.linalg.lstsq(X0_trn_lagged, ytrn_data)
        ## DO THE REGRESSION FOR VOXEL [ii,jj] ##
        wt_big[:,ii,jj] = wt ## THE WEIGHTS FOR THAT REGRESSION ##


# In[168]:


# (2 pts)
# compute predictions & find the prediction correlation 
# for every voxel in the big set
corr_big = np.zeros((100,100))
for ii in range(100):
    for jj in range(100):
        dot_product = np.dot(data1, wt_big[:, ii, jj])
        corr_big[ii,jj] = np.corrcoef(dot_product, ybig_test[:, ii, jj])[0,1] ## SOMETHING ##


# In[169]:


# (2 pts)
# plot the matrix `corr_big` using plt.matshow
# set vmin=0, and vmax=0.6
# this shows correlation across one axial slice through a subject's brain!
plt.matshow(corr_big, vmin=0, vmax=0.6);
plt.colorbar()


# In[170]:


# (4 pts)
# let's make the plot look nicer
# first get something resembling brain anatomy by using ybig_test.std(0) 
# (this just shows where "high signal" voxels are, which are all in cortex)
# plot that using plt.matshow with a grayscale colormap
plt.matshow(ybig_test.std(0), cmap='gray')
plt.colorbar()

# next, create a "thresholded" version of your correlation map
# to do this, first create a copy (remember .copy()?)
# then set all the values in the copy that are below 0.3 to np.nan
# finally, use plt.imshow to plot the thresholded correlations on top of the "anatomy"
copied = corr_big.copy()
copied[copied <= 0.3] = np.nan
plt.imshow(copied)

