#!/usr/bin/env python
# coding: utf-8

# # Programming & Data Analysis in Modern Neuroscience: Problem Set 2

# In[1]:


# RUN THIS CELL TO IMPORT NUMPY
# YOU ONLY NEED TO DO THIS ONCE
# BUT IF YOU DO IT MORE THAN ONCE IT'S NOT A BIG DEAL DON'T WORRY
#Collaborators: Kevin Sheth, Jared Lim
import numpy as np


# # Problem 1.
# 
# Solve each indexing problem, and run your code to make sure it's outputting the right thing. Pay close attention to what the comment says the code should do.
# 
# _For example, if the code looks like this:_
# <pre>
# my_arr = np.arange(5) + 10*np.arange(5)[:,np.newaxis]
# # get the element from row 4, column 1, which should be equal to 41
# my_arr[#SOMETHING#]
# </pre>
# 
# _You should change it to:_
# 
# <pre>
# my_arr[4,1]
# </pre>

# In[2]:


# run this cell first to create my_arr
my_arr = np.arange(5) + 10*np.arange(5)[:,np.newaxis]
print(my_arr)


# In[3]:


# get the second-to-last column of my_arr (should be 3, 13, 23, 33, 43)
my_arr[:,3]


# In[4]:


# get rows 0 and 2 of my_arr, creating a 2x5 matrix ([0, 1, ...], [20, 21, ...])
my_arr[0:2,]


# In[5]:


# get every other column in my_arr, creating a 5x3 matrix ([0, 2, 4], [10, 12, 14], ...)
my_arr[0:,0:5:2]


# In[6]:


# compute the sum of each COLUMN of my_arr, should be (100, 105, 110, 115, 120)
my_arr.sum(axis=0)


# In[7]:


# compute the sum of each ROW of my_arr, should be (10, 60, 110, 160, 210)
my_arr.sum(axis=1)


# In[8]:


# find the largest element in each ROW of my_arr, should be (4, 14, 24, 34, 44)
my_arr.max(axis=1)


# In[9]:


# find the index of the smallest element by value in COLUMN 2 of my_arr, should be 0
my_arr[:,2].argmin(axis=0)


# In[10]:


# find out whether each column in my_arr contains the number 13, should be [False, False, False, True, False]
# (this one is a little tricky. you need to do two operations!)
(my_arr == 13).any(axis=0)


# In[11]:


# find the biggest number in my_arr that is smaller than 27, should be 24
my_arr[my_arr<27].max()


# In[12]:


# find the product of all the numbers in my_arr, should be zero
my_arr.prod()


# # Problem 2.

# Write a function called `checkerboard` that, given a number `N`, returns an N x N array with a checkerboard pattern. Use slicing to construct your checkerboard.
# 
# Eg. for `N = 3`, return 
# <pre>
# array([[0, 1, 0],
#        [1, 0, 1],
#        [0, 1, 0]])
# </pre>
# 
# This function needs to work for any value of `N`.

# In[2]:


def checkerboard(N):
    checkers = np.zeros((N,N))
    checker[1::2,::2] = 1
    checker[::2,1::2] = 1
    return checkers


# # Problem 3.
# Write a function that, given a 1D array `Z` and 2 numbers `x` and `y` (with `x <= y`), subtract 5 from every element of `Z` which is greater than `x` and less than `y`. For example, given `Z = array([4, 5, 9, 1, 5, 7])`, `x = 2`, and `y = 6`, the function should return `array([-1, 0, 9, 1, 0, 7])`. Do this _without_ modifying the original `Z`!

# In[14]:


def subtract_from_between(Z, x, y):
    new_data = Z.copy()
    new_data = np.where((new_data>x) & (new_data<y), -5+new_data,new_data)
    return new_data


# In[15]:


# to test:
my_Z = np.array([4,5,9,1,5,7])
new_Z = subtract_from_between(my_Z, 2, 6)
print(new_Z)
# should output: array([-1, 0, 9, 1, 0, 7])
print(my_Z)
# should output: array([4, 5, 9, 1, 5, 7])


# # Problem 4.
# Write a function that, given a 1D array, negates its minimum value. (You can assume that there is only one minimum value). For example, given `array([5, 6, 7, 8, 1, 5])`, return `array([5, 6, 7, 8, -1, 5])`. Do this _without_ modifying the input array.

# In[16]:


def negate_minimum(arr):
    new_arr = arr.copy()
    i = new_arr == new_arr.min()
    new_arr[i] = new_arr.min() * -1
    return new_arr


# In[17]:


# to test:
my_arr = np.array([5,6,7,8,1,5])
print(negate_minimum(my_arr))
# should output: array([5, 6, 7, 8, -1, 5])
print(my_arr)
# should output: array([5, 6, 7, 8, 1, 5])


# # Problem 5.
# Write a function that, given a 1D array, returns another array with 2 zeros between each of its values. For example, given `array([5, 6, 7, 8])`, return `array([5, 0, 0, 6, 0, 0, 7, 0, 0, 8])`. This function should work for any length of input array.

# In[18]:


def zero_fill(arr):
    zeros_list = np.zeros(len(arr)*3-2,dtype=int)
    zeros_list[::3]=arr
    return zeros_list


# In[19]:


# to test:
zero_fill(np.array([5, 6, 7, 8]))
# should output: array([5, 0, 0, 6, 0, 0, 7, 0, 0, 8])


# # Problem 6.
# 
# ## part (a)
# Write a function that constructs an array with `N` rows and `M` columns (i.e. an N by M matrix) where all the values in each row are equal to that row's index. For example, with `N=4` and `M=3`, you would want to create this matrix:
# <pre>
# 0 0 0
# 1 1 1
# 2 2 2
# 3 3 3
# </pre>
# (And you're not allowed to use the `np.meshgrid` function or `np.mgrid` object.)

# In[20]:


def row_index_matrix(N, M):
    NM_matrix = np.zeros((N,M), dtype=int)
    N = np.arange(N)[:,np.newaxis]
    NM = N + NM_matrix
    return NM


# In[21]:


# to test:
row_index_matrix(4,3)
# should output the matrix [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]


# ## part (b)
# Now write a function that constructs an array with `N` rows and `M` columns (i.e. an N by M matrix) where all the values in each _column_ are equal to that _column_'s index. For example, with `N=4` and `M=3`, you would want to create this matrix:
# <pre>
# 0 1 2
# 0 1 2
# 0 1 2
# 0 1 2
# </pre>

# In[22]:


def col_index_matrix(N, M):
    N = np.zeros((N,M), dtype=int)
    M = np.arange(M)[np.newaxis,:]
    NM = N + M
    return NM


# In[23]:


# to test:
col_index_matrix(4,3)
# should output the matrix [[0,1,2],[0,1,2],[0,1,2],[0,1,2]]


# # Problem 7.
# Suppose we have an image that we want to use as a stimulus in an experiment. This image is 15 x 15 pixels (what is this, a stimulus for ants? well maybe...). But, like a proper visual neuroscientist, you want to apply a circular mask to the image first. Let's suppose we want the mask to look like this (or maybe similar to this but with True and False instead of 1 and 0):
# 
# <pre>
# 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0
# 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
# 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
# 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
# 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
# 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
# 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
# 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
# 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
# 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
# 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
# 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
# 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0
# </pre>
# 
# One way to generate this type of circular mask is to compute the euclidean distance from each point in the image to the center. This distance matrix can then be thresholded to get the mask we see above.
# 
# Fortunately, you have all the tools you'll need to do this!
# 
# Using your `row_index_matrix` and `col_index_matrix` functions from problem 6, create two 15 x 15 matrices, one with row indices, and one with column indices.

# In[24]:


mask_rows = row_index_matrix(15,15)
mask_cols = col_index_matrix(15,15)


# The formula for euclidean distance is the same as the hypotenuse of a triangle. If we have two 2D points, `a=(a_0, a_1)` and `b=(b_0,b_1)`, then the euclidean distance between them is `np.sqrt((a_0 - b_0)**2 + (a_1 - b_1)**2)`. 
# 
# Think of the `mask_rows` and `mask_cols` matrices as containing the y-axis and x-axis locations of each point in a 15 x 15 grid. In order to create the mask you need to compute the euclidean distance between every point in your 15 x 15 grid and a point at the very center (row 7, column 7).
# 
# First compute the number of rows that separates each point from the center. This should be a new 15 x 15 matrix where each row has all the same values, and each column looks like (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7).

# In[25]:


row_separation = mask_rows - 7
print(row_separation)


# Then compute the number of columns that separates each point from the center. This should also be a 15 x 15 matrix, but where each column has all the same values, and each row looks like (-7, -6, ..., 6, 7).

# In[26]:


col_separation = mask_cols - 7
print(col_separation)


# Now you have `(a_0 - b_0)` (this is `row_separation`) and `(a_1 - b_1)` (`col_separation`) for each point in the grid (think of `a_0` and `a_1` as the row and column of each point, and `b_0 = b_1 = 7` is the location of the central point). Use these two matrices to compute the euclidean distance from each point to the center. This should give you a matrix that looks something like this (but probably with more decimal places):
# <pre>
# [[9.9 9.2 8.6 8.1 7.6 7.3 7.1 7.  7.1 7.3 7.6 8.1 8.6 9.2 9.9]
#  [9.2 8.5 7.8 7.2 6.7 6.3 6.1 6.  6.1 6.3 6.7 7.2 7.8 8.5 9.2]
#  [8.6 7.8 7.1 6.4 5.8 5.4 5.1 5.  5.1 5.4 5.8 6.4 7.1 7.8 8.6]
#  [8.1 7.2 6.4 5.7 5.  4.5 4.1 4.  4.1 4.5 5.  5.7 6.4 7.2 8.1]
#  [7.6 6.7 5.8 5.  4.2 3.6 3.2 3.  3.2 3.6 4.2 5.  5.8 6.7 7.6]
#  [7.3 6.3 5.4 4.5 3.6 2.8 2.2 2.  2.2 2.8 3.6 4.5 5.4 6.3 7.3]
#  [7.1 6.1 5.1 4.1 3.2 2.2 1.4 1.  1.4 2.2 3.2 4.1 5.1 6.1 7.1]
#  [7.  6.  5.  4.  3.  2.  1.  0.  1.  2.  3.  4.  5.  6.  7. ]
#  [7.1 6.1 5.1 4.1 3.2 2.2 1.4 1.  1.4 2.2 3.2 4.1 5.1 6.1 7.1]
#  [7.3 6.3 5.4 4.5 3.6 2.8 2.2 2.  2.2 2.8 3.6 4.5 5.4 6.3 7.3]
#  [7.6 6.7 5.8 5.  4.2 3.6 3.2 3.  3.2 3.6 4.2 5.  5.8 6.7 7.6]
#  [8.1 7.2 6.4 5.7 5.  4.5 4.1 4.  4.1 4.5 5.  5.7 6.4 7.2 8.1]
#  [8.6 7.8 7.1 6.4 5.8 5.4 5.1 5.  5.1 5.4 5.8 6.4 7.1 7.8 8.6]
#  [9.2 8.5 7.8 7.2 6.7 6.3 6.1 6.  6.1 6.3 6.7 7.2 7.8 8.5 9.2]
#  [9.9 9.2 8.6 8.1 7.6 7.3 7.1 7.  7.1 7.3 7.6 8.1 8.6 9.2 9.9]]
# </pre>

# In[27]:


euclidean_dist = np.sqrt((row_separation)**2 + (col_separation)**2)
print(euclidean_dist)


# Finally you are ready to create the mask! Find all the points that have distance less than 7.1 to create a mask. To validate, check that your mask looks like the one above (although it's ok if it's True and False instead of 1 and 0), and that it sums to 161.

# In[28]:


mask = euclidean_dist < 7.1

print(mask.sum()) # should print 161


# ## Bonus
# Write the solution to problem 7 in one line.

# In[29]:


mask = (np.sqrt((mask_rows-7)**2+(mask_cols-7)**2)<7.1)
print(mask.sum())


# In[31]:


print(((np.sqrt((mask_rows-7)**2+(mask_cols-7)**2)<7.1)).sum())

