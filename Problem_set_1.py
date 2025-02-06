#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Sonu Yadav"
COLLABORATORS = "Kevin Seth"


# ---

# ## Problem 1
# Define a function to get the following outcomes.

# In[3]:


#Define a function to create a list of numbers and get 4th element from the list.
def get_element():
    my_list = list(range(42))
    
    # get the 4th element of this list

    my_list[3]
    return my_list[3]

get_element()


# In[5]:


#Define a function to create a list of numbers and get the length of the list.
def get_length():
    my_list = list(range(42))
    
    #find the length of this list
    length = len(my_list)
    return length

get_length()


# In[7]:


#Define a function to get the square of a number.
def print_square():
    my_list = list(range(42))
    # print the square of each number
    for n in my_list:
        print(n ** 2)
        
print_square()


# In[320]:


#Define a function to add +1 to each element in a list of numbers.
def print_plus_one():
    my_list = list(range(42))

    # print each number plus one
    for n in my_list:
        print(n + 1)

print_plus_one()


# In[9]:


#Define a function to add all the elements in a list of numbers.
def sum_nums():
    my_list = list(range(42))

    # sum up all the numbers in the list
    total = 0
    for n in my_list:
        total += n
    return total

sum_nums()


# In[11]:


#Define a function to get the last element from a list of numbers.
def last_elem():
    my_list = list(range(42))

    # get the last element of the list

    return my_list[-1]

last_elem()


# In[44]:


#Define a function to add two to each element in a list of numbers
def print_plus_one_2():
    my_list = list(range(42))

    # print each number plus two
    for n in my_list:
        print(n + 2)

print_plus_one_2()


# In[48]:


#Define a function to display one number higher than the number given.
def print_val_higher():
    # print the number that is 1 higher than val
    val = 7
    return (val) + 1
print_val_higher()


# In[17]:


#Define a function to get the value for the key "key" in the dictionary
def get_dict_val():
    my_dict = dict(key=1, other_key=2)

    # get the value for the key "key" in my_dict

    return my_dict['key']

get_dict_val()


# In[19]:


#Define a function to append 6 to a list of numbers
def append_val():
    my_list = list(range(5))

    # append 6 to the end of my_list
    my_list.append(6)
    return my_list
append_val()


# ## Problem 2.
# One important tool for text analysis is the word frequency counter, in which we store the number of times each word is used in a text document. 
# 
# You are given the name of the text file (`filename`) and a list of words (`words`). Write a function `word_frequency_counter` that outputs a list of numbers corresponding to the frequency of the words in order.
# For example, if `words` = ['building', 'rose'] and 'building' is referenced twice and 'rose' once, your function should return [2, 1] as the output.
# 
# Use your word frequency counter to count how many times the words "of" and "kid" appear in the document `text_file.txt`.
# 
# *Note:* you may only use functions and data structures that we've spoken about in class: `dict`, `list`, `for`-loops, etc. in solving this problem.

# In[21]:


def word_frequency_counter(filename, words):
    ### YOUR CODE HERE ##
    filename = open("text_file.txt", "r")
    line = filename.read()
    words = line.split()
    list_kid = []
    list_of = []
    for word in words:
        if word == "kid":
            list_of.append("kid")
    for word in words:
        if word == "of":
            list_kid.append("of")
    return len(list_kid), len(list_of)


# In[23]:


# test:
counts = word_frequency_counter("text_file.txt", ["of", "kid"])
print(counts)
# should produce: [44, 5]


# ## Problem 3.
# Now imagine that you are helping the professor with keeping record of the students in the class. We have 2 separate lists, one with the names of the students and the other having their respective UIDs. It would be more convenient to combine these two lists into a dictionary.
# 
# Write a function `student_records_dict` that takes in the `names` and `UIDs` and gives a `dict` of name-UID pairs. For example, if `names` = ['Adora', 'Bow'] and `UIDs` = ['as324345', 'bl345476'], then the function should return {'Adora' : 'as324345', 'Bow' : 'bl345476'}
# 
# **Bonus:** +1 bonus point if you can do this in one line of code.

# In[25]:


#You can assume names and UIDs to be of the same length
def student_records_dict(names, UIDs):
    ### YOUR CODE HERE ###
    return (dict(zip(names,UIDs)))


# In[27]:


# test:
test_names = ['adora', 'bow', 'catra']
test_uids = ['as239048', 'br109238', 'cm190238']
print(student_records_dict(test_names, test_uids))
# should produce: {'adora': 'as239048', 'bow': 'br109238', 'catra': 'cm190238'}


# ## Problem 4.
# 
# The [dot product](https://en.wikipedia.org/wiki/Dot_product) between two vectors (lists of numbers) is the sum of the products of each corresponding pair. E.g. the dot product of `[1,2,3]` and `[5, 10, 15]` is `1*5 + 2*10 + 3*15`.
# 
# Write a function `dot_product(a,b)` that computes the dot product between two lists of numbers `a` and `b`. You may only use functions and data structures that we've spoken about in class (e.g. `list` and `zip`). No numpy!
# 
# **Bonus:** +1 bonus point if your function checks that (1) `a` and `b` are the same size, and (2) checks that all the elements of `a` and `b` are numbers that can be multiplied, and `raise`s understandable errors if not.

# In[29]:


def dot_product(a,b):
    ### YOUR CODE HERE ###
    if len(a) != len(b):
        return 0
    return sum(i[0] * i[1] for i in zip(a, b))


# In[31]:


# test:
test_a = list(range(100))
test_b = test_a[::-1]
print(dot_product(test_a, test_b))
# should produce: 161700


# ## Problem 5.
# 
# [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) is an operation in linear algebra where two matrices are multiplied together to make a new matrix. If you are multiplying `A * B`, where `A` is $n$ by $m$ (i.e. has $n$ rows and $m$ columns) and `B` is $m$ by $p$, then the product `C = A * B` is an $n$ by $p$ matrix. Each element of `C` is the dot product between one row of `A` and one column of `B`. For example, `C[i][j]` (the element in row $i$ and column $j$) is the dot product between row $i$ of `A` and column $j$ of `B`.
# 
# Write a function that does matrix multiplication. Feel free to use your dot product function from problem 6 if that helps. You may only use things we've spoken about in class: `list`s, `zip`, etc. No numpy!

# In[33]:


def matrix_multiply(A, B):
    ### YOUR CODE HERE ###
        return [[sum(i * j for i, j in zip(r, c)) for c in zip(*B)]
        for r in A]


# In[35]:


# test:
test_A = [[1, 2, 3], [4, 5, 6]] # a 2x3 matrix
test_B = [[1, 2], [3, 4], [5, 6]] # a 3x2 matrix
print(matrix_multiply(test_A, test_B))
# should produce: [[22, 28], [49, 64]]


# ## Problem 6. (10 points)
# In lecture 3 we talked about how to load a set of video labels from a text file. Each line of the file contains the index of a video clip, the filename of the video gif, and then a list of labels. We loaded the labels into a `list` of `dict`s. Each element in the `list` is a `dict` that contains the information about one video clip (the `index`, the `gif_file`, and the list of `labels`).
# 
# Now, suppose that you are doing an analysis where you need to be able to find all the indices that correspond to each label. To do this, you need to re-arrange the data.
# 
# Write a function `find_indices_for_labels` that creates a `dict` where each key is a label and the value is the list of indices for clips that contained that label.
# 
# For example, the resulting `dict` should have the key `"gift.n.01"` with the value `[17,18,19,20]`.

# In[37]:


def parse_labels(filename):
    """Load labels from the file specified in filename.
    
    Assumes that each line of the file is whitespace-delimited, and has this
    structure:
      index gif_file label label label label ...
    with an arbitrary number of labels per line.

    Returns a list of dictionaries, where each contains the keys "index",
    "labels" (which returns a list of labels), and "gif_file".
    """
    f = open(filename)

    labels = [] # create an empty list
    for line in f:
        words = line.split() # split the line into words

        line_dict = dict(index=int(words[0]), # create a dict from words
                         gif_file=words[1],
                         labels=words[2:])
        
        labels.append(line_dict) # append dict to the complete list
    
    return labels

# load the labels
parsed_labels = parse_labels("labels.txt")
print(len(parsed_labels))


def find_indices_for_labels(labels):
    label_indices = dict()
    for line_dict in labels:
        for label in line_dict["labels"]:
            if label in label_indices:
                label_indices[label].append(line_dict["index"]) 
            else:
                label_indices[label] = []
                label_indices[label].append(line_dict["index"])
    return label_indices
print(find_indices_for_labels(parsed_labels))

print(find_indices_for_labels(parsed_labels)['gift.n.01'])


# ## Problem 7.
# Now that you've finished re-arranging the list of labels into a dictionary, you realize that what you _actually_ need is indicator variables that tell you whether each label is present or not in each video clip. For each label, create a list that has one element for every index (each list should be length 500). The value of each element should be `True` if that index corresponds to a video clip that contains that label, and `False` otherwise. The output should be a dictionary where the keys are labels and the values are lists containing the indicator variables.
# 
# For example, `indicator_dict['gift.n.01']` should look like `[False, False, False, ...]` with only four `True`'s.

# In[39]:


def make_label_indicators(label_indices):
    indicator_dict = dict()
    for label in label_indices:
        label_indicators = []
        for index in range(500):
            if index in label_indices[label]:
                label_indicators.append(True)
            else:
                label_indicators.append(False)
        indicator_dict[label] = label_indicators
    return indicator_dict

label_indices = find_indices_for_labels(parsed_labels)
print(make_label_indicators(label_indices)['gift.n.01'])
print(sum(make_label_indicators(label_indices)['gift.n.01']))

