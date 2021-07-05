#!/usr/bin/env python
# coding: utf-8

# (MCP-lab)=
# # MCP Neuron 

# ## Definition

# We need the basic buiding block of the ANN: the (artificial) neuron. The first mathematical model dates back to Warren McCulloch and Walter Pitts (MCP), who proposed it in 1942, hence at the very beginning of the electronic computer age during World War II. The MCP neuron depicted in {numref}`MCP1-fig` in the basic ingredient of all ANNs and is built on very simple general rules, inspired neatly by the biological neuron:
# 
# - The signal enters the nucleus via dendrites from the previous layer.
# - The synaptic connection for each dendrite may have a different (and adjustable) strength.
# - In the nucleus, the signal from all the dendrites is combined (summed up) into $s$.
# - If the combined signal is stronger than a given threshold, then the neuron fires along the axon, in the opposice case it remains still. 
# - In the siplest relization, the strenth of the fired signal has two possible levels: on or off, i.e. 1 or 0. No intermediate values are needed.
# - Axon terminal connect to dendrites of the neurons in the next layer. 

# :::{figure-md} MCP1-fig
# <img src="images/mcp-1a.png" width="320px">
# 
# MCP neuron: $x_i$ are the inputs (different in each instance of the data), $w_i$ are the weights, $s$ is the signal, $b$ is the bias, and $f(s;b)$ represents the acitvation function, yielding the output $y=f(s;b)$. The blue oval encircles the whole neuron, as used in {numref}`ffnn-fig`.
# :::

# Translating this into a mathematical prescription, one assigns to the input cells the numbers $x_1, x_2 \dots, x_n$ (input data). The strength of the synaptic connections is controled with the **weights** $w_i$. Then the combined signal is defined as the weighted sum  
# 
# $$s=\sum_{i=1}^n x_i w_i.$$
# 
# Thesignal becomes an argument of the **activation function**, which to begin takes the simple form of the step function
# 
# $$
# f(s;b) = \left \{ \begin{array}{l} 1 {\rm ~for~} s \ge b \\ 0 {\rm ~for~} s < b \end{array} \right .
# $$
# 
# When the combined signal $s$ is larger than the bias (threshold) $b$, the nucleus fires. i.e. the signal passed along the axon is 1. in the opposite case, the generated signal value is 0 (no firing). This is precisely what we need to mimick the biological prototype.

# There is a convenient notational covnention which is frequently used. Instead of splitting the bias from the input data, we may treat it uniformly. The condition for firing may be triviallly transformed as 
# 
# $$
# s \ge b  \to s-b \ge 0 \to \sum_{i=1}^n x_i w_i - b \ge 0 \to \sum_{i=1}^n x_i w_i +x_0 w_0 \ge 0 
# \to \sum_{i=0}^n x_i w_i \ge 0,
# $$
# 
# where $x_0=1$ and $w_0=-b$. In other words, we may treat the bias as a weight on the edge connected to an additional cell with input set to 1. This notation is shown in {numref}`MCP2-fig`. Now, the activation function is simply 

# ```{math}
# :label: eq-f
# 
# f(s) = \left \{ \begin{array}{l} 1 {\rm ~for~} s \ge 0 \\ 0 {\rm ~for~} s < 0 \end{array} \right .,
# ```

# $$
# f(s) = \left \{ \begin{array}{l} 1 {\rm ~for~} s \ge 0 \\ 0 {\rm ~for~} s < 0 \end{array} \right .,
# $$

# with the summation index in $s$ starting from $0$:
# 
# ```{math}
# :label: eq-f0
# s=\sum_{i=0}^n x_i w_i = x_0 w_0+x_1 w_1 + \dots + x_n w_n.
# ```

# :::{figure-md} MCP2-fig
# <img src="images/mcp-2a.png" width="320px">
# 
# Alternative, more uniform representation of the MCP neuron, with $x_0=1$ and $w_0=-b$.
# :::

# ```{admonition} Hyperparameters
# The weights $w_0=-b,w_1,\dots,w_n$ are referred to as hyperparameters. They determine the functionality of the MCP neuron and may be changed during the learning (training) process (see the following). However, they are kept fixed when using the trained neuron on a particular input data set.
# ```

# ```{important}
# An essential property of neurons in ANNs is the **nonlinearity** of the activation function. Without this feature, the MCP neuron would simply represent a scalar product, and the feed-forward networks would involve trivial matrix multilications.
# ```

# (mcp_P-lab)=
# ## MCP neuron in Python

# In[1]:


import numpy as np

# plots
import matplotlib.pyplot as plt

# display imported graphics
from IPython.display import display, Image


# We will now implement the mathematical model of the neuron of Sec. {ref}`MCP-lab`. First, we obviously need arrays (vectors), which in Python are represented as

# In[2]:


x = [1,3,7]
w = [1,1,2.5]


# and are indexed starting from 0, e.g.

# In[3]:


x[0]


# The numpy library functions carry the prefix **np**, which is the alias given at import. Note that these fumctions act *distributively* over arrays, e.g.

# In[4]:


np.sin(x)


# which is a convenient feature. We also have the scalar product $x \cdot w = \sum_i x_i w_i$ handy, which we can use to build the combined signal $s$.

# In[5]:


np.dot(x,w)


# Next, we need to construct the neuron activation function, which presently is just the step function {eq}`eq-f`. 

# In[6]:


def step(s): # step function (in neural library)
     if s > 0:
        return 1
     else:
        return 0


# where in the comment we indicate, that the function is also defined in the **neural** library, cf. [Appendix](app-lab). For the visualizers, the plot of the step function is following:

# In[7]:


plt.figure(figsize=(2.3,2.3),dpi=120) # set the size of the figure

s = np.linspace(-2, 2, 100)   # array 100+1 equally spaced points between -2 and 2
fs = [step(z) for z in s]     # corresponding array of function values

plt.xlabel('signal s',fontsize=12)      # axes labels
plt.ylabel('response f(s)',fontsize=12)
plt.title('step function',fontsize=13)  # plot title

plt.plot(s, fs);


# Since $x_0=1$ always, we do not want to explicitly carry this in the argument of the functions that will follow. We will be inserting $x_0=1$ into the input, for instance: 

# In[8]:


x=[5,7]
np.insert(x,0,1) # insert 1 in x at position 0


# Now we are ready to construct the [MCP neuron](MCP1-fig):

# In[9]:


def neuron(x,w,f=step): # (in the neural library)
    """
    MCP neuron

    x: array of inputs  [x1, x2,...,xn]
    w: array of weights [w0, w1, w2,...,wn]
    f: activation function, with step as default
    
    return: signal=weighted sum w0 + x1 w1 + x2 w2 +...+ xn wn = x.w
    """ 
    return f(np.dot(np.insert(x,0,1),w)) # insert x0=1, signal s=x.w, output f(s)


# We diligently put the comments in triple quotes to be able to get the help, when needed:

# In[10]:


help(neuron)


# Note that the function f is an argument of neuron, but it has the default set to step and thus does not have to be present. The sample usage with $x_1=3$, $w_0=-b=-2$, $w_1=1$ is 

# In[11]:


neuron([3],[-2,1])


# As we can see, the neuron fired in this case, as $s=1*(-2)+3*1>0$. Next, we show how the neuron operates on a varying input $x_1$ taken in the range $[-2,2]$. We also change the bias parameter, to illustrate its role. It is clear that the bias works as the threshold: if the signal $x_1 w_1$ is above $b=-x_0$, the neuron fires.

# In[12]:


plt.figure(figsize=(2.3,2.3),dpi=120) 

s = np.linspace(-2, 2, 200)
fs1 = [neuron([x1],[1,1]) for x1 in s]      # more function on one plot
fs0 = [neuron([x1],[0,1]) for x1 in s]
fsm12 = [neuron([x1],[-1/2,1]) for x1 in s]

plt.xlabel('$x_1$',fontsize=12)
plt.ylabel('response',fontsize=12)

plt.title("Change of bias",fontsize=13)

plt.plot(s, fs1, label='b=-1')
plt.plot(s, fs0, label='b=0')
plt.plot(s, fsm12, label='b=1/2')
plt.legend();                               # legend


# When the sign of the weight $w_1$ is negative, we get in some sense a **reverse** behavior, where the neuron fires when $x_1 |w_1| < w_0$: 

# In[13]:


plt.figure(figsize=(2.3,2.3),dpi=120) 

s = np.linspace(-2, 2, 200)
gsm = [neuron([x1],[-1,-1]) for x1 in s]

plt.xlabel('$x_1$',fontsize=12)
plt.ylabel('response',fontsize=12)

plt.plot(s, gsm,label='$w_1=-1$')
plt.legend();


# Note that here (and similarly in other places) the trivial code for the above output is hidden and can be found in the corresponding jupyter notebook.
# 
# Admittedly, in the last example one departs from the biological pattern, as negative weights are not possible to realize in a biological neuron. However, this enriches the mathematical model, which one is free to use without constraints. 

# (bool-sec)=
# ## Boolean functions

# Having constructed the MCP neuron in Python, the question is: *What is the simplest (but still non-trivial) application we can use it for?* We will show here that one can easily construct [boolean functions](https://en.wikipedia.org/wiki/Boolean_function), or logical networks, with the help of networks of MCP neurons. Boolean functions, by definition, have arguments and values in the set $\{ 0,1 \}$, or {True, False}.
# 
# To warm up, let us start with some guesswork, where we take the neuron with the weights $w=[w_0,w_1,w_2]=[-1,0.6,0.6]$ (why not). We shall here denote $x_1=p$, $x_2=q$, in accordance with the traditional notation for logical variables, where $p,q \in \{0,1\}$. 

# In[14]:


print("p q n(p,q)") # print the header
print()

for p in [0,1]:       # loop over p
    for q in [0,1]:   # loop over q
        print(p,q,"",neuron([p,q],[-1,.6,.6])) # print all cases


# We immediately recognize in the above output the logical table for the conjunction, $n(p,q)=p \land q$, or the logical **AND** operation. It is clear how the neuron works. The condition for the firing $n(p,q)=1$ is $-1+p*0.6+q*0.6 \ge 0$, and it is satisfied if and only if $p=q=1$, which is the definition of the logical conjunction. Of course, we could use here 0.7 instead of 0.6, or in general $w_1$ and $w_2$ such that $w_1<1, w_2<1, w_1+w_2 \ge 1$. In the electronics terminology, we can call the present system the **AND gate**.
# 
# We can thus define the short-hand 

# In[15]:


def neurAND(p,q): return neuron([p,q],[-1,.6,.6])


# In[16]:


for p in [0,1]: 
    for q in [0,1]: 
        print(p,q,"",neurAND(p,q))


# Quite similarly, we may define other boolean functions (or logical gates) of two variables. In particular, the NAND gate (the negation of conjunction) and the OR gate (alternative) are realized with the following MCP neurons:

# In[17]:


def neurNAND(p,q): return neuron([p,q],[1,-0.6,-0.6])
def neurOR(p,q):   return neuron([p,q],[-1,1.2,1.2])


# They correspond to the logical tables 

# In[18]:


print("p q  NAND OR") # print the header
print()

for p in [0,1]: 
    for q in [0,1]: 
        print(p,q," ",neurNAND(p,q)," ",neurOR(p,q))


# ```{admonition} Exercise
# :class: warning
# Check, by explicitly generating the logical tables, that the above definitions of gates work properly.
# ```

# ### Problem with XOR

# The XOR gate, or the **exclusive alternative**, is defined with the following logical table:
# 
# $$
# \begin{array}{ccc}
# p & q & p \oplus q \\
# 0 & 0 & 0 \\
# 0 & 1 & 1 \\
# 1 & 0 & 1 \\
# 1 & 1 & 0
# \end{array}
# $$
# 
# This is one of possible boolean functions of two arguments (in total, we have 16 different functions of this kind, why?). We could now try very hard to adjust the weights in our neuron to behave as the XOR gate, but we are doomed to fail. Here is the reson:  

# From the first row of the above table it follows that for the input 0 0 the neuron should not fire. Hence 
# 
# $w_0  + 0* w_1 + 0*w_2  < 0$, or $-w_0>0$. 
# 
# For the cases of rows 2 and 3 the neuron must fire, therefore
# 
# $w_0+w_2 \ge 0$ and $w_0+w_1 \ge 0$.
# 
# Adding side-by-side the three obtained inequalities we get $w_0+w_1+w_2 > 0$. However, the fourth row yields
# $w_0+w_1+w_2<0$ (no firing), so we encounter a contradiction. Therefore no choice of $w_0, w_1, w_2$ exists to do the job! 

# ```{important}
# A single MCP neuron cannot represent the **XOR** gate.
# ```

# ### XOR from composition of AND, NAND and OR

# One can solve the XOR problem by composing three MCP neurons, for instance 

# In[19]:


def neurXOR(p,q): return neurAND(neurNAND(p,q),neurOR(p,q))


# In[20]:


print("p q XOR") # print the header
print()

for p in [0,1]: 
    for q in [0,1]: 
        print(p,q,"",neurXOR(p,q))


# The above construction corresponds to the simple network of {numref}`xor-fig`.
# 

# :::{figure-md} xor-fig
# <img src="images/xor.png" width="260px">
# 
# The XOR gate compsed of the NAND, OR, and AND MCP neurons.
# :::

# Note that we are dealing here, for the first time, with a network having an intermediate layer, consisting of the NAND and OR neurons. This layer is indispensable to construct the XOR gate.

# ### XOR composed from NAND

# Within the theory of logical networks, one proves that any network (or boolean function) can be composed of only NAND gates, or only the NOR gates. One says that the NAND (or NOR) gates are **complete**. In particular, the XOR gate can be constructed as 
# 
# [ p NAND ( p NAND q ) ] NAND [ q NAND ( p NAND q ) ],
# 
# which we can write in Python as

# In[21]:


def nXOR(i,j): return neurNAND(neurNAND(i,neurNAND(i,j)),neurNAND(j,neurNAND(i,j)))


# In[22]:


print("p q XOR") # print the header
print()

for i in [0,1]: 
    for j in [0,1]: 
        print(i,j,"",nXOR(i,j)) 


# In[ ]:




One also proves that the logical networks are complete in the Church-Turing sense, i.e., (when sufficiently large) may carry over any possible calculation. This feature directly carries over to ANNs.

```{admonition} Message
:class: note

Message: ANNs (multilayer, large) can do any calculation!
```
# ```{admonition} Exercises
# :class: warning
# 
# Construct (all in Python)
# 
# - gates NOT, NOR
# - gates OR, AND, NOT by composing gates NAND https://en.wikipedia.org/wiki/NAND_logic 
# - the half adder and full adder https://en.wikipedia.org/wiki/Adder_(electronics)
# 
# as networks of MCP neurons.
# ```
