#!/usr/bin/env python
# coding: utf-8

# # Interpolation

# In[1]:


import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys # system library
sys.path.append('./lib_nn') 
# my path (linux, OS)

from neural import * # import my library package


# ## Simulated data

# So far we have been concerned with **classification**, i.e. with networks recognizing whether a given object (in our case a point on a plane) has certain features. Now we pass to another practical application, namely **interpolating functions**. This use of ANNs has become widely used in scientific data analysis. We illustrate the method on a simple example, which explains the basic idea and shows how the method works.
# 
# Imagine you have some experimental data. Here we simulate them in an artificial way, e.g.

# In[2]:


def fi(x):
    return 0.2+0.8*np.sin(x)+0.5*x-3 # some function

def data(): 
    x = 7.*np.random.rand() # random x coordinate
    y = fi(x)+0.4*func.rn() # y coordinate = the function + noise from [-0.2,0.2]
    return [x,y]


# We should now think in terms of supervised learning: $x$ is the "feature", and $y$ the "label".
# 
# We table our noisy data points and plot them together with the function **fi(x)** around which they concentrate. It is an imitation of an experimental measurement, which is always burdened with some error, here mimicked wih random noise.

# In[3]:


tab=np.array([data() for i in range(200)])    # data sample
features=np.delete(tab,1,1)                   # x coordinate
labels=np.delete(tab,0,1)                     # y coordinate


# In[4]:


plt.figure(figsize=(2.8,2.3),dpi=120)

coo=np.linspace(0,7,25)

exact=[fi(x) for x in coo]

plt.plot(coo,exact,c='black',label='original function')

plt.title("Simulated experiment",fontsize=11) 
plt.scatter(features,labels,s=2,label='data')

plt.legend(prop={'size':10})

plt.xlabel('$x$',fontsize=11)
plt.ylabel('$y$',fontsize=11);


# In our current language of ANNs, we therefore have a training sample consisting of points with the input (feature) $x$ and the true output (label) $y$. As before, we minimize the error function from an appropriate neural network,
# 
# $$E(\{w \}) = \sum_p (y_o^{(p)} - y^{(p)})^2. $$
# 
# Since the obained $y_o$ is a certain (weight-dependent) function of $x$, this method is a variant of the **least squares fit**, commonly used in data analysis. The difference is that in the standard least squares method the model function that we fit to the data has some simple analytic form (e.g. $ f(x) = A + B x$), while now it is some "disguised" weight-dependent function provided by the neural network.

# ## ANNs for interpolation

# To understand the fundamental idea, consider a network with just two neurons in the middle layer, with the sigmoid activation function:

# In[5]:


draw.plot_net([1,2,1]);


# The signals entering the two neurons in the middle layer are, in the notation of chapter {ref}`more-lab`,
# 
# $$s_1^{1}=w_{01}^{1}+w_{11}^{1} x, $$
# 
# $$s_2^{1}=w_{02}^{1}+w_{12}^{1} x, $$
# 
# and the outgoing signals are, correspondingly,
# 
# $$\sigma \left( w_{01}^{1}+w_{11}^{1} x \right), $$
# 
# $$\sigma \left( w_{02}^{1}+w_{12}^{1} x \right). $$
# 
# Therefore the combined signal entering the output neuron is 
# 
# $$s_1^{1}=w_{01}^{2}+ w_{11}^{2}\sigma \left( w_{01}^{1}+w_{11}^{1} x \right)
# +  w_{21}^{2}\sigma \left( w_{02}^{1}+w_{12}^{1} x \right). $$ 
# 
# Taking, for an illustation, the weight values 
# 
# $$w_{01}^{2}=0, \, w_{11}^{2}=1, \, w_{21}^{2}=-1, \,
# w_{11}^{1}=w_{12}^{1}=1, \, w_{01}^{1}=-x_1,  \, w_{02}^{1}=-x_2, $$
# 
# where $x_1$ and $x_2$ are parameters with a natural interpretation explained below, we get 
# 
# $s_1^{1}=\sigma(x-x_1)-\sigma(x-x_2)$.
# 
# This function is shown in the plot below, with $x_1=-1$ and $x_2=4$. 
# It tends to 0 at $- \infty$, then grows with $x$ to achieve a maximum at
# $(x_1+x_2)/2$, and then decreases, tending to 0 at $+\infty$. At $x=x_1$ and $x=x_2$, the values are around 0.5.

# In[6]:


plt.figure(figsize=(2.8,2.3),dpi=120)

s = np.linspace(-10, 15, 100)

fs = [func.sig(z+1)-func.sig(z-4) for z in s]

plt.xlabel('x',fontsize=11)
plt.ylabel('$\sigma(x+1)-\sigma(x-4)$',fontsize=10)

plt.plot(s, fs);


# This is an important observation:
# We are able to form, with a pair of neurons, a "hump" signal located around a given value, here $ (x_1 + x_2) / 2 = 2$, and with a spread of the order of $|x_2-x_1|$. Changing the weights, we are able to modify its shape, width, and height.
# 
# One may think as follows: Imagine we have many neurons to our disposal in the intemediate layer. We may join them in pairs, forming humps "specializing" in particular regions of coordinates. Then, adjusting the heights of the humps, we may readily approximate a given function. 
# 
# In an actual fitting procedure, we do not need to "join the neurons in pairs", but make a combined fit of all parameters simutaneously, as we did in the case of classifiers. 
# The example below shows a composition of 8 sigmoids,
# 
# $$
# f = \sigma(z+3)-\sigma(z+1)+2 \sigma(z)-2\sigma(z-4)+
#       \sigma(z-2)-\sigma(z-8)-1.3 \sigma(z-8)-1.3\sigma(z-10). 
# $$
# 
# In the figure, the component functions (the thin lines representing single humps) add up to a function of a rather complicated shape, marked with a thick line. 

# In[7]:


plt.figure(figsize=(2.8,2.3),dpi=120)

s = np.linspace(-10, 15, 100)

f1 = [func.sig(z+3)-func.sig(z+1) for z in s]
f2 = [2*(func.sig(z-0)-func.sig(z-4)) for z in s]
f3 = [func.sig(z-2)-func.sig(z-8) for z in s]
f4 = [-1.3*(func.sig(z-8)-func.sig(z-10)) for z in s]

fs = [func.sig(z+3)-func.sig(z+1)+2*(func.sig(z-0)-func.sig(z-4))+
      func.sig(z-2)-func.sig(z-8)-1.3*(func.sig(z-8)-func.sig(z-10)) 
       for z in s]

plt.xlabel('x',fontsize=11)
plt.ylabel('combination of sigmoids',fontsize=10)

plt.plot(s, fs, linewidth=4)
plt.plot(s, f1)
plt.plot(s, f2)
plt.plot(s, f3)
plt.plot(s, f4);


# There is an important difference in ANNs used for function approximation compared to the binary classifiers discussed earlier. There, the answers were 0 or 1, so we were using a step function in the output layer, or rather its smooth sigmoid variant. For function approximation, the answers form a continuum from the range of the function values. For that reason, in the output layer we just use the **linear** function, i.e., we just pass the incoming signal through. Of course, sigmoids remain in the intermediate layers.
# 
# ```{admonition} Output layer for function approximation
# :class: important
# 
# In ANNs used for function approximation, the activation function in the output layer is **linear**.
# ```

# ### Backprop for one-dimensional functions

# Minimization of the error function leads to the backprop algorithm (with a modified output layer), which we employ to fit our experimental data. Let us take the architecture:

# In[8]:


arch=[1,6,1]


# In[9]:


draw.plot_net(arch);


# and the weights

# In[10]:


weights=func.set_ran_w(arch, 5)


# As just discussed, the output is no longer between 0 and 1: 

# In[11]:


x=func.feed_forward_o(arch, weights,features[1],ff=func.sig,ffo=func.lin)
draw.plot_net_w_x(arch, weights,1,x);


# In the library module **func** we have the function for the backprop algorithm which allows for one activation function in the intermediate layers and a different one in the output layer. We carry out the training in two stages:

# In[12]:


eps=0.02                           # initial learning speed
for k in range(30):                # rounds
    for p in range(len(features)): # loop over the data sample points
        pp=np.random.randint(len(features)) # random point
        func.back_prop_o(features,labels,pp,arch,weights,eps,
                         f=func.sig,df=func.dsig,fo=func.lin,dfo=func.dlin)


# In[13]:


for k in range(400):               # rounds
    eps=0.999*eps                  # dicrease of the learning speed
    for p in range(len(features)): # loop over points taken in sequence
        func.back_prop_o(features,labels,p,arch,weights,eps,
                         f=func.sig,df=func.dsig,fo=func.lin,dfo=func.dlin)


# In[14]:


draw.plot_net_w(arch,weights,.2);


# which nicely does the job:

# In[15]:


res=[func.feed_forward_o(arch, weights, [x], ff=func.sig, ffo=func.lin)[2][0] for x in coo]

plt.figure(figsize=(2.8,2.3),dpi=120)

plt.title("Fit to data",fontsize=11) 
plt.scatter(features,labels,s=2)


plt.plot(coo,res,c='red',label='fit')
plt.plot(coo,exact,c='black',label='original function')

plt.legend(prop={'size':10})

plt.xlabel('$x$',fontsize=11)
plt.ylabel('$y$',fontsize=11);


# We note that the obtained red curve is very close to the function used to generate the data sample (black line). This shows that the approximation works. A construction of a quantitative measure (least square sum) is a topic of a homework problem.

# ## Remarks

# ```{note}
# 
# The activation function in the output layer may be any function with values containing the values of the interpolated function, not necessarily linear. 

# ```{admonition} More dimensions
# :class: important
# 
# To interpolate general functions of two or more arguments, one needs use ANNs with at least 3 neuron layers.
# ```

# ```{tip} 
# The number of neurons reflects the behavior of the interpolated function. If the function varies a lot, one needs more neurons, typically at least twice as many as the number of extrema.
# ```

# ```{admonition} Overfitting
# :class: important
# 
# There must be much more data for fitting than the network parameters, to avoid the so-called overfitting problem.
# ```

# ```{admonition} Exercises
# :class: warning
# 
# 1. Fit the data points generated by your favorite function (of one variable) with noise. Play with the network architecture.
# 
# 2. Compute the sum of squared distances of the values of the data points and the corresponding approximating function, and use it as a measure of the goodness of the fit. Use it to test how the number of neurons in the network affects the result. 
# 
# 3. Use a network with more layers (at least 3 neuron layers) to fit the data points generated with your favorite two-variable function. Make two-dimensional contour plots for this function and for the function obtained from the neural network and compare the results (of course, they should be very similar if everything works).
# ```
