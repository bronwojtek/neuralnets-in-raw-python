"""
Contains functions used in the lecture
"""

import numpy as np


def step(s):
    """
    step function
    
    s: signal
    
    return: 1 if s>0, 0 otherwise
    """
    if s>0:
        return 1
    else:
        return 0
   
   
def neuron(x,w,f=step):
    """
    MCP neuron

    x: array of inputs  [x1, x2,...,xn]
    w: array of weights [w0, w1, w2,...,wn]
    f: activation function, with step as default
    
    return: signal=f(w0 + x1 w1 + x2 w2 +...+ xn wn) = f(x.w)
    """
    return f(np.dot(np.insert(x,0,1),w))
 
 
def sig(s,T=1):
    """
    sigmoid
     
    s: signal
    T: temperature
    
    return: sigmoid(s)
    """
    return 1/(1+np.exp(-s/T))
    
    
def dsig(s, T=1):
    """
    derivative of sigmoid
    
    s: signal
    T: temperature
    
    return: dsigmoid(s,T)/ds
    """
    return sig(s)*(1-sig(s))/T
    
    
def lin(s,a=1):
    """
    linear function
    
    s: signal
    a: constant
    
    return: a*s
    """
    return a*s
  
  
def dlin(s,a=1):
    """
    derivative of linear function
    
    s: signal
    a: constant
    
    return: a
    """
    return a


def relu(s):
    """
    ReLU function
    
    s: signal
    
    return: s if s>0, 0 otherwise
    """
    if s>0:
        return s
    else:
        return 0


def drelu(s):
    """
    derivative of ReLU function
    
    s: signal
    
    return: 1 if s>0, 0 otherwise
    """
    if s>0:
        return 1
    else:
        return 0
 
 
def lrelu(s,a=0.1):
    """
    Leaky ReLU function
  
    s: signal
    a: parameter
    
    return: s if s>0, a*s otherwise
    """
    if s>0:
        return s
    else:
        return a*s


def dlrelu(s,a=0.1):
    """
    derivative of Leaky ReLU function
    
    s: signal
    a: parameter
    
    return: 1 if s>0, a otherwise
    """
    if s>0:
        return 1
    else:
        return a


def softplus(s):
    """
    softplus function

    s: signal

    return: log(1+exp(s))
    """
    return np.log(1+np.exp(s))


def dsoftplus(s):
    """
    derivative of softplus function
 
    s: signal

    return: 1/(1+exp(-s))
    """
    return 1/(1+np.exp(-s))

    
def l2(w0,w1,w2):
    """for separating line"""
    return [-.1,1.1],[-(w0-w1*0.1)/w2,-(w0+w1*1.1)/w2]


def eucl(p1,p2):
    """
    Square of the Euclidean distance between two points in 2-dim. space
    
    input: p1, p2 - arrays in the format [x1,x2]
    
    return: square of the Euclidean distance
    """
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2


def rn():
    """
    return: random number from [-0.5,0.5]
    """
    return np.random.rand()-0.5
 
 
def point_c():
    """
    return: array [x,y] with random point from a cirle
            centered at [0.5,0.5] and radius 0.4
            (used for examples)
    """
    while True:
        x=np.random.random()
        y=np.random.random()
        if (x-0.5)**2+(y-0.5)**2 < 0.4**2:
            break
    return np.array([x,y])
 
 
def point():
    """
    return: array [x,y] with random point from [0,1]x[0,1]
    """
    x=np.random.random()
    y=np.random.random()
    return np.array([x,y])


def set_ran_w(ar,s=1):
    """
    Set network weights randomly
    
    input:
    ar - array of numbers of nodes in subsequent layers [n_0, n_1,...,n_l]
    (from input layer 0 to output layer l, bias nodes not counted)
    
    s - scale factor: each weight is in the range [-0.s, 0.5s]
    
    return:
    w - dictionary of weights for neuron layers 1, 2,...,l in the format
    {1: array[n_0+1,n_1],...,l: array[n_(l-1)+1,n_l]}
    """
    l=len(ar)
    w={}
    for k in range(l-1):
        w.update({k+1: [[s*rn() for i in range(ar[k+1])] for j in range(ar[k]+1)]})
    return w


def set_val_w(ar,a=0):
    """
    Set network weights to a constant value
    
    input:
    ar - array of numbers of nodes in subsequent layers [n_0, n_1,...,n_l]
    (from input layer 0 to output layer l, bias nodes not counted)
    
    a - value for each weight
    
    return:
    w - dictionary of weights for neuron layers 1, 2,...,l in the format
    {1: array[n_0+1,n_1],...,l: array[n_(l-1)+1,n_l]}
    """
    l=len(ar)
    w={}
    for k in range(l-1):
        w.update({k+1: [[a for i in range(ar[k+1])] for j in range(ar[k]+1)]})
    return w
    

def feed_forward(ar, we, x_in, ff=step):
    """
    Feed-forward propagation
    
    input:
    ar - array of numbers of nodes in subsequent layers [n_0, n_1,...,n_l]
    (from input layer 0 to output layer l, bias nodes not counted)
    
    we - dictionary of weights for neuron layers 1, 2,...,l in the format
    {1: array[n_0+1,n_1],...,l: array[n_(l-1)+1,n_l]}
    
    x_in - input vector of length n_0 (bias not included)
    
    ff - activation function (default: step)
    
    return:
    x - dictionary of signals leaving subsequent layers in the format
    {0: array[n_0+1],...,l-1: array[n_(l-1)+1], l: array[nl]}
    (the output layer carries no bias)
    """
    l=len(ar)-1                   # number of neuron layers
    x_in=np.insert(x_in,0,1)      # input, with the bias node inserted
    
    x={}                          # empty dictionary
    x.update({0: np.array(x_in)}) # add input signal
    
    for i in range(0,l-1):        # loop over layers till before last one
        s=np.dot(x[i],we[i+1])    # signal, matrix multiplication
        y=[ff(s[k]) for k in range(ar[i+1])] # output from activation
        x.update({i+1: np.insert(y,0,1)}) # add bias node and update x

    # the last layer - no adding of the bias node
    s=np.dot(x[l-1],we[l])
    y=[ff(s[q]) for q in range(ar[l])]
    x.update({l: y})          # update x
          
    return x


def back_prop(fe,la, p, ar, we, eps,f=sig, df=dsig):
    """
    back propagation algorithm
    
    fe - array of features
    la - array of labels
    p  - index of the used data point
    ar - array of numbers of nodes in subsequent layers
    we - dictionary of weights - UPDATED
    eps - learning speed
    f   - activation function
    df  - derivative of f
    """
 
    l=len(ar)-1 # number of neuron layers (= index of the output layer)
    nl=ar[l]    # number of neurons in the otput layer
   
    x=feed_forward(ar,we,fe[p],ff=f) # feed-forward of point p
   
    # formulas from the derivation in a one-to-one notation:
    
    D={}
    D.update({l: [2*(x[l][gam]-la[p][gam])*
                    df(np.dot(x[l-1],we[l])[gam]) for gam in range(nl)]})
    we[l]-=eps*np.outer(x[l-1],D[l])
    
    for j in reversed(range(1,l)):
        u=np.delete(np.dot(we[j+1],D[j+1]),0)
        v=np.dot(x[j-1],we[j])
        D.update({j: [u[i]*df(v[i]) for i in range(len(u))]})
        we[j]-=eps*np.outer(x[j-1],D[j])


def feed_forward_o(ar, we, x_in, ff=sig, ffo=lin):
    """
    Feed-forward propagation with different output activation
    
    input:
    ar - array of numbers of nodes in subsequent layers [n_0, n_1,...,n_l]
    (from input layer 0 to output layer l, bias nodes not counted)
    
    we - dictionary of weights for neuron layers 1, 2,...,l in the format
    {1: array[n_0+1,n_1],...,l: array[n_(l-1)+1,n_l]}
    
    x_in - input vector of length n_0 (bias not included)
    
    f  - activation function (default: sigmoid)
    fo - activation function in the output layer (default: linear)
    
    return:
    x - dictionary of signals leaving subsequent layers in the format
    {0: array[n_0+1],...,l-1: array[n_(l-1)+1], l: array[nl]}
    (the output layer carries no bias)
    
    """
    l=len(ar)-1                   # number of neuron layers
    x_in=np.insert(x_in,0,1)      # input, with the bias node inserted
    
    x={}                          # empty dictionary
    x.update({0: np.array(x_in)}) # add input signal
    
    for i in range(0,l-1):        # loop over layers till before last one
        s=np.dot(x[i],we[i+1])    # signal, matrix multiplication
        y=[ff(s[k]) for k in range(ar[i+1])] # output from activation
        x.update({i+1: np.insert(y,0,1)}) # add bias node and update x

    # the last layer - no adding of the bias node
    s=np.dot(x[l-1],we[l])
    y=[ffo(s[q]) for q in range(ar[l])] # output activation function
    x.update({l: y})                    # update x
          
    return x


def back_prop_o(fe,la, p, ar, we, eps, f=sig, df=dsig, fo=lin, dfo=dlin):
    """
    backprop with different output activation
    
    fe - array of features
    la - array of labels
    p  - index of the used data point
    ar - array of numbers of nodes in subsequent layers
    we - dictionary of weights - UPDATED
    eps - learning speed
    f   - activation function
    df  - derivative of f
    fo  - activation function in the output layer (default: linear)
    dfo - derivative of fo
    """
    l=len(ar)-1 # number of neuron layers (= index of the output layer)
    nl=ar[l]    # number of neurons in the otput layer
   
    x=feed_forward_o(ar,we,fe[p],ff=f,ffo=fo) # feed-forward of point p
   
    # formulas from the derivation in a one-to-one notation:
    
    D={}
    D.update({l: [2*(x[l][gam]-la[p][gam])*
                   dfo(np.dot(x[l-1],we[l])[gam]) for gam in range(nl)]})
    
    we[l]-=eps*np.outer(x[l-1],D[l])
    
    for j in reversed(range(1,l)):
        u=np.delete(np.dot(we[j+1],D[j+1]),0)
        v=np.dot(x[j-1],we[j])
        D.update({j: [u[i]*df(v[i]) for i in range(len(u))]})
        we[j]-=eps*np.outer(x[j-1],D[j])
