#!/usr/bin/env python
# coding: utf-8

# # Concluding remarks

# In a programmer's life, building a well-functioning ANN, even for simple problems as used for illustrations in these lectures, can be a truly frustrating experience! There are many subtleties involved on the way. To list just a few that we have encountered in this course, one faces a choice of the network architecture and freedom in the initialization of weights (hyperparameters). Then, one has to select an initial learning speed, a neighborhood distance, in general, some parameters controlling the performance/convergence, as well as their update strategy as the algorithm progresses. Further, one frequently tackles with an emergence of a massive number of local minima in the space of hyperparameters, and many optimization methods may be applied here, way more sophisticated than our simplest steepest-descent method. Moreover, a proper choice of the neuron activation functions is crucial for success, which relates to avoiding the problem of "dead neurons". And so on and on, many choices to be made before we start gazing in the screen in hopes that the results of our code converge ... 
# 
# ```{important}
# Taking the right decisions for the above issues is an **art** more than science, based on long experience of multitudes of code developers and piles of empty pizza boxes!
# ```
# 
# Now, having completed this course and understood the basic principles behind the simplest ANNs inside out, the reader may safely jump to using professional tools of modern machine learning, with the conviction that inside the black boxes there sit essentially the same little codes he met here, but with all the knowledge, experience, tricks, provisions, and options built in. Achieving this conviction, through appreciation of simplicity, has been one of the guiding goals of this course. 

# ## Acknowledgments

# The author thanks [Jan Broniowski](https://www.linkedin.com/in/janbroniowski) for priceless technical help and for remarks to the text.

# ```{bibliography} ../_bibliography/references.bib
# ```
