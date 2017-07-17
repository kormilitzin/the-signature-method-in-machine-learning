## What is the signature method?

This repo is dedicated to the theory and practice of the signature method. This is a novel and powerful method for sequential data representation and feature extraction derived from the theory of <a href="https://en.wikipedia.org/wiki/Rough_path" target="_blank">rough paths</a> in stochastic analysis. 

The website is in early stages of preparation, but we are working hard to fill in the content as soon as possible.

In the meantime, you can explore and familiarise yourself with some recent works in machine learning where the signature method has played a major role.

General overview and real-world examples from the founder of the signature method, <a href="https://en.wikipedia.org/wiki/Terry_Lyons_(mathematician)" target="_blank">Professor Terry Lyons</a>: <a href="https://arxiv.org/pdf/1405.4537.pdf" target="_blank">Rough paths, Signatures and the modelling of functions on streams</a>

A gentle introduction with lots of explained examples and an overview of recent application of the method are concisely represented in: <a href="https://arxiv.org/pdf/1603.03788.pdf" target="_blank">A Primer on the Signature Method in Machine Learning</a>

For mathematical ninjas and those with solid background in statistical machine learning, the next step will be this article: <a href="https://arxiv.org/pdf/1601.08169.pdf" target="_blank">Kernels for sequentially ordered data</a>

Pure mathematicians and audience working with time-series data will highly appreciate this seminal work: <a href="https://arxiv.org/pdf/1309.0260.pdf" target="_blank">Learning from the past, predicting the statistics for the future, learning an evolving system</a>


Here is the collection of real applications of the signature method in different domains:


Deep learning and characters recognition: <a href="https://arxiv.org/pdf/1308.0371.pdf" target="_blank">Sparse arrays of signatures for online character recognition</a>


Finance and stock market: <a href="https://arxiv.org/pdf/1307.7244.pdf" target="_blank">Extracting information from the signature of a financial data stream</a>


Medicine and mental health: <a href="https://arxiv.org/pdf/1606.02074.pdf" target="_blank">Application of the Signature Method to Pattern Recognition in the CEQUEL Clinical Trial</a>

Deep learning and recurrent neural networks: <a href="https://arxiv.org/pdf/1705.06849.pdf" target="_blank">Online Signature Verification using Recurrent Neural Network and Length-normalized Path Signature</a>


Last, but not least, the classic book on the theory of rough paths, it is a collection of lectures given at Saint-Flour school: <a href="https://link.springer.com/book/10.1007%2F978-3-540-71285-5" target="_blank">Differential Equations Driven by Rough Paths</a>


Now let's delve into the Signature Method.

## Installation of the ESig package

Practical applications of the signature method (SM) to machine learning and data analysis tasks can be performed using the `ESig` package. The package is written in C++ with a user-friendly Python wrapper. In the near future (expected Nov 2017), the ESig package will be available through a standard `pip install ESig` command line method, but in the meantime, one can download the installer from the <a href="https://sourceforge.net/projects/coropa/files/ESig-2017-06-07/">CoRoPa repository</a>, the `ESig-2017-06-07.7z` zipped file. 

In this tutorial we consider a pragmatic approach to demonstrate the SM at work: starting with practical examples and then moving towards theory. Once you practiced the basics, it will be easier to understand the underlying mathematical foundations and think about future application of the method.

To check the installation, first import the ESig package into your Python session (Here is Python 2.7):

```python
import numpy as np
import esig.tosig as ts
```
and run a simple example: 

```python
two_dim_stream = np.random.random(size=(10,2))
print ts.stream2sig(two_dim_stream, 2)
[1.0, -0.10661163, -0.69629065, 0.00568302, -0.03958541, 0.11381809, 0.242421033]
```
_(the output will depend on your current seed of a random generator and will be different from the presented one, except from the very first term `1`, which is always 1 and will be explained below)_

## A path from ordered data

The key ingredient of the signature method is a path constructed from data. The path is a continuous piece-wise interpolation of data points. For example, consider a collection of pairs ![f1], where _t_ = _{0,1,2,3}_ may be thought as time component and _X_ = _{6,1,8,2}_ is a stock price: 

![Figure (@fig_1): Example of embedding](https://github.com/kormilitzin/the-signature-method-in-machine-learning/blob/master/t_X_path_example_1.png)


*Note: For the sake of simplicity and further computational examples, we considered integer numbers only. Of course there is no conceptual restriction to use real numbers*. 
The red dots are discrete data points and the blue solid line is a path continuously connecting the data points. In fact, we took two 1-dimensional sequences and embedded into a single (1-dim) path in 2-dimension. Generalising the idea, any collection of _d_ 1-dim paths can be embedded into a single path in _d_-dimensions. 

The signature is a transformation (mapping) from a path into a collection of real-valued numbers. Each term in the collection has a particular (geometrical) meaning as a function of data points.

The truncated signature at level (depth) `L=2` of the path presented in above is obtained by running the code:

```python
import numpy as np
import esig.tosig as ts
t_i = np.arange(4)
X_i = np.array([6, 1, 8, 2]).astype(float)
two_dim_stream = np.reshape( zip(t_i, X_i), newshape=(-1,2) )
signatures = ts.stream2sig(two_dim_stream, 2)
print signatures
[1., 3., -4., 4.5., -7., -5., 8.]
```
#Important#: the input array should always be in the form: `length_of_stream x dimension_of_stream`. For example, two dimensional array consisting of 4 points, should be reshaped as `4x2` array (rows are data points and columns are unique streams).

It is crucial to understand each term in the resulting signature. The general form of the signatures is given by iterated integrals (_projections_ or _coordinates_) of a path.

The first term of the signature is always `1`, as it corresponds to a zeroth-order approximation (or a constant). 








[f1]: http://mathurl.com/y9pjrdyy.png
