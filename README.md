## What is the signature method?

This repo is dedicated to the theory and practice of the signature method. This is a novel and powerful method for sequential data representation and feature extraction derived from the theory of <a href="https://en.wikipedia.org/wiki/Rough_path" target="_blank">rough paths</a> in stochastic analysis. 

The website is in early stages of preparation, but we are working hard to fill in the content as soon as possible.

In the meantime, you can explore and familiarise yourself with some recent works in machine learning where the signature method has played a major role.

General overview and real-world examples from the founder of the signature method, <a href="https://en.wikipedia.org/wiki/Terry_Lyons_(mathematician)" target="_blank">Professor Terry Lyons</a>.

<a href="https://arxiv.org/pdf/1405.4537.pdf" target="_blank">Rough paths, Signatures and the modelling of functions on streams</a>

A gentle introduction with lots of explained examples and an overview of recent application of the method are concisely represented in the primer:

<a href="https://arxiv.org/pdf/1603.03788.pdf" target="_blank">A Primer on the Signature Method in Machine Learning</a>

For mathematical ninjas and those with solid background in statistical machine learning, the next step will be this article:

<a href="https://arxiv.org/pdf/1601.08169.pdf" target="_blank">Kernels for sequentially ordered data</a>

Pure mathematicians and audience working with time-series data will highly appreciate this seminal work:

<a href="https://arxiv.org/pdf/1309.0260.pdf" target="_blank">Learning from the past, predicting the statistics for the future, learning an evolving system</a>


Here is the collection of real applications of the signature method in different domains:


Deep learning and characters recognition:

<a href="https://arxiv.org/pdf/1308.0371.pdf" target="_blank">Sparse arrays of signatures for online character recognition</a>


Finance and stock market:

<a href="https://arxiv.org/pdf/1307.7244.pdf" target="_blank">Extracting information from the signature of a financial data stream</a>


Medicine and mental health:

<a href="https://arxiv.org/pdf/1606.02074.pdf" target="_blank">Application of the Signature Method to Pattern Recognition in the CEQUEL Clinical Trial</a>

Deep learning and recurrent neural networks:

<a href="https://arxiv.org/pdf/1705.06849.pdf" target="_blank">Online Signature Verification using Recurrent Neural Network and Length-normalized Path Signature</a>


Last, but not least, the classic book on the theory of rough paths, it is a collection of lectures given at Saint-Flour school.
<a href="https://link.springer.com/book/10.1007%2F978-3-540-71285-5" target="_blank">Differential Equations Driven by Rough Paths</a>


Now let's delve into the Signature Method.


## A path from discretely sampled data

The key ingredient of the signature method is a path constructed from data. The path is a continuous piece-wise interpolation of data points. For example, consider a collection of pairs ![f1], where _t_ may be thought as time component and _X_ is a stock price: 

![alt text](https://github.com/kormilitzin/the-signature-method-in-machine-learning/blob/master/t_X_path_example.png)












[f1]: http://mathurl.com/y9pjrdyy.png
