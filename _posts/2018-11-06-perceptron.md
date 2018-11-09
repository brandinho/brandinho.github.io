---
title: "Learning Probability Distributions in Bounded Action Spaces"
date: 2018-11-07
tags: [Reinforcement Learning, Bayesian, Reparameterization]
header:
  image: "/images/ML-banner.jpg"
excerpt: "Reinforcement Learning, Neural Networks, Bayesian"
mathjax: "true"
---

## Overview

In this post we will learn how to apply reinforcement learning in a probabilistic manner. More specifically, we will be looking at some of the difficulties in applying conventional approaches to bounded action spaces, and provide a solution.

## Reinforcement Learning Background

I am not going to provide a complete background of Reinforcement Learning because there are already some excellent resources online such as [Arthur Juliani's blogs](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) and [David Silver's lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). I highly recommend going through both of them to get a solid understanding of the fundamentals.

With that said, I will explain some of the basics needed to understand this blog.

Provide some basic background on RL:
* Markov Decision Process
* Discrete vs Continuous Policies
* Deterministic vs Probabilistic Continuous Policies

## Probability Background

Provide some basic background on Bayes Theorem:
* We are trying to learn a posterior distribution
* Different ways to make a bayesian policy
* Problem with bounded action spaces
* One solution can be to use a bounded activation function and approximate a bayesian neural network with MC dropout

## Novel Solution

Present our novel solution to the problem. We will show empirically and prove mathematically that our approach is superior to the bayesian approximation.
* Explain the challenge in clipping samples outside the bounds
* Walk through reparameterization for the unbounded case
* Expand it for our solution

## Concluding Remarks

Some conclusions


And here's some *italics*

Here's some **bold** text

What about a [link](https://github.com/brandinho)

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
  import numpy as np

  def test_function(x, y):
    z = np.sum(x, y)
    return z
```

R code block:
```r
  library(tidyverse)
  df <- read_csv("some_file.csv")
  head(df)
```

here's some inline code `x+y`

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/lin-sep.jpg" alt="linearly serparable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/lin-sep.jpg)

Here's some math:

$$z=x+y$$

Here's some inline math $$z=x+y$$
