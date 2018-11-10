---
title: "Learning Probability Distributions in Bounded Action Spaces"
date: 2018-11-07
tags: [Reinforcement Learning, Bayesian, Reparameterization]
header:
  image: "/images/ML-banner.jpg"
excerpt: "Reinforcement Learning, Neural Networks, Bayesian"
mathjax: "true"
comment: true
---

## Overview

In this post we will learn how to apply reinforcement learning in a probabilistic manner. More specifically, we will be looking at some of the difficulties in applying conventional approaches to bounded action spaces, and provide a solution.

## Reinforcement Learning Background

I am not going to provide a complete background on Reinforcement Learning (RL) because there are already some excellent resources online such as [Arthur Juliani's blogs](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) and [David Silver's lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). I highly recommend going through both to get a solid understanding of the fundamentals. With that said, I will explain some concepts that are important for this blog post.

At the most basic level, the goal of RL is to learn a mapping from states to actions. To understand what this means, I think it is important to take a step back and understand the RL framework more generally. Cue the overused RL diagram:

{:refdef: style="text-align: center;"}
![alt]({{ site.url }}{{ site.baseurl }}/images/AgentEnvironment.jpg)
{: refdef}

The first thing to notice is that there is a feedback loop between the agent and the environment. For clarity, the agent refers to the AI that we are creating, while the environment refers to the world that the agent has to navigate through. In order to navigate through an environment, the agent has to take actions. The specific actions will depend on the domain - we will describe a few fairly soon. After the agent takes an action, it receives an observation of the environment (the current state) and a reward (assuming we don't have sparse rewards).

After interacting with the environment for long enough, we hope that our agent learns how to take actions that maximize its cumulative reward over the long-term. It is important to realize that the best action in one state is not necessarily the best action in another state. So going back to our statement about mapping states to actions, this simply means that we want our agent to learn the best actions to take in each environment state.

Now let's talk a bit about actions an agent can take. The first distinction I would like to make is between discrete actions and continuous actions. When we refer to discrete actions, we simply mean that there is a finite set of possible actions an agent can take. For example, in pong an agent can decide to move up or down. On the other hand, continuous actions have an infinite number of possibilities. An example of a continuous action, although kind of silly, is the hiding position of an agent if it is playing hide and seek.

Given enough time, the agent can theoretically hide anywhere - so the action space is unbounded. In contrast, we can have a continuous action space that is bounded. An example close to my heart is position sizing when trading a financial asset. The bounds are -1 (100% Short) and 1 (100% Long). To map states to that bounded action space, we can use $$\tanh$$ in the final layer of a neural network. That seems pretty easy... so why am I writing a blog post about it? Often times we need more than just a deterministic output, especially when the underlying data has a low signal-to-noise ratio. The additional piece of information that we need is *uncertainty* in our agent's decision. We will use a Bayesian approach to model a posterior distribution and sample from this distribution to estimate the uncertainty. Don't worry if that doesn't completely make sense yet - it will by the end of this post!

## Probability Background

For a great introduction to Bayesian statistics I suggest reading [Will Kurt's blog](https://www.countbayesie.com) - Count Bayesie. It's awesome.

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

![Alt Text](/images/posterior.gif)

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
