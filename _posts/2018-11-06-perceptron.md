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

In this post we will learn how to apply reinforcement learning in a probabilistic manner. More specifically, we will be looking at some of the difficulties in applying conventional approaches to bounded action spaces, and provide a solution. This blog assumes you have knowledge in deep learning. If not, check out [Michael Nielsen's book](http://neuralnetworksanddeeplearning.com/chap1.html) - it is very comprehensive and easy to understand.

## Reinforcement Learning Background

I am not going to provide a complete background on Reinforcement Learning (RL) because there are already some excellent resources online such as [Arthur Juliani's blogs](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) and [David Silver's lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). I highly recommend going through both to get a solid understanding of the fundamentals. With that said, I will explain some concepts that are important for this blog post.

At the most basic level, the goal of RL is to learn a mapping from states to actions. To understand what this means, I think it is important to take a step back and understand the RL framework more generally. Cue the overused RL diagram:

{:refdef: style="text-align: center;"}
![alt]({{ site.url }}{{ site.baseurl }}/images/AgentEnvironment.jpg)
{: refdef}

The first thing to notice is that there is a feedback loop between the agent and the environment. For clarity, the agent refers to the AI that we are creating, while the environment refers to the world that the agent has to navigate through. In order to navigate through an environment, the agent has to take actions. The specific actions will depend on the domain - we will describe a few fairly soon. After the agent takes an action, it receives an observation of the environment (the current state) and a reward (assuming we don't have sparse rewards).

After interacting with the environment for long enough, we hope that our agent learns how to take actions that maximize its cumulative reward over the long-term. It is important to realize that the best action in one state is not necessarily the best action in another state. So going back to our statement about mapping states to actions, this simply means that we want our agent to learn the best actions to take in each environment state. The function that maps states to actions is called a policy and is denoted as $$\pi(a \mid s)$$.

Now let's talk a bit about actions an agent can take. The first distinction I would like to make is between discrete actions and continuous actions. When we refer to discrete actions, we simply mean that there is a finite set of possible actions an agent can take. For example, in pong an agent can decide to move up or down. On the other hand, continuous actions have an infinite number of possibilities. An example of a continuous action, although kind of silly, is the hiding position of an agent if it is playing hide and seek.

Given enough time, the agent can theoretically hide anywhere - so the action space is unbounded. In contrast, we can have a continuous action space that is bounded. An example close to my heart is position sizing when trading a financial asset. The bounds are -1 (100% Short) and 1 (100% Long). To map states to that bounded action space, we can use $$\tanh$$ in the final layer of a neural network. That seems pretty easy... so why am I writing a blog post about it? Often times we need more than just a deterministic output, especially when the underlying data has a low signal-to-noise ratio. The additional piece of information that we need is *uncertainty* in our agent's decision. We will use a Bayesian approach to model a posterior distribution and sample from this distribution to estimate the uncertainty. Don't worry if that doesn't completely make sense yet - it will by the end of this post!

## Probability Distributions

For a great introduction to Bayesian statistics I suggest reading [Will Kurt's blog](https://www.countbayesie.com) - Count Bayesie. It's awesome.

Distributions can be thought of as representing beliefs about the world. Specifically as it relates to our task at hand, the probability distributions represent our beliefs in how good an action is, given the state. In the financial markets context, where the action space is continuous and bounded between -1 and 1, a mean close to 1 represents a belief that it is a good time to buy that asset, so we should long it. A mean close to -1 represents the opposite, so we should short the asset. Building on this example, if the standard deviation in our distribution is large (small) then our agent is uncertain (certain) in its decision. In other words, if the agent's policy has a large standard deviation, then it has not developed a strong belief yet.

Whenever you hear anyone talking about Bayesian statistics, you always hear "prior" and "posterior". Simply put, a prior is your belief about the world *before* receiving new information. However, once you receive new information, then you update your prior distribution to form a posterior distribution. After that, if you receive more information, then your posterior becomes your prior, and the new information gets incorporated to form a new posterior distribution. Essentially, there is this feedback loop of continual learning that happens as more and more new information gets processed by your agent.

Our goal is to learn a good posterior distribution on actions, conditioned on the state that the agent is in. If you are familiar with [this paper](https://arxiv.org/pdf/1506.02142.pdf), then you might be thinking that we can just use Monte Carlo (MC) dropout with a $$\tanh$$ output layer. For those who are not familiar with this concept, let me explain. Dropout is a technique that was originally used for neural network regularization. With each pass, it will randomly "drop" neurons from each hidden layer by setting their output to 0. This reduces the output's dependency on any one particular neuron, which should help generalization. However, researchers at Cambridge found that using dropout during inference can be used to approximate a posterior distribution. This is because each time you pass inputs through the network, a different set of neurons will be dropped, so the output is going to be different for each run - creating a distribution of outputs.

The great thing about this architecture is that you can easily pass gradients through the policy network. The loss function that we are minimizing throughout this blog is $$\mathcal{L} = - r \times \pi(s)$$, where $$r$$ denotes the reward and $$\pi(s)$$ denotes the policy output given the states (i.e. the action). We wanted to demonstrate how the distribution changes in a controlled environment. So we use the same state input throughout all our experiments and continually feed it a positive reward to view the changes during training. Below is the first example using the MC Dropout method and a $$\tanh$$ output layer.

![Alt Text](/images/MC_dropout_posterior.gif)

I omitted a kernel density estimation (KDE) plot on top of the histogram because as training progressed, the KDE became much more jagged and not representative of the actual probability density function (PDF). I was using `sns.kdeplot`, if anyone knows how to fix this, please let me know in the comments section!

There are two things that I don't particularly like about this approach. The first is that it is possible to have multiple peaks in the distribution, as seen when the neural network is first initialized. I realize that as training went on, only one peak emerged. However, the fact that an agent can potentially learn such a distribution (with multiple peaks) makes me uncomfortable. If we go back to our example in the financial markets, an action of -1 will have the exact opposite reward of 1 (because it is the other side of the trade), so having peaks at both ends of the spectrum is quite confusing. I would much rather just have one peak near 0 with a large standard deviation if the agent is uncertain which action to take. The second is that it becomes overly optimistic in its decision when compared to a gaussian output (demonstrated below), which could possibly indicate that it is understating the uncertainty.

Great, so that's the solution - let's use a normal distribution in the output!

## Reparameterization Trick

You might be thinking that it doesn't make sense to have a normal distribution as the output since you can't take the derivative of a random variable. You are correct, but we can apply the reparameterization trick to move the random variable outside of the neural network. If our neural network parameters are denoted by $$\theta$$, then we can define $$\mu_{\theta}$$ and $$\sigma_{\theta}$$ as outputs of the neural network, such that:

$$\pi \sim \mathcal{N}(\mu_{\theta}(s), \sigma_{\theta}(s))$$

However, if we want to use backpropagation, we have to make the whole computational graph differentiable. We can do this by defining $$\epsilon$$, which does not depend on $$\theta$$. So now, gradients can flow through the entire graph without passing through the random variable:

$$\epsilon \sim \mathcal{N}(0,I)$$
$$\pi = \mu_{\theta} + \sigma_{\theta} \dot \epsilon$$

Python code to take the random variable outside of the computation graph is shown below:


## Novel Solution

Present our novel solution to the problem. We will show empirically and prove mathematically that our approach is superior to the bayesian approximation.

Since we actually know the mean and standard deviation of the distribution, we can plot both the histogram and the PDF as follows:

```python
  import matplotlib.pyplot as plt
  import scipy.stats as stats

  pdf_probs = stats.truncnorm.pdf(bayesian_policy, lower_bound, upper_bound, policy_mean, policy_std)

  plt.hist(bayesian_policy, bins = 50, normed = True, alpha = 0.3, label = "Histogram")
  plt.plot(bayesian_policy[bayesian_policy.argsort()], pdf_probs[bayesian_policy.argsort()], linewidth = 2.3, label = "PDF Curve")
  plt.xlim(-1,1)
  plt.legend(loc = 2)

  plt.show()
```

![Alt Text](/images/clipped_posterior.gif)

* Explain the challenge in clipping samples outside the bounds
* Walk through reparameterization for the unbounded case
* Expand it for our solution

![Alt Text](/images/posterior.gif)


## Concluding Remarks

Some conclusions


{% include disqus.html %}
