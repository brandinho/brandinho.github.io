---
title: "The Math of Loss Functions"
date: 2019-02-06
tags: [Gradient Descent]
header:
  image: "/images/connected-nodes-banner.jpg"
excerpt: "Gradient Descent"
mathjax: "true"
---

## Overview

In this post we will go over some of the math associated with popular supervised learning loss functions. Specifically, we are going to focus on linear, logistic, and softmax regression. We show that the derivatives used for parameter updates are the same for all of those models! Most people probably won’t care because they use automatic differentiation libraries like TensorFlow, but I find it cool.

Each section in this blog is going to start out with a few lines of math that explain how the model works. Then we are going to dive into the derivative of the loss function with respect to $$z$$. I go into a lot of detail when calculating the derivatives - probably more than necessary. I do this because I want everybody to completely understand how the math works.

We will layout the math for each of the models in the following way:

* Define a linear equation which will be denoted $$z$$
* Define an activation function if there is any
* Define a prediction (transform of the linear equation) which will be denoted $$\hat{y}$$
* Define a loss function which will be denoted $$\mathcal{L}$$

I am laying it out this way to maintain consistency between each of the models. Keep this in mind when you realize how silly it is to move from $$z$$ to $$\hat{y}$$ for linear regression.

Before diving into the math it is important to note that I shape the input $$X$$ as $$(\text{# instances}, \text{# features})$$, and the weight matrix $$w$$ as $$(\text{# features}, \text{# outputs})$$. This is an important distinction because the equations would look slightly different if you used $$X^\boldsymbol{\top}$$, which a lot of people use for some reason. I personally dislike using rows for features and columns for instances, but if it floats your boat then go for it (you'll just need to make minor changes to the math notation).

## Linear Regression

To start, let’s define our core functions for linear regression:

$$
\begin{align*}  
  &\text{Linear Equation}: &&z = Xw + b \\[1.5ex]
  &\text{Activation Function}: &&\text{None} \\[1.5ex]
  &\text{Prediction}: &&\hat{y} = z \\[0.5ex]
  &\text{Loss Function}: &&\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2
\end{align*}
$$

We can also define the functions in python code:

```python
  weights = np.random.normal(size = n_features).reshape(n_features, 1)
  bias = 0

  def linear_regression_inference(inputs):
      return np.matmul(inputs, weights) + bias   

  def calculate_error(x, y):
      ### Mean Squared Error (I know I'm not taking an average, but you get the point)
      y_hat = linear_regression_inference(x)
      return 0.5 * (yhat - y)**2      
```

We are interested in calculating the derivative of the loss with respect to $$z$$. Throughout this post, we will do this by applying the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z}$$

First we will calculate the partial derivative of the loss with respect to our prediction:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y$$

Next, although silly, we calculate the partial derivative of our prediction with respect to the linear equation. Of course since the linear equation is our prediction (since we're doing linear regression), the partial derivative is just 1:

$$\frac{\partial \hat{y}}{\partial z} = 1$$

When we combine them together, the derivative of the loss with respect to the linear equation is:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

Although this was pretty straight forward, the next two sections are a bit more involved, so buckle up. Get ready to have your mind blown as you learn that $$\frac{\partial \mathcal{L}}{\partial z} = (\hat{y} - y)$$ for logistic regression and softmax regression as well!

## Logistic Regression

Like linear regression, we will define the core functions for logistic regression:

$$
\begin{align*}  
  &\text{Linear Equation}: &&z = Xw + b \\[0.5ex]
  &\text{Activation Function}: &&\sigma(z) = \frac{1}{1 + e^{-z}} \\[0.5ex]
  &\text{Prediction}: &&\hat{y} = \sigma(z) \\[1.5ex]
  &\text{Loss Function}: &&\mathcal{L} = -(y\log\hat{y} + (1-y)\log(1-\hat{y}))
\end{align*}
$$

We can also define the functions in python code:

```python
  weights = np.random.normal(size = n_features).reshape(n_features, 1)
  bias = 0

  def sigmoid(x):
      return 1 / (1 + np.exp(-x))

  def logistic_regression_inference(x):
      return sigmoid(np.matmul(x, weights) + bias)

  def calculate_error(x, y):
      ### Binary Cross-Entropy
      y_hat = logistic_regression_inference(x)
      return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
```

**NOTE**: When calculating the error for logistic regression we usually add a small constant inside the $$\log$$ calculation to prevent taking the log of 0.

Again, we use the chain rule to calculate the partial derivative of interest:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z}$$

The partial derivative of the loss with respect to our prediction is pretty simple to calculate:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

Next we will calculate the derivative of our prediction with respect to the linear equation. We can use a little algebra to move things around and get a nice expression for the derivative:

$$
\begin{align*}
	\frac{\partial \hat{y}}{\partial z} &= \frac{\partial}{\partial z}\left[\frac{1}{1 + e^{-z}}\right] \\[0.75ex]
    &= \frac{e^{-z}}{(1 + e^{-z})^2} \\[0.75ex]
    &= \frac{1 + e^{-z} - 1}{(1 + e^{-z})^2} \\[0.75ex]
    &= \frac{1 + e^{-z}}{(1 + e^{-z})^2} - \frac{1}{(1 + e^{-z})^2} \\[0.75ex]
    &= \frac{1}{1 + e^{-z}} - \frac{1}{(1 + e^{-z})^2} \\[0.75ex]
    &= \frac{1}{1 + e^{-z}} \left(1 - \frac{1}{1 + e^{-z}}\right) \\[0.75ex]
    &= \hat{y}(1 - \hat{y})
\end{align*}
$$

Isn't that awesome?! Anyways, enough of my love for math, let's move on. Now we'll combine the two partial derivatives to get our final expression for the derivative of the loss with respect to the linear equation.

$$
\begin{align*}
  \frac{\partial \mathcal{L}}{\partial z} &= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right)\hat{y}(1 - \hat{y}) \\[0.75ex]
  &= -\frac{y}{\hat{y}}\hat{y}(1 - \hat{y}) + \frac{1-y}{1-\hat{y}}\hat{y}(1 - \hat{y}) \\[0.75ex]
  &= -y(1 - \hat{y}) + (1-y)\hat{y} \\[0.75ex]
  &= -y + y\hat{y} + \hat{y} - y\hat{y} \\[0.75ex]
  &= \hat{y} - y
\end{align*}
$$

Would you look at that, it's the exact same!! If you think that is cool (which you should), then just wait for the next section where we go through softmax regression.

## Softmax Regression

Once again, we will define the core functions for softmax regression:

$$
\begin{align*}  
  &\text{Linear Equation}: &&z = Xw + b \\[0.5ex]
  &\text{Activation Function}: &&\varphi(z_i) = \frac{e^{z_i}}{\sum_n e^{z_n}} \\[0.5ex]
  &\text{Prediction}: &&\hat{y_i} = \varphi(z_i) \\[1.5ex]
  &\text{Loss Function}: &&\mathcal{L} = -\sum_i y_i\log\hat{y_i}
\end{align*}
$$

We can also define the functions in python code:

```python
  weights = np.random.normal(size = n_features).reshape(n_features, 1)
  bias = 0

  def softmax(x):
      return np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

  def softmax_regression_inference(x):
      return softmax(np.matmul(x, self.weights) + self.bias)     

  def calculate_error(x, y):
      ### Categorical Cross-Entropy
      y_hat = softmax_regression_inference(x)
      return -np.mean(np.sum(y * np.log(yhat), axis = 1))     
```

**NOTE**: With softmax regression, we also typically add a small constant inside $$\log$$ for the same reason as logistic regression.

For the last time, we will restate the partial derivative using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \frac{\partial \mathcal{L}}{\partial \hat{y_i}} \frac{\partial \hat{y_i}}{\partial z_j}$$

Let's calculate the first partial derivative of the loss with respect to our prediction:

$$\frac{\partial \mathcal{L}}{\partial \hat{y_i}} = -\sum_i \frac{y_i}{\hat{y_i}}$$

That was pretty easy! Now let's tackle the monster... the partial derivative of our prediction with respect to the linear equation:

$$\frac{\partial \hat{y_i}}{\partial z_j} = \frac{\sum_n e^{z_n} \frac{\partial}{\partial z_j}[e^{z_i}] - e^{z_i} \frac{\partial}{\partial z_j}\left[\sum_n e^{z_n}\right]}{\left(\sum_n e^{z_n}\right)^2}$$

It is important to realize that we need to break this down into two parts. The first is when $$i = j$$ and the second is when $$i \neq j$$.

if $$i = j$$:

$$
\begin{align*}
  \frac{\partial \hat{y_i}}{\partial z_j} &= \frac{e^{z_j}\sum_n e^{z_n} - e^{z_i}e^{z_j}}{\left(\sum_n e^{z_n}\right)^2} \\[0.75ex]
  &= \frac{e^{z_i}\sum_n e^{z_n}}{\left(\sum_n e^{z_n}\right)^2} - \frac{e^{z_i}e^{z_j}}{\left(\sum_n e^{z_n}\right)^2} \\[0.75ex]
  &= \frac{e^{z_i}}{\sum_n e^{z_n}} - \frac{e^{z_i}e^{z_j}}{\left(\sum_n e^{z_n}\right)^2} \\[0.75ex]
  &= \frac{e^{z_i}}{\sum_n e^{z_n}} - \frac{e^{z_i}}{\sum_n e^{z_n}} \frac{e^{z_j}}{\sum_n e^{z_n}} \\[0.75ex]
  &= \frac{e^{z_i}}{\sum_n e^{z_n}} \left(1 - \frac{e^{z_j}}{\sum_n e^{z_n}}\right) \\[0.75ex]
  &= \hat{y_i}(1 - \hat{y_j})
\end{align*}
$$


if $$i \neq j$$:

$$
\begin{align*}
  \frac{\partial \hat{y_i}}{\partial z_j} &= \frac{0 - e^{z_i}e^{z_j}}{\left(\sum_n e^{z_n}\right)^2} \\[0.75ex]
  &= - \frac{e^{z_i}}{\sum_n e^{z_n}} \frac{e^{z_j}}{\sum_n e^{z_n}} \\[0.75ex]
  &= - \hat{y_i}\hat{y_j}
\end{align*}
$$

We can therefore combine them as follows:

$$\frac{\partial \mathcal{L}}{\partial z_j} = - \hat{y_i}(1 - \hat{y_j})\frac{y_i}{\hat{y_i}} - \sum_{i \neq j} \frac{y_i}{\hat{y_i}}(-\hat{y}_i\hat{y_j})$$

The left side of the equation is where $$i = j$$, while the right side is where $$i \neq j$$. You will notice that we can cancel out a few terms, so the equation now becomes:

$$\frac{\partial \mathcal{L}}{\partial z_j} = - y_i(1 - \hat{y_j}) + \sum_{i \neq j} y_i\hat{y_j}$$

These next few steps trip some people out, so pay close attention. The first thing we're going to do is change the subscript on the left side from $$y_i$$ to $$y_j$$ since $$i = j$$ for that part of the equation:

$$\frac{\partial \mathcal{L}}{\partial z_j} = - y_j(1 - \hat{y_j}) + \sum_{i \neq j} y_i\hat{y_j}$$

Next, we are going to multiply out the left side of the equation to get:

$$\frac{\partial \mathcal{L}}{\partial z_j} = - y_j + y_j\hat{y_j} + \sum_{i \neq j} y_i\hat{y_j}$$

We will then factor out $$\hat{y_j}$$ to get:

$$\frac{\partial \mathcal{L}}{\partial z_j} = - y_j + \hat{y_j}\left(y_j + \sum_{i \neq j} y_i\right)$$

This is where the magic happens. We realize that inside the bracket $$y_j$$ can become $$y_i$$ since it is from the left side of the equation. Since $$y$$ is a one-hot encoded vector:

$$y_j + \sum_{i \neq j} y_i = 1$$

So our final partial derivative equals:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \hat{y_j} - y_j = \hat{y} - y$$

## Partial Derivative to Update Parameters

As you may have noticed, the equation for $$z$$ is the same for all of the models mentioned above. This means that the derivative for the parameter updates will also be the exact same, since the only other step is to chain together $$\frac{\partial \mathcal{L}}{\partial z}$$ and $$\frac{\partial z}{\partial w}$$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial w}$$

Since $$\frac{\partial z}{\partial w} = X$$, to get the partial derivative of the loss with respect to the weights, we simply take the dot product between the transpose of the input and $$(\hat{y} - y)$$. We transpose $$X$$ to make the shapes line up nicely for matrix multiplication. Thus, we get:

$$\frac{\partial \mathcal{L}}{\partial w} = X^\boldsymbol{\top}(\hat{y} - y)$$

## Concluding Remarks

After reading this post it might be temping to say that you can use Mean Squared Error (MSE) for logistic regression since the derivatives for linear and logistic regression are the same. However, this is incorrect. It is important to realize that derivative only works out to be the same because there is no activation function for linear regression. If you now have a sigmoid activation function in the output, then $$\frac{\partial \mathcal{L}}{\partial z} \neq (\hat{y} - y)$$ for $$\mathcal{L}_{MSE}$$.

I hope you enjoyed learning about the math behind some supervised learning loss functions! In the future I might make another blog post about loss functions, except with less math and more visuals.
