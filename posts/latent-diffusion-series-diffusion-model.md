<!--
.. title: Latent Diffusion Series: Latent Diffusion Model
.. slug: latent-diffusion-series-diffusion-model
.. date: 2023-11-29 15:10:30 UTC+02:00
.. previewimage: /files/mnist_autoencoder_bg.png
.. has_math: true
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text
-->

In the Latent Diffusion Series of blog posts, I'm going through all components needed to train a latent diffusion model to generate random digits from the MNIST dataset. In the third, and last, post, we will finally build and train a latent diffusion model which will be trained to generate random MNIST digits.<!-- TEASER_END --> For the other posts, please look below:

1. [MNIST Classifier](/posts/latent-diffusion-series-mnist-classifier)
2. [Variational Autoencoder (VAE)](/posts/latent-diffusion-series-variational-autoencoder)
3. **Latent Diffusion Model**

The links will become active as soon as they the posts are completed. Even though this blog post is part of a series, I will try my best to write it in such a way that it's not required to have read the previous blog posts.

In this post I will discuss [Diffusion Models](https://en.wikipedia.org/wiki/Diffusion_model) and more specifically, Latent Diffusion Models, which are trained to denoise latent representations. This enables us to generate anything we have a large dataset of by sampling noise from a normal distributions and denoising it using a diffusion model trained for this task. Here we will build a latent diffusion model using components from the previous two posts and train it on the MNIST dataset, and compare the results we got when we generated digits using a VAE. If you'd like a bit more about the MNIST dataset or VAEs, please look at the previous blog posts linked above. I have created a Python [notebook on Colab](https://colab.research.google.com/drive/18UReo17EOUNYoEqVwazgS-JgXjbgOKmr), which you can use to follow along and experiment with this post's code.

# Diffusion Model

Diffusion probabilistic models are a class of models whose job is to remove noise of a known distribution from data – At least, that's one way of looking at diffusion models, and if you'd like to learn more about the many sides of Diffusion models, have a look at [this article](https://yang-song.net/blog/2021/score/) by Yang Song, and [this one](https://sander.ai/2023/07/20/perspectives.html) by Sander Dieleman, as well as [this](http://arxiv.org/abs/2208.11970) paper by Calvin Luo. – Diffusion models were first developed as [generative models](https://en.wikipedia.org/wiki/Generative_model) for generating samples that follows the original dataset distribution. The goal of a generative models is to learn to model the true data distribution from observed samples, so that generating new samples is as simple as sampling the learned distribution. Diffusion models achieve this by corrupting the dataset with progressively larger amounts of noise and training a set of probabilistic models to reverse corruption step. This reverse problem is made tractable by using knowledge of the functional form of the reverse distributions.
<img src="/files/mnist_corruption.png" style="display:block; padding:1em 0;"/>
Above you see an example of images from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) corrupted with increasing amounts of noise.

<!--
TODO:
* Perhaps do a recap about VAEs and what their weaknesses are.
* Overview of what diffusion models are and how they can overcome these weaknesses?
* Examples of diffusion models, and how succesful they are.
* Perhaps discuss the connection with score-based models briefly?
* Go over the mathematics involved.
* Build the basic functions needed to do the diffusion, and show examples of how images are diffused.
* Latent variables need not be different than regular images, as long as normally distributed?
* Go through training of the NN. Show results, compare with VAE.
* Conclusions.
-->

# Conclusion

In this blog post, we set out to learn more about a variational autoencoders (VAEs). This, because it's a base component of what makes latent diffusion models tick. To get a better understanding of the latent space, we created a latent space for the MNIST dataset using a simple autoencoder. Even though we found out that autoencoders are lousy at MNIST digit generation, it gave us insights into how we would like to normalize the latent space. VAEs assume the latent variables follow a standard normal probability distribution, and using some math and some additional assumptions, we get a recipe for training and sampling a VAE and the decoder and encoder can be seen as the posterior and likelihood of the latent given a MNIST sample. The digits generated with the VAE are very much improved compared to the simple autoencoder, but they are not perfect. Let's see what we can do with the latent diffusion model in the next post. For now, I invite you to play around with the [notebook](https://colab.research.google.com/drive/18UReo17EOUNYoEqVwazgS-JgXjbgOKmr) for this post.

