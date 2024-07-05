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

I briefly introduced diffusion models in the [first post](/posts/latent-diffusion-series-mnist-classifier), but here I'll give a more extensive overview. As I mentioned there, diffusion models were first developed as [generative models](https://en.wikipedia.org/wiki/Generative_model) for generating samples that follows the original dataset distribution. The goal of a generative models is to learn to model the true data distribution from observed samples, so that generating new samples is as simple as sampling the learned distribution. Diffusion models achieve this by corrupting the dataset with progressively larger amounts of noise, leading to samples with pure noise, and training a set of probabilistic models to reverse the corruption step in the probabilistic sense. This reverse problem is made tractable by using knowledge of the functional form of the reverse distributions.
<img src="/files/mnist_corruption.png" style="display:block; padding:1em 0;"/>

When I first researched diffusion models, I started with the original paper by Sohl-Dicksten *et al*, 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics'[^1], which as the name suggests, is inspired by non-equilibrium statistical physics. I spent some time with this paper, but I quickly realised that it's not the best paper to read as an introduction to diffusion models. It doesn't make it easy to understand how to actually build and train a diffusion model. Fortunately, diffusion models are old enough by now that later papers have made various simplifications and have managed to give easier to understand explanations. On the other hand, because diffusion models have been derived in a number of different ways; from stochastic differential equations to score-based models, it can make it frustrating to understand the relationship between the different derivations, and can lead to mixing up concepts. While researching for the easiest way to explain diffusion models, I stumbled upon a paper, 'Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model' [^3], I thought that this might be the best candidate, but something bothered me about it; I was missing the motivation behind the derivation. In the end I found this motivation in stochastic interpolants [^4] which describes the diffusion process as a way of efficiently moving through the flow field defined by probability distribution of the intermediate steps. Stochastic interpolants, as you might have guessed, is not easy to understand, especially if you are not familiar with certain topics in mathematics such as [measure theory](https://en.wikipedia.org/wiki/Measure_(mathematics)). Thus, I decided that in this blog post I would concentrate on briefly explaining the paper by Ho *et al* titled 'Denoising Diffusion Probabilistic Models' (DDPM), which is also referenced in the seminal LDM paper [^5], and leave the above discussion open for a bonus blog post in the near future.

If you are interested to learn more about diffusion methods and alternative approaches, I suggest you start by the amazing [blog post by Miika Aittala](https://developer.nvidia.com/blog/generative-ai-research-spotlight-demystifying-diffusion-based-models/), and then check the posts by [Yang Song](https://yang-song.net/blog/2021/score/), [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), and [Sander Dieleman](https://sander.ai/2023/07/20/perspectives.html).

<!-- theory starts from here -->
As I mentioned, the diffusion process consists of a number of steps, N, which gradually add noise:
\\[
\mathbf{x}\_{t+1} = \theta\_{t+1} \mathbf{x}\_t + \phi\_{t+1} \mathbf{\epsilon}, \quad \mathbf{\epsilon} \in \mathcal{N}(0,1)
\tag{1}
\\]
where \\( \theta\_t, \phi\_t \\), are appropriately chosen multipliers whose main function is to make sure that the magnitude of \\( x\_t \\) does not get too big from adding noise as \\(t \rightarrow T\\).
We see that if we replace \\( x\_t \\) in the above equation with the expression we get from the same formula, we find,
\\[
\mathbf{x}\_{t+1} = \theta\_{t+1} \theta\_t \mathbf{x}\_{t-1} + (\theta\_{t+1} \phi\_t + \phi\_{t+1}) \mathbf{\epsilon}
\\]
which, if we recursively replace the expressions for \\(x\\), we finally get an expression with respect to \\(x_0\\),
\\[
\mathbf{x}\_t = \bar{\theta}\_t \mathbf{x}\_0 + \bar{\phi}\_t \mathbf{\epsilon}
\tag{2}
\\]
where we substituted \\(t + 1\\) with \\(t\\), and set
\\[
\begin{split}
\bar{\theta}\_t &= \prod\_{i=1}^t \theta\_i\\\\
\bar{\phi}\_t &= \sum\_{i=1}^t (\bar{\theta}\_t / \bar{\theta}\_i) \phi\_i
\end{split}
\\]
also note that \\(\epsilon\\) represents a random sample from the Gaussian distribution, which is different for each step \\(x_t \rightarrow x_{t+1}\\), but because the samples are i.i.d., we can factor them out.

The diffusion process can be seen as interpolating between the data and noise, and choosing \\( \theta\_t = 1 - \phi\_t \\) makes sense in this case. In literature [^6], \\( \phi\_t, \theta\_t \\) are often set to \\( \phi\_t = \sqrt{\beta\_t} \\) and \\( \theta\_t = \sqrt{1 - \beta\_t} \\). This latter choice is connected to the variance of the noise we want to add, and as such, \\( \beta\_t \\) is also often referred to as the 'variance schedule'. This can also be seen from the fact that equation (1) above is basically a scaled and translated standard Gaussian, which gives a normal distribution, \\( \mathcal{N}(\theta\_{t+1} \mathbf{x}\_t, \phi\_{t+1}) \\) with mean \\(\theta\_{t+1} \mathbf{x}\_t \\) and variance \\( \phi\_{t+1} \\).

Now, say we know \\(\mathbf{x}\_{t+1}\\) in equation (1), above, then we can train a neural network to either predict \\(\mathbf{x}\_t\\) directly which would cause the model to learn the distribution of \\(\mathbf{x}\_t\\) given \\(\mathbf{x}\_{t+1}\\), or we can train it to learn the noise sample \\(\mathbf{\epsilon}\\). The last approach is the one followed by most papers such as Ho *et al* [^8], but a few more could be considered; one possibility could be to predict \\(\mathbf{x}\_0\\) itself from \\(\mathbf{x}\_{t+1}\\), but that would be to hard for the model to learn, so instead you could then use equation (2) again to get a new \\(\mathbf{x}\_t\\). 

<!-- code starts from here -->
Let's now jump right into it and implement the training of a (denoising) neural network that predicts the noise sample \\(\epsilon\\), and then, using equation (1), iteratively calculate \\(\mathbf{x}\_t\\) until we reach \\(\mathbf{x}\_0\\). Let's first start by implementing equation (2) in the forward direction, i.e., given sample \\(\mathbf{x}\_0\\), produce a noisy result \\(\mathbf{x}\_t\\). As you might imagine, the code is really simple, so let's sample a few digits from the MNIST validation set and visualize the diffusion process, reproducing the figure above:
```python
time_steps = 1000
fig_shape = np.array([4, 8]) # rows, columns
fig_size  = tuple(fig_shape[::-1] * 0.75)

# Setup fig_shape[1] uniformly spaced time steps. We clamp the last
# step to time_steps - 1.
t = torch.arange(fig_shape[1]) / (fig_shape[1] - 1) * time_steps
t = t.int()
t[-1] -= 1
t = t.repeat(fig_shape[0],1).T.reshape(-1)

# Use a simple linear schedule and set theta = 1 - phi
phi_t = torch.linspace(0.0, 1.0, time_steps)
phi_t = phi_t[t].view(-1,1,1,1)
theta_t = 1.0 - phi_t

# Sample fig_shape[0] digits from the validation set.
perm = np.random.choice(
    np.arange(data_val.shape[0]), fig_shape[0],
    replace=False
)
x_0 = torch.from_numpy(data_val[perm,...])
x_0 = x_0.repeat(fig_shape[1],1,1,1)

# Sample the noise for the diffusion process
N = torch.normal(0, 1, size=x_0.shape)

# Forward diffusion
x_t = theta_t * x_0 + phi_t * N

# Plot samples
with plt.style.context(pyplot_context):
    fig, axes = plt.subplots(
        fig_shape[0], fig_shape[1], figsize=fig_size
    )
    for i in range(fig_shape[0] * fig_shape[1]):
        img = np.clip(0.5 * (x_t[i,:,:,0] + 1.0), 0.0, 1.0)
        ai, aj = (i  % fig_shape[0], i // fig_shape[0])
        axes[ai,aj].imshow(img, cmap='gray')
        axes[ai,aj].axis('off')
    plt.show()
```
Here I chose a simple linear schedule, with \\(\bar{\theta}\_t = 1 - \bar{\phi}\_t\\).
<!-- We can try a few different schedules later? -->

Before setting up the latent diffusion model for training, we need to calculate the latent variables for our dataset. We will do this using the VAE we trained in the last blog post. The following code shows how to calculate the latent variables for the training set using the encoder part of the VAE we trained in the previous blog post.
``` python
num_batches = int(math.ceil(x_train.shape[0] / batch_size))
lat_train = []
with torch.no_grad():
    for bid in range(num_batches):
        x = x_train[bid*batch_size:(bid+1)*batch_size,...]
        x = x.to(device)
        x = torch.transpose(x, 1, 3)

        mu_sigma = encoder(0.5 * (x + 1.0))
        mu    = mu_sigma[:,:latent_channels,:,:]
        sigma = mu_sigma[:,latent_channels:,:,:]
        z = torch.normal(0, 1, size=mu.shape, device=device)
        z = z * sigma + mu
        lat_train.append(z.cpu().numpy())

lat_train = np.concatenate(lat_train, axis=0)
```
If you haven't yet, please check that post first, although in the [notebook](todo) I shared, I give you the option to download a pre-trained model.

With the dataset prepared, we now turn our attention to the model itself. We will use a Unet, similar to the autoencoder or VAE, but we will now also pass the time variable, \\(t\\) along. To propagate the time information through the network, it was found that using a so called sinusoidal positional embedding can be effective.
<!-- @todo: Formulae for sinusoidal positional embedding? Also short
explanation about intuition? I actually don't understand why it's used here
since we don't need any relative positions. -->
``` python
class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim=16, scale=1000):
        super().__init__()
        half_dim = dim // 2
        emb_scale = math.log(scale) / (half_dim - 1)
        self.emb_factor = torch.exp(
            -emb_scale * torch.arange(half_dim, device=device)
        )

    def forward(self, time):
        embeddings = time[:, None] * self.emb_factor
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
```
The UNet we will be using here is very similar to the one we used before, only we pass the time embedding into each convolutional block, which is then added to the output of the convolution.
``` python
class TimeEmbeddingConvBlock(torch.nn.Module):
    def __init__(self, fin, fout, tin, *args, **kwargs):
        super(UnetConvBlock, self).__init__()
        self._conv = torch.nn.Conv2d(fin, fout, *args, **kwargs)
        self._norm = torch.nn.InstanceNorm2d(fout)
        self._relu = torch.nn.LeakyReLU()
        self._emb_linear = torch.nn.Linear(tin, fout)

    def forward(self, x, t_emb):
        t_emb = self._emb_linear(self._relu(t_emb))
        x = self._conv(x)
        return self._relu(self._norm(x) + t_emb[:,:,None,None])
```

<!--
@important:
Final decision. Just introduce DDPM with little or no derivation for now and
make a separate blog post explaining iterative alpha-(de)blending, and how it
connects with stochastic interpolants or flow matching. Alpha-(de)blending
gives very nice results though! I think since the latent diffusion model used
DDPM we should stick with that, but I think introducing the alpha-(de)blending
is really interesting. Not only can we do fun things like moving along the path
based on the blending/deblending, but also make a connection to flow matching
and stochastic interpolants.
-->


# Conclusion

In this blog post, we set out to learn more about a variational autoencoders (VAEs). This, because it's a base component of what makes latent diffusion models tick. To get a better understanding of the latent space, we created a latent space for the MNIST dataset using a simple autoencoder. Even though we found out that autoencoders are lousy at MNIST digit generation, it gave us insights into how we would like to normalize the latent space. VAEs assume the latent variables follow a standard normal probability distribution, and using some math and some additional assumptions, we get a recipe for training and sampling a VAE and the decoder and encoder can be seen as the posterior and likelihood of the latent given a MNIST sample. The digits generated with the VAE are very much improved compared to the simple autoencoder, but they are not perfect. Let's see what we can do with the latent diffusion model in the next post. For now, I invite you to play around with the [notebook](https://colab.research.google.com/drive/18UReo17EOUNYoEqVwazgS-JgXjbgOKmr) for this post.

[^1]: [J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli -- Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585)
[^2]: [C. Luo -- Understanding Diffusion Models: A Unified Perspective](http://arxiv.org/abs/2208.11970)
[^3]: [E. Heitz, L. Belcour, T. Chambon -- Iterative α-(de)Blending: a Minimalist Deterministic Diffusion Model](https://arxiv.org/abs/2305.03486)
[^4]: [M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden -- Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden](https://arxiv.org/abs/2303.08797)
[^5]: [R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer -- High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
[^6]: [Y. Song, J. K. Sohl-Dickstein, P. D. Kingma, A. Kumar, S. Ermon, B. Poole -- Score-Based Generative Modeling through Stochastic Differential Equations](http://arxiv.org/abs/2011.13456)
[^7]: [A. Nichol, P. Dhariwal -- Improved Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2102.09672)
[^8]: [J. Ho, A. Jain, P. Abbeel -- Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
[^9]: [T. Karras, M. Aittala, T. Aila, S. Laine -- Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
