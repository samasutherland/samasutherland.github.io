---
layout: post
title: "Latent Attention"
date: 2026-04-25
categories: [lLMs]
tags: []
---

# Latent Attention

One of the key innovations DeepSeek made when they disrupted the LLM space was the introduction of latent attention. 
Latent attention decreases the parameter overhead of the attention mechanism by splitting it into two parts.
The first part takes the embedding vector and reduces it to a vector of lower dimension via a rectangular matrix.
The second part then reprojects this intermediate latent vector into the K and V vectors needed for attention.
This process reduces the parameter count of attention as a single matrix of shape \[embedding_dim, KV_dim\] uses significantly more parameters than two matrices of \[embedding_dim, projection_dim\] and \[projection_dim, KV_dim\] for a sufficiently small projection_dim.
Not only that, but DeepSeek actually shares the first half of the layer over all K and V vectors.
This both reduces the parameter count further, and allows the compressed latent KV vectors to be cached with reduced memory cost, at the cost of reprojecting them each time.
The latent intermediate vector design exploits the low-rank nature of attention.
Since each attention head focusses on a subset of the properties of each embedded token, these properties likely have a lower effective dimension than the full embedding space.
Explicitly building this assumption into the model results in a decomposition similar to a Singular Value Decomposition (SVD), where a matrix is decomposed into two orthogonal matrices and a diagonal matrix containing the singular values, revealing the rank structure of the original matrix.
By absorbing the singular values into either of the orthogonal matrices, a split two-step approach pops out, exactly as previously described.

While I think that the primary concern for the DeepSeek researchers was the inference efficiency and therefore the effectiveness of the KV caching, for a fixed training compute budget I am interested to find out if the increased parameter efficiency results in models that perform better than full-rank attention.
My prediction is that BabyLM will benefit from latent attention, as it will allow an increase in the depth of the model for the same parameter count, but simplestories won't benefit too much due to its preference for width over depth.
## Implementation
The implementation of latent attention is very simple. In my repository, I simply override the KV transformation with the latent version:
```python
self.kv = nn.Sequential(nn.Linear(embedding_dim, projection_dim), 
                        nn.Linear(projection_dim, (self.qk_dim + v_dim) * n_heads))
```
It seems that the standard implementation uses a latent representation for K and V, but not Q.
This is likely due to the main intention being for caching efficiency, however also likely stems from the degree of precision needed to adequately describe the query vs the target.
For each attention call, each query is compared to *many* keys and values, so precision of the query is likely more important than that of the keys and values.

## Projection dim sweep
I decided to sweep the projection dimension on the BabyLM dataset. The embedding dimension is 256, the qk dimension is 64, with 4 heads (so full dimension also 256).
A full-rank matrix would use 256 * 256 = 65536 parameters, and a projection dim that matches this would use 2 * 256 * p_dim = 65536 -> p_dim = 128, so the projection dim has to be smaller than that.
The sweep tests values 32, 64, 96, and 128. Too small, and the latent representation is not large enough to capture necessary features, degrading performance.
Too large, and the reduced parameter count is outweighed by the additional matrix multiplications.

The results of the sweep are in this graph: ![projection_sweep](/assets/img/lLM/projected_dimension.png)
Unfortunately, while a projected dimension of 64 appears to be the sweet spot for performance, none were able to outperform the full-rank baseline. 
All models processed about the same number of tokens/parameter, and they all had the same number of layers, meaning that any reduction in parameter count was offset by the increase in computation.
It seems that the main benefit of the projection is the KV caching after all, unless the balance changes significantly with much larger models.

## Final performance
As usual, two training runs with seeds 100 and 101 were performed and their final validation bpb was averaged. The projection dimension used was 64. Since 64/256 = 1/4, for SimpleStories with an embedding dimension of 288, the projection dimension used was 72. Here are the results:

| Dataset       | Bits per Byte |
|---------------|---------------|
| BabyLM        | 1.36          |
| SimpleStories | 0.58          |

BabyLM training docker images: [seed 100](docker.io/snoodmcguinness/model_trainer:sha-31bd5f11f07fc76b8e4efc0d015e89fcdc889b03); [seed 101](docker.io/snoodmcguinness/model_trainer:sha-c7abf598df55eb53a67e50372e64ba1b0611402e)

Example Generations:
*caitlin stood on the* ground.
*jayden had a jolly good time*.
*in japanese culture, women are often* more than accurate.

SimpleStories training docker images: [seed 100](docker.io/snoodmcguinness/model_trainer:sha-3e944e8c776064a82bcd3fe55f0e828817f1c1e8"); [seed 101](docker.io/snoodmcguinness/model_trainer:sha-62749e05a241442ba80670d9f557154e5316110c)
*caitlin stood on the* edge of a cliff, gazing at the ocean below. a boy named samuel loved to explore the ocean. one day, he found a shell that shimmered like the stars. when he touched it, he was whisked away to a world of wonders. in this new place, he met a wise turtle named rita. "welcome, young one! what brings you here?" she asked. samuel replied, "i want to see the wonders of the sea." rita smiled and said, "then let\'s
*jayden had a jolly good time*. one day, a girl named mia found a strange map in her attic. the map showed a path to a hidden treasure. excited, she decided to follow it. mia packed a small bag with snacks and a flashlight. she walked through the woods, feeling brave. the trees whispered secrets as she went deeper. suddenly, she heard a loud noise. it was a big bear! the bear looked hungry and hungry. mia thought quickly. she remembered the map and ran back to her house. she
*in japanese culture, women are often* lost in thought. she was a great artist, but her heart was heavy. she had painted many things, but her art was not good enough. one day, she found an old paintbrush in her attic. it was dusty and had strange designs. she thought it could help her create something new. she took it to her workshop and began to paint. as she worked, she felt a spark of hope. but when she finished, she realized she had no paint. the brush was not what she