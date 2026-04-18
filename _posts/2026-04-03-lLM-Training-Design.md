---
layout: post
title: "lLM Training Design"
date: 2026-04-03
categories: [ml]
tags: []
---

Short teaser paragraph about the experiment. <!--more-->

[//]: # (Training a performant language model is a time and cost intensive process that accounts for a significant chunk of the total cost of a large model. Here, the aim is to emulate the process of experimenting with large models, but using small models instead. For this to make sense, all parts of the pipeline should be scaled down. This means using a single GPU, smaller embedding dimensions, a less verbose vocabulary, and less training examples.)

[//]: # ()
[//]: # (To train smaller language models, some compromises have to be made to the data. Firstly, it has been found that as much as X% of a model's weights are dedicated purely to storing the definitions of words including nouns and proper nouns, so a dataset designed for smaller models must seriously limit the quantity of these types of words. Secondly, the shallower models limit the depth to which complex grammatical structures can be parsed. Finally, ...)

[//]: # ()
[//]: # (Though it may be shocking to some of you, I'm not the first person to come up with the idea of training small models. As such, many datasets have been used to try to train smaller models. A common choice is to take a small subset of wikipedia pages~\cite{}. Unfortunately, wikipedia pages tend to introduce many nouns, taking up valuable space in the embedding feature spaces and tokenizers. )

The aim of this package is to train small language models with a budget of 30 minutes training (not inclusive of pre-train tuning).

To do this 

To train smaller models, 
-- SimpleStories Dataset
-- Pick an embedding dim
-- Vary depth to get tokens/parameter correct
-- 



## Results
Findings, charts, tables…

## Takeaways
Bulleted conclusions.
