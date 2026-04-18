---
title: Little Language Models
layout: page
permalink: /ml/
---

10 years ago, state-of-the-art machine learning was all about coming up with new and innovative architectural changes that injected better inductive bias into models. CNNs, skip connections, GRUs, GNNs... new layers and structural modifications galore dominated the literature, with large leaps in performance stemming from more efficient data representation and large-scale GPU training. Today, with the advent of large language models, natural language processing has a very different research landscape, with the fundamental model architecture (the transformer) largely stable, access to data practically infinite, and priorities switching to inference efficiency for serving models to consumers.

A major driving factor for this change was the advent of scaling laws, where relationships between performance, parameter count, training tokens, and training FLOPs were found to persist over orders of magnitude variations in the parameters, and to be somewhat robust to changes in model structure. Having identified this, OpenAI took a huge gamble and committed an ungodly amount of money to a (for the time) behemoth training run of the first frontier-scale LLM, GPT-3, banking on those scaling laws remaining intact at scales not previously seen. Their bet was successful, and now GPT-3 and its successors drive a multi-billion-dollar industry and have transformed the way that computers can be interacted with.

Since scaling compute has been so effective, transformers have become the dominant architecture, and RLHF has pretty much solved making chatbots more "chatty", research has naturally shifted to analysing these huge LLMs; studying the surprising and seemingly emergent abilities they possess, how to prompt in interesting ways to get the best responses, and of course trying to understand how they actually work and what they're actually doing to prevent mass extinction of the human race.

For people like myself who have a keen interest in machine learning experimentation, but unfortunately don't have $100 million handy to throw at some H100s, these developments mean a few positive things:

1. With access to model APIs, *anyone* can analyse LLMs and invent their own imaginative prompting schemes
2. Scaling laws work both ways: I can run experiments on small models and have more confidence that results will propagate to larger models
3. GPUs are now pretty cheap to rent

I'm currently wrapping up my PhD in quantum computing at Silicon Quantum Computing, and I'm excited to shift into AI research afterward. This page documents my attempt to exploit these three boons to get up to date in the field for my imminent re-entry into the industry.

### little Language Models (lLMs)
The best way to learn is to do. I recently found out that hiring RTX 4090s on runpod costs only $0.59 an hour and figured, hey, I can probably afford 30 minutes of training. So I decided to try to build the best language model I can with a maximum of 30 minutes training time (excluding pre-train tuning). To do this, I created a tokenizer, downloaded the SimpleStories dataset, and wrote a little package to train some models. Read about the setup [here](), and see below to read about each model.
### lLM Leaderboard
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Score</th>
      <th>Notes</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
  {%- assign rows = site.data.llms -%}
  {%- if rows and rows.size > 0 -%}
    {%- assign rows = rows | sort: "score" | reverse -%}
    {%- for r in rows -%}
      <tr>
        <td>{{ forloop.index }}</td>
        <td>
          {%- if r.link -%}
            <a href="{{ r.link }}">{{ r.model }}</a>
          {%- else -%}
            {{ r.model }}
          {%- endif -%}
        </td>
        <td>{{ r.score }}</td>
        <td>{{ r.notes }}</td>
        <td>{{ r.date }}</td>
      </tr>
    {%- endfor -%}
  {%- else -%}
    <tr><td colspan="5">No models yet.</td></tr>
  {%- endif -%}
  </tbody>
</table>

### Posts
<ul class="post-list">
  {% assign ml_posts = site.categories.ml %}
  {% if ml_posts and ml_posts.size > 0 %}
    {% assign ml_posts = ml_posts | sort: "date" | reverse %}
    {% for post in ml_posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> — {{ post.date | date_to_string }}</li>
    {% endfor %}
  {% else %}
    <p>No ML posts yet.</p>
  {% endif %}
</ul>