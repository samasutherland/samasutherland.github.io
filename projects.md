---
title: Projects
layout: page
permalink: /projects/
---

A collection of side projects and experiments.

### little Language Models (lLMs)

An attempt to train the best language model I can in 30 minutes of GPU time, with a
sortable leaderboard and write-ups of each model and architecture experiment.
[Read more]({{ '/llms/' | relative_url }}).

### Second-Order Jacobian Lens

A feasibility study, started as a MATS application project, into whether the Jacobian
Lens can be extended with second-order information to surface multi-token concepts and
context dependence.
[Read more]({{ '/projects/second-order-jacobian-lens/' | relative_url }}).

<ul class="post-list">
  {% assign project_posts = site.categories.projects %}
  {% if project_posts and project_posts.size > 0 %}
    {% assign project_posts = project_posts | sort: "date" | reverse %}
    {% for post in project_posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> — {{ post.date | date_to_string }}</li>
    {% endfor %}
  {% endif %}
</ul>
