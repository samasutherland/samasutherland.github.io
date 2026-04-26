---
title: little Language Models
layout: page
permalink: /llms/
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
The best way to learn is to do. I recently found out that hiring RTX 4090s on runpod costs only $0.59 an hour and figured, hey, I can probably afford 30 minutes of training. So I decided to try to build the best language model I can with a maximum of 30 minutes training time (excluding pre-train tuning). To do this, I created a tokenizer, downloaded the SimpleStories dataset, and wrote a little package to train some models. You can access the repository [here](https://github.com/samasutherland/little-language-models), and see below to read about each model.
### lLM Leaderboard
<p><em>Click any column header to sort. Click again to reverse.</em></p>
<style>
  #llm-leaderboard thead th {
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  #llm-leaderboard thead th::after {
    content: " \00a0";
    opacity: 0.35;
    font-size: 0.75em;
  }
  #llm-leaderboard thead th.sort-asc::after { content: " \25b2"; opacity: 0.7; }
  #llm-leaderboard thead th.sort-desc::after { content: " \25bc"; opacity: 0.7; }
</style>
<table id="llm-leaderboard">
  <thead>
    <tr>
      <th scope="col" data-sort-col="0" data-sort-type="number">Rank</th>
      <th scope="col" data-sort-col="1" data-sort-type="string">Model</th>
      <th scope="col" data-sort-col="2" data-sort-type="number">BabyLM<br>param count</th>
      <th scope="col" data-sort-col="3" data-sort-type="number">BabyLM BPB</th>
      <th scope="col" data-sort-col="4" data-sort-type="number">SimpleStories<br>param count</th>
      <th scope="col" data-sort-col="5" data-sort-type="number">SimpleStories BPB</th>
    </tr>
  </thead>
  <tbody>
  {%- assign rows = site.data.llms -%}
  {%- if rows and rows.size > 0 -%}
    {%- assign rows = rows | sort: "babylm_bpb" -%}
    {%- for r in rows -%}
      <tr>
        <td data-sort="{{ forloop.index }}">{{ forloop.index }}</td>
        <td data-sort="{{ r.model | escape }}">
          {%- if r.link -%}
            {%- if r.link contains '://' -%}
              <a href="{{ r.link }}">{{ r.model }}</a>
            {%- else -%}
              <a href="{{ r.link | relative_url }}">{{ r.model }}</a>
            {%- endif -%}
          {%- else -%}
            {{ r.model }}
          {%- endif -%}
        </td>
        <td data-sort="{{ r.babylm_param_count }}">{{ r.babylm_param_count }}</td>
        <td data-sort="{{ r.babylm_bpb }}">{{ r.babylm_bpb }}</td>
        <td data-sort="{{ r.simplestories_param_count }}">{{ r.simplestories_param_count }}</td>
        <td data-sort="{{ r.simplestories_bpb }}">{{ r.simplestories_bpb }}</td>
      </tr>
    {%- endfor -%}
  {%- else -%}
    <tr><td colspan="6">No models yet.</td></tr>
  {%- endif -%}
  </tbody>
</table>
<script>
(function () {
  var table = document.getElementById("llm-leaderboard");
  if (!table) return;

  var tbody = table.tBodies[0];
  if (!tbody) return;

  var headers = Array.prototype.slice.call(table.querySelectorAll("thead th[data-sort-col]"));
  var sortState = { col: -1, dir: 1 };

  function parseScaledNumber(value) {
    var s = String(value || "").trim().replace(/,/g, "");
    var m = s.match(/^(-?\d*\.?\d+)\s*([kKmMbBtT])?$/);
    if (!m) return parseFloat(s);
    var base = parseFloat(m[1]);
    if (isNaN(base)) return NaN;
    var suffix = (m[2] || "").toLowerCase();
    var scale = 1;
    if (suffix === "k") scale = 1e3;
    else if (suffix === "m") scale = 1e6;
    else if (suffix === "b") scale = 1e9;
    else if (suffix === "t") scale = 1e12;
    return base * scale;
  }

  function getCellValue(row, col) {
    var cell = row.cells[col];
    if (!cell) return "";
    var raw = cell.getAttribute("data-sort");
    return raw == null ? cell.textContent.trim() : String(raw).trim();
  }

  function compareRows(a, b, col, type) {
    var va = getCellValue(a, col);
    var vb = getCellValue(b, col);

    if (type === "number") {
      var na = parseScaledNumber(va);
      var nb = parseScaledNumber(vb);
      if (!isNaN(na) && !isNaN(nb)) return na - nb;
    }

    if (type === "date") {
      var da = Date.parse(va);
      var db = Date.parse(vb);
      if (!isNaN(da) && !isNaN(db)) return da - db;
    }

    return va.localeCompare(vb, undefined, { numeric: true, sensitivity: "base" });
  }

  function clearHeaderClasses() {
    headers.forEach(function (h) {
      h.classList.remove("sort-asc", "sort-desc");
    });
  }

  function sortBy(th) {
    var col = parseInt(th.getAttribute("data-sort-col"), 10);
    if (isNaN(col)) return;
    var type = th.getAttribute("data-sort-type") || "string";

    if (sortState.col === col) sortState.dir = -sortState.dir;
    else {
      sortState.col = col;
      sortState.dir = 1;
    }

    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
    if (rows.length < 2) return;

    rows.sort(function (a, b) {
      return sortState.dir * compareRows(a, b, col, type);
    });

    rows.forEach(function (row) {
      tbody.appendChild(row);
    });

    clearHeaderClasses();
    th.classList.add(sortState.dir === 1 ? "sort-asc" : "sort-desc");
  }

  headers.forEach(function (th) {
    th.tabIndex = 0;
    th.setAttribute("role", "button");
    th.setAttribute("aria-label", "Sort by " + th.textContent.trim());
    th.setAttribute("title", "Sort by this column");

    th.addEventListener("click", function () {
      sortBy(th);
    });

    th.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        sortBy(th);
      }
    });
  });
})();
</script>

### Posts
<ul class="post-list">
  {% assign llm_posts = site.categories.llms %}
  {% if llm_posts and llm_posts.size > 0 %}
    {% assign llm_posts = llm_posts | sort: "date" | reverse %}
    {% for post in llm_posts %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> — {{ post.date | date_to_string }}</li>
    {% endfor %}
  {% else %}
    <p>No ML posts yet.</p>
  {% endif %}
</ul>
