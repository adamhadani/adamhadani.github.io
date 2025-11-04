---
layout: splash
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: # Add your header image path here if you have one
  actions:
    - label: "Read My Blog"
      url: "/blog/"
excerpt: "Code, AI/ML, Math, and other esoteric topics"
intro:
  - excerpt: 'Welcome to my personal website. I write about technology, artificial intelligence, mathematics, music, and whatever else captures my curiosity.'
feature_row:
  - image_path: # Add image if you have one
    title: "Blog"
    excerpt: "Thoughts on code, AI/ML, mathematics, and more."
    url: "/blog/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: # Add image if you have one
    title: "Music"
    excerpt: "My musical projects and compositions."
    url: "/music/"
    btn_label: "Listen"
    btn_class: "btn--primary"
  - image_path: # Add image if you have one
    title: "About Me"
    excerpt: "Learn more about me and what I do."
    url: "/about/"
    btn_label: "About"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

{% include feature_row %}
