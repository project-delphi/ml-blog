---
title: "Quarto Workshop"
author: "Ravi Kalia"
date: "2025-01-22"
format:
  revealjs:
    theme: black
    transition: slide
---

# Quarto Workshop

## Prerequisites

- GitHub account
- Linux-like environment
  - VSCode installed
  - Configured Git
  - Conda installed

**Skills Needed:**
- Creating GitHub repositories
- Branching (GitHub Flow)

---

## What is Quarto?

**Quarto** is a modern, interactive publishing system built on **pandoc**:

- Converts Markdown files to multiple formats:
  - **HTML**
  - **PDF**
  - **RMarkdown**
  - **Slides (Reveal.js)**
  - **Jupyter Notebooks (ipynb)**

**Roots:**
- `org-mode` (Emacs)
- `Sweave` / `knitr`
- `RMarkdown` / `Shiny`

### Why Use Quarto?

- Focus on reproducibility
- Easy interactivity
- Flexible output formats

---

## Quarto Example Gallery

### Explore Examples

1. Open the [Quarto Examples Gallery](https://quarto.org/docs/gallery/).
2. Browse through the examples of:
   - Scientific reports
   - Data visualizations
   - Blogs
   - Interactive web apps

**Exercise:**
- Identify one example to recreate during the workshop.

---

## Underlying Technology

Quarto can be viewed as a transpiler that converts markdown to various formats. The foundational technologies are:

* Javascript
* HTML
* CSS

---

## Installation

### Install Quarto

1. Follow instructions at [quarto.org](https://quarto.org).
2. Install the Quarto extension for VSCode:
   - Extensions > Search "Quarto" > Install

**Exercise:**
- Install [Quarto](https://quarto.org/docs/get-started/) on your system.

---

## Setting Up Your Environment

1. Create a new directory for the workshop:
   ```bash
   mkdir quarto-workshop
   cd quarto-workshop
   ```

2. Create a Conda environment:
   ```bash
   conda create -n quarto-env python=3.10 -y
   conda activate quarto-env
   ```

3. Initialize Git:
   ```bash
   git init
   ```

---

## Markdown Basics

### Write and Convert Markdown

1. Write a small `example.md` file:
   ```markdown
   # Hello, Quarto!

   This is a demo of Markdown features:

   - **Bold text**
   - *Italic text*
   - [Links](https://example.com)

   ```

2. Convert Markdown to:
   ```bash
   quarto render example.md --to pdf
   quarto render example.md --to html
   quarto render example.md --to ipynb
   ```

**Exercise:**
- Write and convert a Markdown file.

---

## Quarto Markdown Features

### Enhance with Quarto Syntax

1. Write a small `example.qmd` file:
   ```markdown
   ---
   title: "Demo"
   format: html
   ---

   # Hello, Quarto!

   ::: {.panel-tabset}
   ### Tab 1
   Content for tab 1.

   ### Tab 2
   Content for tab 2.
   :::
   ```

2. Render the file:
   ```bash
   quarto render example.qmd
   ```

**Tip:** Use `#|` for metadata.

**Exercise:**
- Write a `.qmd` file with tabs or fenced divs.

---

## Create a Blog Directory

### Skeleton Setup

1. Create a Quarto blog project:
   ```bash
   quarto create-project blog --type website
   ```

2. Initialize Git and create branches:
   ```bash
   git init
   git checkout -b blog-setup
   ```

3. Write an environment file:
   ```yaml
   name: blog-env
   channels:
     - conda-forge
   dependencies:
     - python=3.10
     - quarto
   ```

4. Activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate blog-env
   ```

---

## Writing a Blog Post

1. Create a new `.qmd` file in the `posts` folder:
   ```markdown
   ---
   title: "My First Blog Post"
   date: 2025-01-20
   ---

   # Welcome!

   This is my first post using **Quarto**!
   ```

2. Render and preview:
   ```bash
   quarto preview
   ```

---

## Configure GitHub Pages

1. Configure `_quarto.yml`:
   ```yaml
   project:
     type: website
   publish:
     gh-pages:
       branch: gh-pages
   ```

2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Initial blog setup"
   git push origin blog-setup
   ```

3. Merge to `main` and deploy.

---

## Group Exercise

### Collaborate on a Blog

1. Form groups of 3–4.
2. Each group member creates a branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Write posts in the `posts/` directory.
4. Merge branches to `main`.
5. Push and share your blog URL.

---

## Recap

- **Quarto** combines flexibility and reproducibility.
- Create professional blogs, reports, and more.
- Practice collaborative workflows with GitHub.

**Next Steps:**
- Explore advanced Quarto features.
- Build your personal portfolio site!

---

## Questions?
