# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Professional portfolio website for Dr. Diego Bravo, Senior Data Scientist and AI Leader. Built with Jekyll and optimized for GitHub Pages deployment. Showcases professional experience, ML projects, technical blog posts, research publications, and CV.

## Technology Stack

- **Static Site Generator**: Jekyll 3.9+ (GitHub Pages compatible)
- **Ruby Version**: 3.3.6 (managed via `.ruby-version`)
- **Theme**: Minima (with custom styling)
- **Deployment**: GitHub Pages
- **Features**: Markdown content, syntax highlighting (Python focus), LaTeX/MathJax support, responsive design, SEO optimization

## Development Commands

### Setup
```bash
# Install dependencies
bundle install

# Run local development server
bundle exec jekyll serve

# Run with live reload
bundle exec jekyll serve --livereload

# Build site (output to _site/)
bundle exec jekyll build

# Clean generated files
bundle exec jekyll clean
```

### Common Tasks

**Test locally before pushing**:
```bash
bundle exec jekyll serve
# Visit http://localhost:4000
```

**Check for broken links/issues**:
```bash
bundle exec jekyll build --verbose
```

**Update dependencies**:
```bash
bundle update
```

## Project Structure

```
ds-portfolio/
├── _config.yml              # Jekyll configuration, site metadata
├── Gemfile                  # Ruby dependencies (github-pages gem)
├── index.md                 # Landing page with hero section
├── _pages/                  # Static pages
│   ├── about.md            # Professional background, education, experience
│   ├── projects.md         # Projects overview page
│   ├── cv.md               # Full CV/resume
│   └── blog.md             # Blog listing page
├── _projects/              # Project case studies (collection)
│   ├── 01-recommender-system.md
│   ├── 02-pricing-optimization.md
│   └── 03-genai-sales-agent.md
├── _posts/                 # Blog posts (collection)
│   └── YYYY-MM-DD-title.md
├── assets/
│   ├── css/custom.css      # Custom styling
│   ├── js/main.js          # JavaScript enhancements
│   └── images/             # Images and media
└── .gitignore
```

## Content Management

### Adding a Blog Post

1. Create file: `_posts/YYYY-MM-DD-title.md`
2. Add front matter:
```yaml
---
layout: post
title: "Post Title"
date: 2025-MM-DD
author: Dr. Diego Bravo
tags: [machine-learning, python, mlops]
reading_time: 10
excerpt: Brief description for listings
---
```
3. Write content in Markdown
4. Test locally, then commit

### Adding a Project

1. Create file: `_projects/NN-project-name.md` (NN for ordering)
2. Add front matter:
```yaml
---
layout: project
title: "Project Name"
tech_stack: Python, AWS, XGBoost, Docker
date: 2025-MM-DD
company: Company Name
tags: [machine-learning, optimization]
excerpt: One-sentence project summary
---
```
3. Include: overview, challenge, solution, results, tech stack, learnings
4. Use code examples (Python) to demonstrate technical depth

### Updating CV

Edit `_pages/cv.md` with latest experience, publications, or certifications.

## Deployment

### GitHub Pages

Site automatically rebuilds when pushing to `main` branch (typically 2-5 minutes).

**Deployment checklist**:
1. Test locally: `bundle exec jekyll serve`
2. Check no build errors
3. Commit and push to `main`
4. Wait for GitHub Pages rebuild
5. Verify live site

**Configuration** (`_config.yml`):
```yaml
url: "https://<username>.github.io"
baseurl: ""  # or "/repo-name" if not user site
```

## Content Guidelines

### Tone & Style

- **Professional but approachable**: Technical depth without jargon overload
- **Results-oriented**: Lead with business impact, follow with technical details
- **Code examples**: Use Python primarily, with clear comments
- **Quantify impact**: "$10M revenue", "97% reduction", "95% coverage"

### Technical Writing

- Use proper terminology: ML, MLOps, LLM, GenAI, NLP, CV
- Include code snippets for blog posts (demonstrate expertise)
- Cite specific metrics and results
- Link to relevant papers/documentation when appropriate

### SEO Optimization

- Use descriptive titles and excerpts
- Include relevant tags
- Add alt text for images (when used)
- Internal linking between posts/projects

## Syntax Highlighting

Configured for Python (default), also supports:
- SQL
- Bash
- JavaScript
- YAML

Example:
````markdown
```python
import pandas as pd

def preprocess_data(df):
    return df.dropna()
```
````

## Math/LaTeX Support

MathJax enabled for equations:

**Inline**: `$E = mc^2$`

**Display**:
```latex
$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$
```

## Customization

### Changing Colors/Styling

Edit `assets/css/custom.css`:
```css
:root {
  --primary-color: #0066cc;
  --secondary-color: #004999;
  --accent-color: #ff6b35;
}
```

### Modifying Layouts

Default Minima theme layouts are used. Override by creating files in `_layouts/`:
- `default.html` - Base template
- `page.html` - Static pages
- `post.html` - Blog posts
- `project.html` - Project case studies

## Common Issues

### Build fails on GitHub Pages

- Check `_config.yml` syntax (YAML errors)
- Ensure all plugins are GitHub Pages compatible
- Review GitHub Actions build log for details

### Local serve errors

```bash
# Clear and rebuild
bundle exec jekyll clean
bundle exec jekyll build --verbose

# Update gems
bundle update

# Check Ruby version
ruby -v  # Should be 3.3.6
```

### Missing images/assets

- Ensure paths are correct: `/assets/images/file.png`
- Check `.gitignore` doesn't exclude asset files
- Verify files are committed to repository

## Performance Optimization

- Keep images under 200KB (optimize before uploading)
- Minimize custom CSS/JS
- Use lazy loading for images if needed
- Target Lighthouse score: 90+

## Maintenance

- Update CV quarterly (new experience, publications)
- Add blog posts monthly (maintain thought leadership)
- Refresh project descriptions as completed
- Review and update dependencies: `bundle update`

## Resources

- Jekyll Docs: [jekyllrb.com](https://jekyllrb.com)
- GitHub Pages: [docs.github.com/pages](https://docs.github.com/pages)
- Markdown Guide: [markdownguide.org](https://www.markdownguide.org)
- MathJax: [mathjax.org](https://www.mathjax.org)

## Contact Information

**Dr. Diego Bravo**
- Email: dbravo27@gmail.com
- LinkedIn: [linkedin.com/in/diegobravoguerrero](https://linkedin.com/in/diegobravoguerrero)
- Location: Plano, Texas
- Visa: O1-A (Extraordinary Abilities in AI and Mathematics)
