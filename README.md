# Dr. Diego Bravo - Portfolio Website

Professional portfolio website for Dr. Diego Bravo, Senior Data Scientist and AI Leader.

## Overview

This Jekyll-based portfolio showcases:
- Professional experience and achievements
- Featured machine learning projects
- Technical blog posts
- Research publications
- Complete CV/Resume

## Technology Stack

- **Static Site Generator**: Jekyll 3.9+ (GitHub Pages compatible)
- **Theme**: Minima (customized)
- **Hosting**: GitHub Pages
- **Languages**: Ruby 3.3.6, HTML, CSS, JavaScript
- **Features**:
  - Markdown content
  - Syntax highlighting (Python, SQL, etc.)
  - LaTeX/MathJax for equations
  - Responsive design
  - SEO optimized

## Local Development Setup

### Prerequisites

- Ruby 3.3.6 (check with `ruby -v`)
- Bundler (`gem install bundler`)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ds-portfolio
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run local development server**
   ```bash
   bundle exec jekyll serve
   ```

4. **View site locally**

   Open browser to: `http://localhost:4000`

### Development Commands

```bash
# Serve with live reload
bundle exec jekyll serve --livereload

# Serve with drafts visible
bundle exec jekyll serve --drafts

# Build site (output to _site/)
bundle exec jekyll build

# Clean generated files
bundle exec jekyll clean
```

## Project Structure

```
ds-portfolio/
├── _config.yml           # Site configuration
├── Gemfile               # Ruby dependencies
├── index.md              # Home page
├── _pages/               # Static pages
│   ├── about.md
│   ├── projects.md
│   ├── cv.md
│   └── blog.md
├── _projects/            # Project case studies
│   ├── 01-recommender-system.md
│   ├── 02-pricing-optimization.md
│   └── 03-genai-sales-agent.md
├── _posts/               # Blog posts
│   └── 2025-09-15-production-ml-systems-lessons-learned.md
├── assets/               # Static assets
│   ├── css/
│   │   └── custom.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── .gitignore
├── CLAUDE.md             # Developer guide
└── README.md             # This file
```

## Content Management

### Adding a New Blog Post

1. Create file in `_posts/` with format: `YYYY-MM-DD-title.md`
2. Add front matter:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: 2025-MM-DD
   author: Dr. Diego Bravo
   tags: [machine-learning, python]
   excerpt: Brief description
   ---
   ```
3. Write content in Markdown
4. Commit and push

### Adding a New Project

1. Create file in `_projects/` with format: `NN-project-name.md`
2. Add front matter:
   ```yaml
   ---
   layout: project
   title: "Project Title"
   tech_stack: Python, AWS, XGBoost
   date: 2025-MM-DD
   tags: [ml, optimization]
   excerpt: Project summary
   ---
   ```
3. Write detailed project description
4. Commit and push

### Updating CV

Edit `_pages/cv.md` with latest information.

## GitHub Pages Deployment

### Initial Setup

1. **Create GitHub repository** named `<username>.github.io` or any name

2. **Enable GitHub Pages**
   - Go to repository Settings → Pages
   - Source: Deploy from branch
   - Branch: `main` (or `master`), folder: `/ (root)`
   - Save

3. **Configure site URL** in `_config.yml`:
   ```yaml
   url: "https://<username>.github.io"
   baseurl: ""  # or "/repository-name" if not <username>.github.io
   ```

4. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

5. **Wait for deployment** (2-5 minutes)

6. **Visit site**: `https://<username>.github.io`

### Custom Domain (Optional)

1. Add `CNAME` file to root with your domain:
   ```
   www.diegobravo.com
   ```

2. Configure DNS:
   - Add A records pointing to GitHub IPs:
     - 185.199.108.153
     - 185.199.109.153
     - 185.199.110.153
     - 185.199.111.153
   - Or CNAME record: `<username>.github.io`

3. Enable HTTPS in GitHub Pages settings

### Deployment Workflow

GitHub Pages automatically rebuilds the site when you push to the configured branch:

```bash
# Make changes locally
vim _posts/2025-10-15-new-post.md

# Test locally
bundle exec jekyll serve

# Commit and push
git add .
git commit -m "Add new blog post"
git push origin main

# Site rebuilds automatically on GitHub
# Visit site in 2-5 minutes
```

## Features

### Syntax Highlighting

Supports Python, SQL, JavaScript, and more:

```python
def example():
    return "Highlighted code"
```

### Math Equations

Inline: `$E = mc^2$`

Display:
```latex
$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

### Responsive Design

Optimized for:
- Desktop (1920px+)
- Laptop (1024px-1920px)
- Tablet (768px-1024px)
- Mobile (<768px)

## SEO Optimization

Built-in features:
- Jekyll SEO Tag plugin
- Sitemap generation
- RSS feed
- Open Graph tags
- Twitter Cards
- Schema.org markup

## Customization

### Changing Colors

Edit `assets/css/custom.css`:
```css
:root {
  --primary-color: #0066cc;  /* Change this */
}
```

### Modifying Layout

- Templates in `_layouts/` (if created)
- Default theme layouts used otherwise
- Override by creating matching file in `_layouts/`

### Adding Analytics

Add to `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXXX-X
```

## Troubleshooting

### Site not building on GitHub Pages

- Check repository Settings → Pages for error messages
- Verify `_config.yml` syntax
- Ensure Ruby version compatibility
- Check GitHub Actions tab for build logs

### Local development issues

```bash
# Clear cache and rebuild
bundle exec jekyll clean
bundle exec jekyll build --verbose

# Update dependencies
bundle update

# Check Ruby version
ruby -v  # Should be 3.3.6
```

### Missing gems

```bash
bundle install
```

## Maintenance

### Updating Dependencies

```bash
# Check for updates
bundle outdated

# Update gems
bundle update

# Test locally
bundle exec jekyll serve
```

### Keeping Content Fresh

- Update CV quarterly
- Add blog posts monthly
- Update project descriptions as completed
- Refresh About page annually

## Performance

- Lighthouse Score Target: 90+
- Mobile-friendly: Yes
- Page load time: <2s
- Images: Optimized (<200KB)

## Support

For Jekyll documentation: [jekyllrb.com](https://jekyllrb.com)
For GitHub Pages help: [docs.github.com/pages](https://docs.github.com/pages)

## License

© 2025 Dr. Diego Bravo. All rights reserved.

Content is personal portfolio material. Code structure can be referenced for educational purposes.

## Contact

**Dr. Diego Bravo**
- Email: dbravo27@gmail.com
- LinkedIn: [linkedin.com/in/diegobravoguerrero](https://linkedin.com/in/diegobravoguerrero)
- Location: Plano, Texas

---

*Built with Jekyll and hosted on GitHub Pages*
