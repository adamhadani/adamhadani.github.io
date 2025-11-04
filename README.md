# Adam Ever-Hadani's Personal Website

A personal portfolio and blog built with Jekyll and hosted on GitHub Pages.

**Live Site**: [https://adamhadani.github.io](https://adamhadani.github.io)

## About This Site

This is a Jekyll-based blog and portfolio website featuring:
- Blog posts about AI/ML, software engineering, mathematics, music, and more
- Professional biography and accomplishments
- Minimal Mistakes theme with the "Sunrise" skin
- Automatic deployment via GitHub Pages

## Prerequisites

- **Ruby**: Version 3.4.7 (see `.ruby-version`)
- **Bundler**: Ruby dependency manager

## Getting Started

### 1. Install Dependencies

```bash
# Install Ruby gems
bundle install
```

### 2. Run Locally

```bash
# Start the development server
bundle exec jekyll serve

# Site will be available at: http://localhost:4000
```

The site will automatically rebuild when you make changes to files (except `_config.yml`, which requires a server restart).

### 3. Build for Production

```bash
# Generate static site in _site/ directory
bundle exec jekyll build
```

## Writing Content

### Creating a New Blog Post

1. Create a new file in `_posts/` following the naming convention:
   ```
   YYYY-MM-DD-title-slug.markdown
   ```

2. Add front matter at the top:
   ```yaml
   ---
   layout: single
   title: "Your Post Title"
   date: 2025-03-13 12:00:00 -0500
   categories: technology ai-ml
   ---
   ```

3. Write your content in Markdown below the front matter

4. Save and preview locally with `bundle exec jekyll serve`

5. Commit and push to deploy

### Updating Existing Pages

Pages are located in `_pages/`:
- `about.md` - About/biography page
- `contact.md` - Contact information
- `music.md` - Music projects (currently placeholder)

Edit these files directly in Markdown format.

### Homepage

The homepage (`index.html`) displays your blog posts in reverse chronological order. It uses the `home` layout and automatically handles pagination (5 posts per page).

## Customization

### Changing Site Configuration

Edit `_config.yml` to modify:
- Site title and description
- Author information and bio
- Social media links (GitHub, Twitter, LinkedIn)
- Theme settings

**Important**: After changing `_config.yml`, restart the Jekyll server.

### Navigation Menu

Edit `_data/navigation.yml` to add/remove menu items:

```yaml
main:
  - title: "New Page"
    url: /new-page/
```

### Theme and Colors

The site uses the Minimal Mistakes "Sunrise" theme (warm oranges and reds).

To change themes, edit `_config.yml`:
```yaml
minimal_mistakes_skin: "sunrise"  # Options: air, aqua, contrast, dark, dirt, neon, mint, plum, sunrise
```

## Deployment

The site automatically deploys to GitHub Pages when you push to the `main` branch.

### Deployment Workflow

1. Make your changes locally
2. Test with `bundle exec jekyll serve`
3. Commit your changes:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```
4. Push to GitHub:
   ```bash
   git push origin main
   ```
5. GitHub Pages will automatically rebuild and deploy (takes 1-2 minutes)

## Project Structure

```
├── _config.yml           # Site configuration
├── _data/
│   └── navigation.yml    # Navigation menu
├── _pages/               # Static pages (About, Contact, etc.)
├── _posts/               # Blog posts
├── _site/                # Generated site (don't edit, git-ignored)
├── index.html            # Homepage/blog listing
├── Gemfile               # Ruby dependencies
└── CLAUDE.md             # Detailed architecture documentation
```

## Troubleshooting

### Server won't start

```bash
# Clean and rebuild
rm -rf _site .jekyll-cache
bundle exec jekyll serve
```

### Dependency issues

```bash
# Update all gems
bundle update

# Or reinstall from scratch
rm Gemfile.lock
bundle install
```

### Changes not appearing

- **Config changes**: Restart the Jekyll server (`Ctrl+C` then `bundle exec jekyll serve`)
- **Content changes**: Should auto-rebuild (check terminal for errors)
- **On GitHub Pages**: Wait 1-2 minutes for deployment to complete

### Build warnings

The build should complete without warnings. If you see warnings about:
- **Layouts**: Ensure posts use `layout: single`
- **Pagination**: Ensure `index.html` exists (not `index.md`)

## Common Tasks

### Adding a social link

Edit `_config.yml` in the `author.links` and `footer.links` sections:

```yaml
- label: "Your Platform"
  icon: "fab fa-fw fa-platform-icon"
  url: "https://yourplatform.com/username"
```

### Updating your bio

Edit the `author.bio` field in `_config.yml` for the sidebar bio, or edit `_pages/about.md` for the full biography page.

### Adding code snippets to posts

Use fenced code blocks with syntax highlighting:

````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

## Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Minimal Mistakes Theme Documentation](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## Technical Details

For detailed architecture information and guidance for AI assistants, see [CLAUDE.md](CLAUDE.md).

## License

This is a personal website. Content is © Adam Ever-Hadani.

## Contact

- GitHub: [@adamhadani](https://github.com/adamhadani)
- Twitter: [@adamhadani](https://twitter.com/adamhadani)
- LinkedIn: [adamhadani](https://www.linkedin.com/in/adamhadani)
