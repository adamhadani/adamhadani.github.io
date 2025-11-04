# adamhadani.github.io - Codebase Architecture Guide

## Project Overview

This is a **personal portfolio and blog website** built with Jekyll, hosted on GitHub Pages. It serves as a digital presence showcasing blog posts, music projects, and personal information for Adam Ever-Hadani.

**Site URL**: https://adamhadani.github.io
**Repository**: https://github.com/adamhadani/adamhadani.github.io
**Technology Stack**: Jekyll 4.x + Minimal Mistakes Theme v4.26.2

---

## 1. Overall Project Structure

```
adamhadani.github.io/
├── _config.yml                 # Primary site configuration
├── _data/                      # Data files (YAML)
│   └── navigation.yml          # Main navigation menu configuration
├── _pages/                     # Static pages (not date-based)
│   ├── about.md               # About page
│   ├── blog.md                # Blog listing page
│   ├── contact.md             # Contact page
│   └── music.md               # Music projects page (placeholder)
├── _posts/                     # Blog posts (date-based content)
│   └── 2025-03-13-welcome-to-jekyll.markdown
├── _site/                      # Generated static site (build output)
├── 404.html                    # Custom 404 error page
├── index.markdown              # Homepage with splash layout
├── Gemfile                     # Ruby dependencies specification
├── Gemfile.lock                # Locked dependency versions
├── .ruby-version               # Ruby version specification (3.4.7)
├── .gitignore                  # Git ignore rules
└── .git/                       # Git repository metadata
```

**Key Pattern**: This follows the standard Jekyll project structure with:
- Content separated into posts (time-stamped blog articles) and pages (static content)
- Configuration-driven approach with minimal custom code
- Remote theme usage (no custom theme files in repo)

---

## 2. Key Configuration Files and Their Roles

### 2.1 `_config.yml` - Site Configuration (PRIMARY FILE)

**Purpose**: Central configuration hub for the entire Jekyll site.

**Key Sections**:

#### Site Identity
```yaml
title: Adam Ever-Hadani
email: adamhadani@gmail.com
description: Personal blog about code, AI/ML, math, and esoteric topics
baseurl: ""                              # No subdirectory
url: "https://adamhadani.github.io"      # Full domain
```

#### Social Links
```yaml
twitter_username: adamhadani
github_username: adamhadani
```

#### Theme Configuration
```yaml
remote_theme: "mmistakes/minimal-mistakes@4.26.2"  # Remote theme (loaded from GitHub)
minimal_mistakes_skin: "dark"                        # Dark theme variant
```

#### Plugin Configuration
```yaml
plugins:
  - jekyll-feed              # RSS/Atom feed generation
  - jekyll-include-cache     # Cache include files for performance
  - jekyll-paginate          # Pagination support for blog
  - jekyll-sitemap           # Auto-generate XML sitemap
  - jekyll-gist              # Embed GitHub gists
```

#### Pagination
```yaml
paginate: 5                 # Show 5 posts per page
paginate_path: /page:num/   # Pagination URL structure
```

#### Author & Footer Configuration
```yaml
author:
  name: "Adam Ever-Hadani"
  bio: "Code, AI/ML, Math, and other esoteric topics"
  email: "adamhadani@gmail.com"
  links:
    - label: "GitHub"
      icon: "fab fa-fw fa-github"
      url: "https://github.com/adamhadani"
    - label: "Twitter"
      icon: "fab fa-fw fa-twitter-square"
      url: "https://twitter.com/adamhadani"

footer:
  links:
    - label: "GitHub"
      icon: "fab fa-fw fa-github"
      url: "https://github.com/adamhadani"
    - label: "Twitter"
      icon: "fab fa-fw fa-twitter-square"
      url: "https://twitter.com/adamhadani"
```

#### Front Matter Defaults
```yaml
defaults:
  # Posts: single layout, author profile, read time, sharing enabled
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: false
      share: true
      related: true
  
  # Pages: single layout with author profile
  - scope:
      path: ""
      type: pages
    values:
      layout: single
      author_profile: true
```

**Architecture Pattern**: Configuration-first design - all theme customization and behavior controlled through `_config.yml` rather than code modifications.

### 2.2 `_data/navigation.yml` - Navigation Structure

```yaml
main:
  - title: "Blog"
    url: /blog/
  - title: "Music"
    url: /music/
  - title: "About"
    url: /about/
  - title: "Contact"
    url: /contact/
```

**Purpose**: Defines the main navigation menu. Referenced by Minimal Mistakes theme template.
**Architecture Pattern**: Externalized navigation data - updates here automatically reflect in the theme.

### 2.3 Gemfile - Ruby Dependency Management

**Current Setup**:
- Uses GitHub Pages gem `~> 232` (includes Jekyll 4.3+)
- Specifies compatible jekyll-feed, jekyll-paginate, jekyll-sitemap, jekyll-gist plugins
- Includes performance fixes: `faraday-retry`, `openssl` (Ruby 3.4.7 compatibility)
- Platform-specific gems for Windows and JRuby support

**Key Decision**: Uses `github-pages` gem rather than explicit `jekyll` gem - ensures compatibility with GitHub Pages build environment.

### 2.4 .ruby-version - Ruby Version Lock

```
3.4.7
```

**Purpose**: Specifies Ruby 3.4.7 for use with rbenv/rvm.
**Importance**: Ensures consistency across development environments.

---

## 3. Content Organization

### 3.1 Pages (`_pages/` directory)

Static pages that aren't date-based. Each uses front matter to configure presentation.

**Structure Pattern**: 
```yaml
---
layout: single
title: "Page Title"
permalink: /page-url/
author_profile: true
---
```

**Pages**:

1. **about.md** - Personal biography
   - Describes Adam's interests: Code, AI/ML, Mathematics, Esoteric Topics
   - Links to GitHub and Twitter profiles
   - `permalink: /about/`

2. **blog.md** - Blog listing page
   - Uses `layout: home` (Minimal Mistakes blog listing layout)
   - `permalink: /blog/`
   - Shows paginated list of blog posts with 5 posts per page

3. **contact.md** - Contact information
   - Email, GitHub, Twitter links
   - Responsive message about response time
   - `permalink: /contact/`

4. **music.md** - Placeholder for music projects
   - Currently showing "Coming soon..."
   - Ready for future expansion with musical compositions
   - `permalink: /music/`

### 3.2 Posts (`_posts/` directory)

Date-stamped blog articles following Jekyll naming convention.

**Current Posts**:
- `2025-03-13-welcome-to-jekyll.markdown` - Default Jekyll example post (includes code highlighting example)

**Naming Convention**: `YYYY-MM-DD-title-slug.markdown`

**Post Front Matter**:
```yaml
---
layout: post
title: "Post Title"
date: 2025-03-13 22:54:25 -0500
categories: jekyll update
---
```

**Default Configuration Applied** (via `_config.yml` defaults):
- `layout: post`
- `author_profile: true`
- `read_time: true`
- `share: true`
- `related: true`

### 3.3 Data Files (`_data/` directory)

YAML/JSON data referenced across the site.

**Current Files**:
- `navigation.yml` - Navigation menu structure

**Architecture Pattern**: Externalized data for dynamic content without templates.

### 3.4 Assets

The Minimal Mistakes remote theme handles all CSS, JavaScript, and image assets. No custom assets committed to repository - theme assets are loaded from the remote theme repository.

---

## 4. Theme and Layout Structure

### 4.1 Theme: Minimal Mistakes v4.26.2

**Type**: Remote theme (loaded from GitHub)
**Configuration**: `remote_theme: "mmistakes/minimal-mistakes@4.26.2"` in `_config.yml`
**Skin**: Dark theme

**Why Remote Theme**:
- No theme files in repository (cleaner codebase)
- Theme updates managed through version pinning
- Significantly reduces repository size
- Standard pattern for GitHub Pages

### 4.2 Layout Types Used

1. **splash** - Homepage with overlay image and feature cards
   - Used in `index.markdown`
   - Includes header actions and feature row sections

2. **home** - Blog listing page with pagination
   - Used in `_pages/blog.md`
   - Automatically lists posts, handles pagination

3. **single** - Single column layout with optional author profile
   - Used for all regular pages and posts
   - `author_profile: true` displays author info in sidebar
   - Default for posts and pages (set in defaults)

4. **page** - Basic page layout (used for 404.html)

### 4.3 Minimal Mistakes Features Enabled

Through `_config.yml` configuration:
- Author profile sidebar (name, bio, social links)
- Reading time estimate (posts)
- Social sharing buttons (posts)
- Related posts suggestions (posts)
- Pagination (5 posts per page)
- RSS feed generation
- XML sitemap

**Note**: No custom `_layouts/` or `_includes/` directories - all provided by remote theme.

---

## 5. Build and Development Workflow

### 5.1 Local Development Setup

**Prerequisites**:
- Ruby 3.4.7 (specified in `.ruby-version`)
- Bundler

**Installation**:
```bash
# Install dependencies
bundle install

# Serve locally with auto-rebuild
bundle exec jekyll serve

# Build static site
bundle exec jekyll build
```

**Result**: Site available at `http://localhost:4000`

### 5.2 Build Process

1. Jekyll reads `_config.yml`
2. Loads remote theme from GitHub
3. Processes Markdown files in `_posts/` and `_pages/`
4. Applies layouts and plugins
5. Generates HTML files in `_site/` directory
6. Applies plugins: feeds, sitemap, gist embeds, pagination

### 5.3 Build Artifacts

**`_site/` directory** (excluded from git via `.gitignore`):
- Complete static HTML site ready for deployment
- Regenerated on each build
- Never committed to git

### 5.4 Jekyll Cache

**Excluded from git**:
- `.jekyll-cache/` - Jekyll internal cache
- `.sass-cache/` - Sass compilation cache
- `.jekyll-metadata` - Post metadata

---

## 6. Custom Plugins or Features

### 6.1 Jekyll Plugins

All plugins managed through `Gemfile` and `_config.yml`:

1. **jekyll-feed** (v0.12+)
   - Auto-generates RSS/Atom feed at `/feed.xml`
   - Automatically included by github-pages gem

2. **jekyll-include-cache** (v0.2.1)
   - Caches include files for performance
   - Recommended by Minimal Mistakes theme

3. **jekyll-paginate** (v1.1.0)
   - Enables blog post pagination
   - Configured: 5 posts per page, `/page:num/` URL structure

4. **jekyll-sitemap** (v1.4.0)
   - Auto-generates XML sitemap for SEO
   - Located at `/sitemap.xml`

5. **jekyll-gist** (v1.5.0)
   - Enables embedding GitHub gists in posts/pages
   - Usage: `{% gist gist-id %}`

### 6.2 GitHub Pages Bundled Plugins

Via `github-pages` gem v232 (includes):
- `jekyll-avatar` - User avatars from GitHub profiles
- `jekyll-mentions` - @mentions support
- `jekyll-redirect-from` - Redirect manager
- `jekyll-relative-links` - Relative link conversion
- `jekyll-optional-front-matter` - Optional front matter support
- `jekyll-readme-index` - Auto-index from README
- And ~15 others

### 6.3 No Custom Plugins

**No custom Ruby plugins in repository** - all functionality through standard Jekyll plugins or theme configuration.

**Architecture Decision**: Rely on GitHub Pages supported plugins for compatibility.

---

## 7. Deployment Approach

### 7.1 GitHub Pages Deployment

**Deployment Type**: Automatic from GitHub repository
**Repository URL**: https://github.com/adamhadani/adamhadani.github.io
**Published Site**: https://adamhadani.github.io

### 7.2 How GitHub Pages Works with This Site

1. Push commits to `main` or `master` branch
2. GitHub detects Jekyll site (presence of `_config.yml`)
3. GitHub builds site automatically using github-pages gem v232
4. Generated HTML published to `https://adamhadani.github.io`

### 7.3 Build Environment

**Automatically Used by GitHub Pages**:
- Ruby version compatible with `github-pages` gem
- Bundler to resolve `Gemfile` dependencies
- Jekyll 4.3+ (via github-pages gem)
- All specified plugins

### 7.4 No Custom CI/CD

**Current Setup**: 
- No `.github/workflows/` directory
- No custom GitHub Actions
- Relies entirely on GitHub's built-in Jekyll build process

**Advantages**:
- Zero configuration needed
- GitHub Pages gem guarantees compatibility
- Automatic builds on push

**Limitations**:
- Can't use plugins outside GitHub Pages whitelist
- Build customization limited to `_config.yml`

### 7.5 Git Strategy

**Current Branch**: `claude/jekyll-website-updates-011CUoEjHQ3rVrfEpSKj8AxQ` (development branch)
**Main Branch**: `main` (deployment branch)

**Workflow**:
1. Develop on feature/task branches
2. Commit changes
3. Push to main for automatic deployment
4. GitHub Pages rebuilds and publishes

---

## 8. Architectural Decisions and Patterns

### 8.1 Configuration-Driven Approach

**Philosophy**: Minimize code, maximize configuration.

**Implementation**:
- Theme customization entirely in `_config.yml`
- Navigation structure in `_data/navigation.yml`
- No custom layouts/includes (using remote theme)
- No Ruby plugins

**Benefits**:
- Simple to maintain and modify
- Low risk of breaking changes
- Easy for non-developers to update content

### 8.2 Remote Theme Usage

**Decision**: Use Minimal Mistakes as remote theme instead of forking/copying.

**Benefits**:
- Smaller repository size
- Easy theme updates (change version in one place)
- Clear separation between content and presentation
- No theme maintenance burden

**Implications**:
- Can't directly modify theme templates (but can override via `_config.yml`)
- Must understand Minimal Mistakes theme variables

### 8.3 GitHub Pages Native Deployment

**Decision**: Use GitHub Pages built-in Jekyll build (not custom CI/CD).

**Benefits**:
- No additional configuration needed
- Automatic builds on push
- Zero deployment management
- Free hosting

**Constraints**:
- Limited to GitHub Pages supported plugins
- Can't use custom Ruby gems
- Build process not customizable

### 8.4 Minimal Content Currently

**Current State**:
- 1 example blog post (Welcome to Jekyll)
- 4 static pages (About, Blog, Contact, Music)
- Music page is a placeholder

**Future Expansion Pattern**:
- Add blog posts in `_posts/` with date-slug naming
- Extend music page content
- Leverage pagination automatically (5 posts per page)

---

## 9. Important Files Reference

| File | Purpose | Type |
|------|---------|------|
| `_config.yml` | Main site configuration | CONFIG |
| `_data/navigation.yml` | Menu structure | DATA |
| `index.markdown` | Homepage | PAGE |
| `_pages/*.md` | Static pages | CONTENT |
| `_posts/*.markdown` | Blog posts | CONTENT |
| `404.html` | Error page | TEMPLATE |
| `Gemfile` | Ruby dependencies | CONFIG |
| `.ruby-version` | Ruby version | CONFIG |
| `Gemfile.lock` | Locked dependencies | CONFIG |

---

## 10. Development Checklist

### When Adding a New Blog Post:

1. Create file: `_posts/YYYY-MM-DD-title-slug.markdown`
2. Add front matter (title, date, categories)
3. Write content in Markdown
4. Run `bundle exec jekyll serve` to preview
5. Commit and push to `main`

### When Modifying Navigation:

1. Edit `_data/navigation.yml`
2. Add/remove menu items with title and URL
3. Run locally to verify
4. Commit and push

### When Updating Site Description:

1. Edit `_config.yml`
2. Modify relevant fields (description, author.bio, etc.)
3. Changes take effect immediately on next build
4. Commit and push

### When Adding New Page:

1. Create file: `_pages/new-page.md`
2. Add front matter with title and permalink
3. Add link to `_data/navigation.yml` if it should appear in menu
4. Run locally to preview
5. Commit and push

---

## 11. Troubleshooting and Common Tasks

### Build locally fails:
```bash
# Clean and rebuild
rm -rf _site .jekyll-cache
bundle exec jekyll serve
```

### Bundler issues:
```bash
# Update gems
bundle update
# Or reinstall
rm Gemfile.lock
bundle install
```

### Changes not appearing:
- Restart `jekyll serve` (config changes require restart)
- Check `_site/` generated files
- Verify front matter YAML syntax

### Dependencies mismatch:
- Ensure Ruby 3.4.7 is active: `ruby -v`
- Run: `bundle install`
- If issues persist: `rm Gemfile.lock && bundle install`

---

## 12. Performance Considerations

**Current Optimizations**:
- Remote theme reduces static asset downloads
- `jekyll-include-cache` caches included files
- XML sitemap auto-generated for SEO
- RSS feed auto-generated for subscribers

**Build Time**:
- Currently fast (minimal content, remote theme)
- Will remain fast even with 100+ posts (pagination handles UI)

---

## 13. Security Considerations

**Current Security**:
- GitHub Pages provides HTTPS by default
- No user authentication needed
- Static HTML (no server-side vulnerabilities)
- Dependencies via bundler with lockfile

**Best Practices**:
- Keep Ruby and gems updated: `bundle update`
- No sensitive data in repository (emails are public-facing)
- Monitor GitHub security advisories

---

## 14. Future Expansion Points

**Easy to Add**:
- More blog posts (add to `_posts/`)
- More static pages (add to `_pages/`)
- Music projects content (expand music.md)
- Custom CSS (add to theme via `_config.yml`)
- Search functionality (Minimal Mistakes plugin)

**Would Require Change**:
- Different theme (change `remote_theme` version or switch theme)
- Custom Jekyll plugins (would need custom CI/CD)
- Database/dynamic content (incompatible with GitHub Pages)

---

## Summary

This is a **clean, minimal-complexity Jekyll blog** optimized for:
- **Easy maintenance** through configuration-driven design
- **Automatic deployment** via GitHub Pages
- **Professional appearance** through Minimal Mistakes theme
- **Growth-ready** with blog pagination and modular page structure

The architecture favors **simplicity and maintainability** over customization, making it ideal for a personal portfolio where content updates are more frequent than configuration changes.

**Key Takeaway**: To work with this codebase effectively, focus on content management (adding posts/pages) rather than theme customization. The build and deployment pipeline is fully automated - just commit and push.
