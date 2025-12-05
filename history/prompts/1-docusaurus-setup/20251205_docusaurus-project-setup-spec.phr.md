---
PHR_ID: 20251205120001
STAGE: spec
TITLE: Docusaurus project setup spec
DATE: 2025-12-05
FEATURE: 1-docusaurus-setup
---

## Prompt Text

```text
Docusaurus project setup with GitHub Pages deployment for Physical AI book

Target audience: Myself (developer setting up the infrastructure)
Focus: Working Docusaurus site with proper folder structure, ready for content addition

Success criteria:
- Docusaurus 3.x initialized with custom configuration
- GitHub repository created with proper .gitignore
- GitHub Actions workflow configured for automatic deployment to GitHub Pages
- Basic site structure with placeholder homepage
- Site accessible via GitHub Pages URL (https://[username].github.io/[repo-name])
- Sidebar configured for 4 main modules + hardware section
- Navigation works without errors

Deliverables:
- Initialized Docusaurus project with custom docusaurus.config.js
- README.md with setup instructions
- .github/workflows/deploy.yml for CI/CD
- docs/ folder structure:
  * intro.md (landing page)
  * module-1/ (ROS 2)
  * module-2/ (Gazebo & Unity)
  * module-3/ (NVIDIA Isaac)
  * module-4/ (VLA)
  * hardware/ (requirements & setup)
  * weekly-breakdown/ (13 weeks)
- Working deployment to GitHub Pages

Constraints:
- Use Docusaurus 3.x (latest stable)
- Node.js 18+ required
- Deployment must be automated (no manual builds)
- Site must build without warnings or errors
- Use classic theme with default styling (customize later)

Not building (defer to later phases):
- Actual module content (just create empty markdown files)
- Custom styling or branding
- Search functionality optimization
- Interactive code examples
- Hardware comparison calculators

Technical requirements:
- Package manager: npm or yarn
- Base URL configured for GitHub Pages
- Organization URL set correctly
- Trailing slash handling configured
- Build output: static files in build/ directory
```

## Response Text

The feature specification (specs/1-docusaurus-setup/spec.md) and its quality checklist (specs/1-docusaurus-setup/checklists/requirements.md) have been successfully created and updated.

All clarifications have been resolved, and the specification is now ready for the planning phase.

**Branch Name:** 1-docusaurus-setup
**Specification File:** specs/1-docusaurus-setup/spec.md

You can now proceed with `/sp.plan` for this feature.
