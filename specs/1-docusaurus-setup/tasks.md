# Tasks: Docusaurus Project Setup

## Feature: Docusaurus Project Setup

This document outlines the detailed, dependency-ordered tasks for setting up the Docusaurus 3.x project with GitHub Pages deployment.

## 1. Task Breakdown

### Phase 1: Local Setup and Verification
- [X] T001 Initialize Docusaurus project locally using the classic template and npm
- [X] T002 Configure `docusaurus.config.js` with `projectName: 'physical-ai-robotics-book'`, `organizationName: 'fun33333'`, `baseUrl: '/physical-ai-robotics-book/'`, `trailingSlash: true`, and initial sidebar structure for modules and hardware guide
- [X] T003 Create placeholder content structure in `docs/`:
  - `docs/intro.md`
  - `docs/module-1/.gitkeep`
  - `docs/module-2/.gitkeep`
  - `docs/module-3/.gitkeep`
  - `docs/module-4/.gitkeep`
  - `docs/hardware/.gitkeep`
  - `docs/weekly-breakdown/.gitkeep`
- [X] T004 Local Build Test: Run `npm run build` to verify site builds without warnings or errors
- [X] T005 Local Serve Test: Run `npm run serve` to confirm site serves locally and basic navigation works

### Phase 2: GitHub Repository and Initial Push
- [ ] T006 Create a new public GitHub repository named `physical-ai-robotics-book` under `fun33333`
- [ ] T007 Add a comprehensive `.gitignore` file to exclude `node_modules`, `build/`, `.docusaurus/`, etc.
- [ ] T008 Initial Commit and Push: Commit the initialized Docusaurus project and push to the new GitHub repository

### Phase 3: GitHub Actions Deployment Setup
- [ ] T009 Create `.github/workflows/deploy.yml` for GitHub Actions workflow
- [ ] T010 Configure `deploy.yml` to trigger on pushes to `main`, use Node.js 18+, build Docusaurus, and deploy `build/` directory to GitHub Pages
- [ ] T011 Commit Workflow: Commit and push `deploy.yml` to the `main` branch

### Phase 4: End-to-End Deployment Verification
- [ ] T012 Monitor GitHub Actions workflow run for successful completion
- [ ] T013 Access GitHub Pages Site: Verify site is accessible at `https://fun33333.github.io/physical-ai-robotics-book/`
- [ ] T014 Test Basic Navigation: Confirm internal navigation works on the deployed site

## 2. Dependency Graph

All tasks within a phase are sequential. Phases are sequential:

Phase 1 (T001-T005) -> Phase 2 (T006-T008) -> Phase 3 (T009-T011) -> Phase 4 (T012-T014)

## 3. Parallel Execution Opportunities

None identified at this stage, as tasks are foundational and sequential for initial setup.

## 4. Independent Test Criteria (per User Scenario)

### User Scenario 1: Initial Project Setup
- **Criteria**: Docusaurus 3.x is initialized, `.gitignore` is correct, GitHub Actions workflow file exists, basic site structure is present, sidebar is configured, and local navigation works without errors.

### User Scenario 2: Site Deployment and Accessibility
- **Criteria**: GitHub Actions workflow successfully deploys, and the site is publicly accessible via the GitHub Pages URL with working navigation.

## 5. Suggested MVP Scope

The entire set of tasks outlined in this document constitutes the MVP for the Docusaurus project setup, ensuring a fully functional and deployable site infrastructure.

## 6. Implementation Strategy

Adopt an incremental delivery approach, focusing on completing each phase and verifying its outcomes before proceeding to the next. This ensures a stable foundation and early detection of any configuration issues.
