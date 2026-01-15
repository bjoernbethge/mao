# PyPI Publishing Setup

This document explains how to set up automated PyPI publishing for the mao package.

## Prerequisites

1. A PyPI account (create one at https://pypi.org/account/register/)
2. Admin access to this GitHub repository

## Setup Steps

### 1. Create PyPI API Token

1. Log in to your PyPI account at https://pypi.org/
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Fill in the token details:
   - **Token name**: `mao-github-actions` (or similar)
   - **Scope**: Choose "Project: mao" if the project exists, or "Entire account" for first-time setup
5. Click "Add token"
6. **IMPORTANT**: Copy the token immediately - it will only be shown once!
   - The token starts with `pypi-`

### 2. Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings → Secrets and variables → Actions**
3. Click "New repository secret"
4. Fill in the secret details:
   - **Name**: `PYPI_API_TOKEN`
   - **Value**: Paste the token you copied from PyPI (including the `pypi-` prefix)
5. Click "Add secret"

### 3. Publishing Releases

The workflow is now configured! You can publish to PyPI in two ways:

#### Option A: Create a GitHub Release (Recommended)

1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Fill in the release details:
   - **Tag**: `v0.1.0` (already created!)
   - **Release title**: `v0.1.0 - First Release`
   - **Description**: Add release notes
4. Click "Publish release"

The workflow will automatically trigger and publish to PyPI!

**Note**: The v0.1.0 tag has already been created locally. Push it with:
```bash
git push origin v0.1.0
```

Then create the GitHub release using this tag.

#### Option B: Manual Workflow Dispatch

1. Go to **Actions → Publish Package**
2. Click "Run workflow"
3. Enter the version number (e.g., `0.1.0`) or leave empty to use version from pyproject.toml
4. Click "Run workflow"

## Workflow Features

The publishing workflow includes:

- ✅ Automated package building with uv and Python build tools
- ✅ Package validation with twine check
- ✅ Publishing to PyPI with API token authentication
- ✅ Skip existing versions (won't fail if version already exists)
- ✅ Manual version override via workflow dispatch
- ✅ Triggered automatically on GitHub releases
- ✅ Support for uv package manager

## Current Status

- ✅ Workflow configured at `.github/workflows/publish.yml`
- ✅ Release tag v0.1.0 created locally
- ⏳ Waiting for GitHub release creation to trigger first publish
- ⏳ Waiting for PYPI_API_TOKEN secret to be configured

## Troubleshooting

### "Package already exists" Error

If you see this error, the version already exists on PyPI. The workflow uses `skip-existing: true`, so this should not cause failures. To publish a new version:

1. Update the version in `pyproject.toml`
2. Commit and push
3. Create a new release with the new version tag

### "Invalid credentials" Error

This means the `PYPI_API_TOKEN` secret is incorrect or expired:

1. Generate a new token on PyPI
2. Update the GitHub secret with the new token
3. Re-run the workflow

### Workflow Not Triggering

Make sure:

1. The workflow file exists at `.github/workflows/publish.yml`
2. You're creating a "Release" (not just a git tag)
3. The release is "Published" (not a draft)
4. The tag has been pushed to the remote repository

## Security Best Practices

- ✅ Never commit API tokens to the repository
- ✅ Use repository secrets for sensitive data
- ✅ Rotate tokens periodically
- ✅ Use scoped tokens (project-specific) when possible
- ✅ Enable two-factor authentication on your PyPI account

## Alternative: Trusted Publishing (Advanced)

For enhanced security, you can use PyPI's Trusted Publishing (no API tokens needed):

1. On PyPI, go to your project → Manage → Publishing
2. Add a new "trusted publisher":
   - **Owner**: Your GitHub username/org
   - **Repository**: mao
   - **Workflow**: publish.yml
   - **Environment**: (leave empty)
3. Remove the `password` line from the workflow (keep id-token: write permission)
4. The workflow will use OpenID Connect for authentication

This method is more secure as it doesn't require storing secrets!
