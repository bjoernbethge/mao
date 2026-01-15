# PyPI Publishing Setup for mao

This document explains how to configure PyPI publishing for the mao package using GitHub Actions and trusted publishing.

## Important: First Release

The **mao** package has NEVER been released to PyPI! This PR updates the workflow to use trusted publishing and provides instructions for creating the first v0.1.0 release.

## Prerequisites

1. A PyPI account at https://pypi.org
2. Admin access to the GitHub repository

## Setup Instructions

### 1. Configure PyPI Trusted Publishing

PyPI's trusted publishing eliminates the need for API tokens by using OpenID Connect (OIDC) to verify that the package is being published from the correct GitHub repository.

1. Go to https://pypi.org and log in
2. Navigate to your account settings
3. Go to "Publishing" section
4. Click "Add a new pending publisher"
5. Fill in the details:
   - **PyPI Project Name**: `mao`
   - **Owner**: `bjoernbethge`
   - **Repository name**: `mao`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave empty)
6. Click "Add"

**Note**: For the first publish, you need to create the project as a "pending publisher" before the package exists on PyPI.

### 2. Create Your First Release (v0.1.0)

Once trusted publishing is configured, create the first release:

1. Verify the version in `pyproject.toml` is set to `0.1.0`
2. Go to https://github.com/bjoernbethge/mao/releases/new
3. Click "Choose a tag" and create a new tag: `v0.1.0`
4. Set the release title: "mao v0.1.0 - Initial Release"
5. Add release notes describing the package features:
   ```markdown
   # mao v0.1.0 - Initial Release

   First public release of MCP Agent Orchestra - A modern framework for orchestrating AI agents.

   ## Features
   - FastAPI-based agent orchestration
   - Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
   - LangChain and LangGraph integration
   - Vector storage with Qdrant
   - DuckDB for data management
   - Docker support
   ```
6. Click "Publish release"

The GitHub Action will automatically:
- Build the package
- Run quality checks
- Publish to PyPI using trusted publishing

### 3. Subsequent Releases

For future releases:

1. Update the version in `pyproject.toml` (e.g., `0.2.0`)
2. Commit the change
3. Create a new GitHub release with the matching tag (e.g., `v0.2.0`)

### 4. Manual Publishing (Optional)

You can also trigger publishing manually without creating a release:

1. Go to https://github.com/bjoernbethge/mao/actions/workflows/publish.yml
2. Click "Run workflow"
3. Select the branch to publish from
4. Optionally specify a version override
5. Click "Run workflow"

## Verifying the Package

After publishing, verify your package at:
- https://pypi.org/project/mao/

Install it using:
```bash
pip install mao
```

## Changes in This PR

This PR updates the publishing workflow to use trusted publishing:

- Removed API token requirement (more secure!)
- Changed trigger from `created` to `published` (GitHub best practice)
- Reduced timeout from 30 to 15 minutes (more appropriate for Python packages)
- Updated documentation to reflect trusted publishing setup

### Migration from Token-Based Publishing

If you previously had `PYPI_API_TOKEN` in GitHub secrets, you can now:
1. Delete the secret (it's no longer needed)
2. Follow the trusted publishing setup above
3. Trusted publishing is more secure and doesn't require token management

## Troubleshooting

### First Publish Fails

If the first publish fails with "project does not exist", make sure you:
1. Created the pending publisher on PyPI first
2. Used the exact workflow filename (`publish.yml`)
3. The repository owner and name match exactly

### Permission Denied

If you get permission errors:
1. Verify the trusted publisher is configured correctly on PyPI
2. Ensure the workflow has `id-token: write` permissions (already configured)
3. Check that the repository owner matches the PyPI project owner

## Version Management

The package version is defined in `pyproject.toml`:

```toml
[project]
name = "mao"
version = "0.1.0"
```

Before creating a new release:
1. Update the version in `pyproject.toml`
2. Commit the change
3. Create a new release with a matching tag (e.g., `v0.2.0`)

## Security

Trusted publishing is more secure than API tokens because:
- No long-lived credentials stored in GitHub secrets
- Automatic verification of publisher identity
- Per-repository and per-workflow restrictions
- Automatic token rotation

## Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions: Publishing Python Packages](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries)
