# fiery-things

Template repository for `fiery` PyTorch projects.

This template includes:

- a `pyproject.toml` configured for a `fiery.things` package
- a `fiery` namespace package under `src/`
- reusable GitHub Actions for linting, testing, and publishing (testing the
  original torch/python matrix via pip from the PyTorch wheel index)

The workflow wrappers intentionally track `bagofseeds/actions@main` so
template-generated repositories inherit shared CI updates without manually
refreshing pinned workflow SHAs.

When using the template, replace `things` with your project-specific package
name.
