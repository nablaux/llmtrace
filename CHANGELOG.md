# CHANGELOG


## v0.2.0 (2026-03-01)

### Continuous Integration

- Add semantic release for automated versioning and publishing
  ([`5d49764`](https://github.com/nablaux/llmtrace/commit/5d4976414be8884516e079577eea5643b0f99473))

Replace manual `make release` workflow with python-semantic-release. Pushes to main with feat:/fix:
  commits now auto-bump version, tag, create a GitHub release, and publish to PyPI via OIDC.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Add semantic release for automated versioning and publishing
  ([`f2e9c1b`](https://github.com/nablaux/llmtrace/commit/f2e9c1b33a4dca0b2db2d37a5ad4007fedb583ee))

Automate version bumping, tagging, and PyPI publishing on push to main using
  python-semantic-release. Replaces manual make release workflow.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.1.0 (2026-03-01)

### Bug Fixes

- Linting + setup pre-commit hooks
  ([`9cb421d`](https://github.com/nablaux/llmtrace/commit/9cb421d4a19b1a63b19c59de4ccdb6aca9fd8e72))

- Linting + setup pre-commit hooks
  ([`76bb838`](https://github.com/nablaux/llmtrace/commit/76bb83829ded30e47451450742a3b63f5dbd1a0e))

### Chores

- Add codecov
  ([`d60b05a`](https://github.com/nablaux/llmtrace/commit/d60b05a09eae028c4e5270d8a1467ce9c420d03d))

- Add codeowners file
  ([`fc5b967`](https://github.com/nablaux/llmtrace/commit/fc5b96787720eb7797d2838a8be81d658d6eb172))

- Change package name to llm-trace for pypi collision
  ([`0986899`](https://github.com/nablaux/llmtrace/commit/0986899d096770c7935d152eee176c634d219ff8))

- Change package name to llmtrace-sdk for pypi collision
  ([`518bfd9`](https://github.com/nablaux/llmtrace/commit/518bfd91f668886fe646d9fb0d5a9a67a02e6646))

- Update examples with real datadog + otel local setup example
  ([`e413ce9`](https://github.com/nablaux/llmtrace/commit/e413ce98b51c019875bf3d935d81943b484c8bc0))

- Update examples with real datadog + otel local setup example
  ([`115bcce`](https://github.com/nablaux/llmtrace/commit/115bcce31a182d7c3343df0267c1ca8f735e0560))

### Features

- First version of library - initial commit
  ([`73a37c7`](https://github.com/nablaux/llmtrace/commit/73a37c7f6e3adb4d90fccc4e81d99cf8029f6140))
