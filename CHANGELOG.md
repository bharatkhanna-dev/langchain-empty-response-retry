# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-25

### Added

- `EmptyResponseRetryMiddleware`: retries with a corrective nudge when the model
  returns no text and no tool calls.
- `is_empty_ai_message` and `EmptyModelResponseError`.
