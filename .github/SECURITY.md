# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `main` branch | Yes |
| Tagged releases | Yes |
| Older commits | No |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue in the engine, the build system, or the Docker image, please report it privately:

1. Go to the **Security** tab of this repository on GitHub.
2. Select **Report a vulnerability** (GitHub Private Security Advisory).
3. Provide a clear description of the issue, steps to reproduce, and potential impact.

We will acknowledge the report within 7 days and aim to release a fix within 30 days depending on severity.

## Scope

Security reports are relevant for:

- Memory safety issues in C++ code (buffer overflows, use-after-free, etc.)
- CUDA kernel correctness issues that could cause silent data corruption during training
- Dockerfile vulnerabilities that could compromise the build environment
- Supply chain issues (dependency tampering)

Out of scope: model output quality, inference accuracy, or training convergence issues — these are not security vulnerabilities.

---

*Ghassan AI Team — 2024-2026*
