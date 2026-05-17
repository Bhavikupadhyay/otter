# Security Policy

## Supported Versions

Security updates and coordinated disclosure apply to the **current `main` branch** only. There are no tagged long-term support releases at this time.

## Reporting a Vulnerability

**Do not** open a public GitHub issue for security reports.

Email **bhavikupadhyay08@gmail.com** privately with a description of the issue, affected versions or commit if known, and steps to reproduce if possible. You should receive an initial response within **72 hours**.

## Scope

Otter is a C++ library with **manual CUDA memory management**. The following are in scope for private security reports:

- Memory-safety defects (out-of-bounds access, use-after-free, double-free)
- Concurrency defects affecting device or host memory (data races, unsynchronized access)
- Other issues that could lead to memory corruption, information disclosure, or denial of service in library or test code shipped in this repository

General correctness bugs (wrong numerical results without memory or thread safety implications) may be reported as regular issues unless you believe they indicate a deeper safety problem.
