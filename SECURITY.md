# Security policy

## Supported versions

Security fixes are applied to the latest minor release. When a vulnerability is
fixed, the patch lands in the next release and is noted in the changelog.

| Version | Supported |
| --- | --- |
| 0.3.x | yes |
| < 0.3 | no |

## Reporting a vulnerability

Please do not open a public issue for a security problem. Instead, report it
privately so it can be assessed and fixed before being disclosed.

- Use GitHub's private vulnerability reporting feature on this repository, or
- Contact the maintainer directly.

Reports are acknowledged within a few days. After a fix is released, a public
advisory and CVE (where applicable) will be issued.

## Scope

diapason is an evaluation harness: it reads manifests and prediction files,
computes metrics, and writes reports. It does not execute model code and its
HTTP adapter only makes outbound requests to endpoints you configure. As always,
treat manifests and prediction files from untrusted sources as untrusted input.
