"""Command line interface for diapason.

The CLI is a thin layer over the Python API: it parses arguments, loads the
suite config, and delegates to :mod:`diapason.runner` and
:mod:`diapason.report`.
"""

from __future__ import annotations

import click

from diapason.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="diapason")
def main() -> None:
    """Reproducible evaluation harness for audio language models."""


if __name__ == "__main__":
    main()
