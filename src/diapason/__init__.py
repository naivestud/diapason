"""diapason: reproducible evaluation harness for audio language models.

diapason scores model outputs on three task families — speech recognition
(ASR), sound event detection (SED) and audio question answering (AQA) — and
turns the results into shareable reports.
"""

from diapason.version import __version__

__all__ = ["__version__"]
