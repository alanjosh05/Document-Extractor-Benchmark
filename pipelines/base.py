"""
Base pipeline interface for the PDF extraction benchmark framework.

Every concrete pipeline must:
  - Set a class-level ``name`` attribute (str).
  - Implement :py:meth:`run`, which accepts a PDF path and returns a
    standardised result dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BasePipeline(ABC):
    """Abstract base for all extraction pipelines.

    Subclasses must define the class attribute ``name`` and implement
    :py:meth:`run`.
    """

    #: Unique pipeline identifier – used for output folder names and
    #: benchmark reporting.
    name: str = ""

    @abstractmethod
    def run(self, pdf_path: str) -> dict:
        """Convert *pdf_path* to Markdown and return a result dict.

        Parameters
        ----------
        pdf_path:
            Absolute or relative path to the source PDF file.

        Returns
        -------
        dict
            Must contain at minimum:

            .. code-block:: python

                {
                    "success":       bool,
                    "pipeline":      str,   # pipeline name
                    "pdf":           str,   # basename of the input PDF
                    "markdown_path": str,   # path where .md was saved
                    "time_sec":      float, # wall-clock seconds
                }

            On failure, replace ``markdown_path`` / ``time_sec`` with
            an ``"error"`` key containing the exception message.
        """

    # ------------------------------------------------------------------
    # Helpers available to all pipelines
    # ------------------------------------------------------------------

    def _output_dir(self) -> Path:
        """Return (and create) the pipeline-specific output directory."""
        path = Path("outputs") / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _temp_dir(self) -> Path:
        """Return (and create) a pipeline-specific temp directory."""
        path = Path("temp") / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
