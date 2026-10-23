"""Shared tabular exports for local and global equilibrium-policy evidence."""
from __future__ import annotations

from typing import Any

import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


class _PolicyReport:
    """Delegate formatting to the package's shared escaping/report layer."""

    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Return the same independent table as :meth:`to_frame`."""
        return self.to_frame()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the result table as Markdown."""
        return df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the result table through the shared LaTeX formatter."""
        return df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the result table through the shared Typst formatter."""
        return df_to_typst(self.to_frame(), **kwargs)
