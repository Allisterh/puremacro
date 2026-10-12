"""Local language models behind one entry point (being assembled).

For now this package holds the completion cache, :mod:`puremacro.llm.cache`: a
SQLite file of completions keyed by the canonical model name, the prompt and the
sampling parameters, so that a cache built on one machine answers the same call
on any other. The engines, the model ladder and the ``complete`` / ``chat``
facade are added to this package next.

Importing the package imports nothing but this docstring.
"""
