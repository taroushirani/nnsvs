Development guide
=================

This page summarizes docs for developers of NNSVS. If you want to contribute to NNSVS itself, please check the document below.

Installation
------------

For development purposes, it is recommended to install full requirements with editiable mode  (``-e`` with pip) enabled:

.. code::

   pip install -e ".[dev,lint,test,docs]"

This allows your local changes available to your python environment without manually re-installing NNSVS.


Repository structure
---------------------

Here's the list of important components of the NNSVS repository:

- ``nnsvs``: The core Python library. Neural network implementations for SVS systems can be found here.
- ``recipes``: Recipes.  The recipes are written mostly in bash and YAML-style configs. Some recipes use small Python scripts.
- ``docs``: Documentation. It is written by `Sphinx <https://www.sphinx-doc.org/>`_.
- ``notebooks``: Jupyter notebooks. Notebooks are helpful for interactive debugging and development.
- ``utils``: Utility scripts that are used by the recipes.
- ``tests``: Tests

Python docstring style
----------------------

NNSVS follows the `Google's style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
If you write a docstrings for your new functinoality, please follow the same style.

Formatting and linting
----------------------

https://github.com/pfnet/pysen is used for formatting and linting. Please run the following commands when you make a PR.

Formatting
^^^^^^^^^^^

.. code::

   pysen run format

Linting
^^^^^^^

.. code::

   pysen run lint

Tests
-----

To prevent unintentional bugs, it is better to write tests as much as possible. If you propose a new function, please consdier to write tests.
You can run the tests by the following command:

.. code::

    pytest -v -s

Please make sure tests are all passing before making a PR.

``torch.load`` and ``weights_only``
------------------------------------

Since PyTorch 2.6, ``torch.load`` defaults to ``weights_only=True``, which restricts
unpickling to a small allow-list of "safe" types. NNSVS checkpoints pickle an
``omegaconf.listconfig.ListConfig`` alongside the ``state_dict``, so loading them with
the new default fails with ``UnpicklingError: Weights only load failed``.

All first-party ``torch.load`` call sites in this repository (``nnsvs/``, ``utils/``,
``recipes/_common/``, ``tests/``) explicitly pass ``weights_only=False`` to restore the
pre-2.6 behavior. This is intentional, but it comes with a trade-off worth knowing about
if you touch this code or write new checkpoint-loading scripts:

- ``weights_only=False`` disables PyTorch's unpickling allow-list and falls back to
  Python's regular ``pickle``, which can execute arbitrary code embedded in the file
  (see the `CVE-2025-32434 <https://nvd.nist.gov/vuln/detail/CVE-2025-32434>`_ class of
  issues, and PyTorch's own `serialization security notes
  <https://pytorch.org/docs/stable/notes/serialization.html#security>`_).
- This is considered acceptable here because these checkpoints are treated as
  first-party artifacts, produced by this repository's own training code. **Do not**
  load ``.pth`` files downloaded from an untrusted source (e.g. a third-party voice
  bank distribution) with the current code without inspecting them first.
- When adding a new ``torch.load`` call, pass ``weights_only=False`` explicitly for
  consistency, but do not silently reuse it to load files whose provenance you don't
  trust.

If this becomes a real distribution concern (e.g. sharing pre-trained checkpoints
publicly), the more robust long-term fix is to stop pickling ``ListConfig``/config
objects into the checkpoint at all -- save the config as a separate YAML file next to
the checkpoint (as ``SPSVS`` already does for model definitions) and keep the
``state_dict``-only checkpoint loadable under the default ``weights_only=True``.

Building docs locally
---------------------

Run the following command at the top of nnsvs directory:

.. code::

    sphinx-autobuild docs docs/_build/html/
