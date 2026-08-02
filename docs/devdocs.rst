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
unpickling to a small allow-list of "safe" types.

``save_checkpoint`` (see ``nnsvs/train_util.py``) itself only saves plain tensors:
``model.state_dict()``, ``optimizer.state_dict()``, and ``lr_scheduler.state_dict()``.
The training config as a whole is *not* pickled into the checkpoint. However,
optimizers used to be instantiated as ``optimizer_class(model.parameters(),
**optim_config.optimizer.params)``, where ``optim_config.optimizer.params`` is an
OmegaConf ``DictConfig``. Unpacking it with ``**`` does not convert nested containers to
plain Python types, so list-valued hyperparameters defined in YAML (e.g. Adam's
``betas: [0.9, 0.999]``, as used in several recipe configs) could be passed through as
``omegaconf.listconfig.ListConfig`` objects, which end up stored in the optimizer's
``param_groups`` and therefore get pickled into ``optimizer_state`` when the checkpoint
is saved. Loading such a checkpoint under ``weights_only=True`` fails with
``UnpicklingError: Weights only load failed`` because ``ListConfig`` is not on the
allow-list.

**This has been fixed at the source.** ``_instantiate_optim`` and ``setup`` in
``nnsvs/train_util.py`` now convert ``optim_config.optimizer.params`` and
``optim_config.lr_scheduler.params`` to plain Python types via
``OmegaConf.to_container(..., resolve=True)`` before unpacking them into the
optimizer/scheduler constructors. So for checkpoints saved by current code,
``optimizer_state`` never contains a ``ListConfig`` in the first place (verified: a real
recipe config's ``betas`` now ends up as a plain ``tuple`` in ``param_groups``, and the
resulting checkpoint loads fine under ``weights_only=True``). The remaining risk is
**old checkpoints saved before this fix**, which may still carry a raw ``ListConfig`` in
``optimizer_state`` -- see the migration tools below.

**``add_safe_globals`` cannot fix this for checkpoints that already contain
``ListConfig``.** It is tempting to reach for
``torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])`` to allow
just this one type back in under ``weights_only=True``. This was tried and does not
work: PyTorch's ``weights_only`` unpickler (``torch/_weights_only_unpickler.py``)
separately hard-codes that the pickle ``SETITEM``/``SETITEMS`` opcodes may only target a
``dict``, ``collections.OrderedDict``, or ``collections.Counter`` instance, regardless of
``add_safe_globals``. ``ListConfig`` internally uses a ``collections.defaultdict``, which
trips this check and fails with::

    Can only SETITEM for dict, collections.OrderedDict, collections.Counter, but got
    <class 'collections.defaultdict'>

This was confirmed against real checkpoints in this repository (58 files under
``recipes/`` with a ``ListConfig``-containing ``optimizer_state``). So there is no way to
make an existing ``ListConfig``-containing checkpoint loadable under
``weights_only=True`` without first removing the ``ListConfig`` from it.

How this repo actually handles it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every first-party ``torch.load`` call site in this repository (``nnsvs/``, ``utils/``,
``recipes/_common/``, ``tests/``) passes ``weights_only=True`` explicitly, restoring
PyTorch 2.6's own default. This includes packed/distributed inference model loading
(``SPSVS`` in ``nnsvs/svs.py``, and the shared ``load_vocoder`` in ``nnsvs/util.py``) as
well as raw training checkpoint loading (``train_util.py``'s resume path,
``nnsvs/bin/synthesis.py`` / ``generate.py`` / ``gen_static_features.py``,
``clean_checkpoint_state.py``, and the ``tests/test_compat.py`` fixture).

The consequence is that any checkpoint whose ``optimizer_state`` still contains an
OmegaConf ``ListConfig`` (see above) -- i.e. any raw checkpoint saved before the
``betas``-leak was fixed at the source, or before being run through the migration tool
below -- will fail to load anywhere in this codebase, including
``clean_checkpoint_state.py`` and resuming training. This is intentional: it keeps every
``torch.load`` call site on PyTorch's restricted unpickler, so a tampered or malicious
``.pth`` file (whether a packed model or a raw checkpoint) can no longer execute
arbitrary code on load. Old checkpoints must be migrated first (see below) before they
can be resumed from, packed, or otherwise loaded again.

Migrating old raw checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``utils/migrate_checkpoint_weights_only.py`` converts a single raw checkpoint's embedded
``ListConfig``/``DictConfig`` objects into plain ``list``/``dict`` via
``OmegaConf.to_container(..., resolve=True)`` and re-saves it, verifying the result loads
under ``weights_only=True``:

.. code::

    python utils/migrate_checkpoint_weights_only.py exp/foo/latest.pth exp/foo/latest_migrated.pth

For migrating a whole ``pretrained_expdir`` (the directory-per-model layout that
``train_*.sh`` scripts expect when ``pretrained_expdir`` is set for fine-tuning/resume),
use ``utils/migrate_expdir_weights_only.py`` instead. It recursively finds every
``latest*.pth`` under the input directory (``latest.pth`` and, for GAN training,
``latest_D.pth`` -- the only files the resume path actually reads) and migrates each one
into the output directory with the same relative layout; files that already load fine
under ``weights_only=True`` are copied as-is. Other checkpoints in the same directories
(``best_loss.pth``, ``epochNNNN.pth``, ...) are intentionally left untouched since resume
never reads them:

.. code::

    python utils/migrate_expdir_weights_only.py /path/to/old_pretrained_expdir /path/to/new_pretrained_expdir

Both scripts are one-off, manually-run tools -- neither is wired into any recipe stage.
Because old checkpoints are, by definition, not yet safe to load under
``weights_only=True``, ``migrate_checkpoint_weights_only.py`` is the one intentional
exception left in the codebase that still loads a checkpoint with
``weights_only=False``. Only run it against checkpoints whose provenance you trust (e.g.
your own past training runs), and use its output -- not the original file -- for
resuming training, packing, or anything else in this codebase going forward.

Building docs locally
---------------------

Run the following command at the top of nnsvs directory:

.. code::

    sphinx-autobuild docs docs/_build/html/
