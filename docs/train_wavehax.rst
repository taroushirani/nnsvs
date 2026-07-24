How to train Wavehax vocoders
==============================

Please check :doc:`recipes` and :doc:`overview` first.

NNSVS supports `Wavehax <https://github.com/chomeyama/wavehax>`_, an STFT/ConvNeXt-based neural
vocoder, as a fourth ``vocoder_type`` alongside ``pwg``, ``usfgan`` (which also covers SiFiGAN),
and plain WORLD.

.. warning::

    Unlike PWG/uSFGAN/SiFiGAN, Wavehax has no official singing-voice-synthesis recipe or
    checkpoint upstream; only a JVS (speech) recipe exists. The configs and 48kHz
    ``sample_rate``/``n_fft`` values shipped with NNSVS are project-local tuning choices made
    without an upstream 48kHz precedent, not validated defaults. A community member has since
    reported a successful end-to-end run on real singing data (see `Pre-trained base checkpoint`_
    below), but this is a single external report without published objective/subjective metrics
    or audio samples — not a benchmark run or verified by the NNSVS project itself.

Install Wavehax
----------------

NNSVS installs Wavehax from an unofficial nnsvs-specific fork,
`taroushirani/wavehax <https://github.com/taroushirani/wavehax/tree/nnsvs>`_ (``nnsvs`` branch),
rather than upstream ``chomeyama/wavehax``. The fork's only functional difference is that it
writes/reads checkpoints directly under ``out_dir`` instead of an ``out_dir/checkpoints/``
subdirectory (see the note in Stage 14 below).

.. code::

    pip install git+https://github.com/taroushirani/wavehax@nnsvs --no-build-isolation

Pre-trained base checkpoint
-----------------------------

A community member has published a Wavehax base checkpoint trained on 5 singing-voice
databases for 100k steps: `Canon/nnsvs-wavehax-base <https://huggingface.co/Canon/nnsvs-wavehax-base>`_
(MIT-licensed, free for commercial use and redistribution).

Per their report, fine-tuning this base on an unseen singer for 2,000 steps produced synthesis
(driven by acoustic-model-predicted features, not ground truth) judged comparable in quality to
WORLD vocoding. This was relayed via community correspondence rather than published as a
benchmark with metrics or audio samples, and Wavehax's output is non-deterministic, so results
may vary run to run — treat it as a promising starting point to fine-tune from and evaluate
yourself, not a verified guarantee.

Fine-tuning from a pre-trained checkpoint follows the same ``load_optimizer: false`` resume
convention documented in :doc:`tips` for other model types.

Stage 9: Prepare features for neural vocoders
-----------------------------------------------

Same as PWG/uSFGAN/SiFiGAN (see :doc:`train_vocoders`).

Convert data to Wavehax's format
---------------------------------

Wavehax's dataloader (``wavehax.datasets.AudioFeatDataset``) expects HDF5 features and a
per-feature-name scaler dict, which is a different convention from uSFGAN's ``aux_feats``-based
format. ``utils/nnsvs2wavehax.py`` converts nnsvs's pre-processed features accordingly:

.. code-block:: bash

    python utils/nnsvs2wavehax.py config.yaml dump_wavehax --feature_type world  # or melf0

This writes ``dump_wavehax/{stats,scp,hdf5,wav}``, analogous to ``dump_usfgan/``.
``recipes/_common/spsvs/train_wavehax.sh`` runs this conversion automatically before training.

Config tree
-----------

Configs live under ``recipes/_common/conf/jp_dev_48k_nodyn/train_wavehax/``:

.. code-block::

    train_wavehax/
    ├── train.yaml
    ├── generator/
    │   ├── nnsvs_world_wavehax_sr48k.yaml
    │   └── nnsvs_melf0_wavehax_sr48k.yaml
    ├── discriminator/
    │   └── nnsvs_univnet.yaml
    ├── train/
    │   ├── nnsvs_wavehax.yaml
    │   └── nnsvs_wavehax_test.yaml
    └── data/
        ├── nnsvs_world_sr48k.yaml
        ├── nnsvs_world_sr48k_test.yaml
        ├── nnsvs_melf0_sr48k.yaml
        └── nnsvs_melf0_sr48k_test.yaml

The ``data/*.yaml`` files use Wavehax's own key names (``train_audio``/``train_feat``/``feat_names``/
``use_continuous_f0``/``batch_max_length``/...), which are **not** the same keys used by
uSFGAN/SiFiGAN's own ``data/*.yaml`` (``sampling_rate``/``aux_feats``/``dense_factors``/...).

Stage 14: Training Wavehax vocoder
------------------------------------

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 14 --stop-stage 14 \
        --vocoder-model nnsvs_world_wavehax_sr48k

Checkpoints are written under ``exp/${speaker name}/${vocoder config name}`` directly, the same
flat layout PWG/uSFGAN/SiFiGAN use. This matches the nnsvs fork of Wavehax installed above
(``taroushirani/wavehax@nnsvs``), which removed the ``checkpoints/`` subdirectory that upstream
``chomeyama/wavehax`` still uses. ``recipes/_common/spsvs/train_wavehax.sh``, ``pack_model.sh``,
and ``run_common_steps_dev.sh``'s stage 99 all check the flat layout first, falling back to the
legacy ``out_dir/checkpoints/`` subdirectory only for checkpoints produced by upstream
``chomeyama/wavehax``.

How to use the packed model with the trained vocoder?
---------------------------------------------------------

Please specify ``vocoder_type="wavehax"`` with the :doc:`modules/svs` module. An example:

.. code-block::

    import numpy as np
    import pysinsy
    from nnmnkwii.io import hts
    from nnsvs.svs import SPSVS
    from nnsvs.util import example_xml_file

    model_dir = "/path/to/your/packed/model_dir"
    engine = SPSVS(model_dir)

    contexts = pysinsy.extract_fullcontext(example_xml_file(key="get_over"))
    labels = hts.HTSLabelFile.create_from_contexts(contexts)

    wav, sr = engine.svs(labels, vocoder_type="wavehax")
