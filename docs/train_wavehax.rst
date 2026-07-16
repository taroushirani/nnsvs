How to train Wavehax vocoders
==============================

Please check :doc:`recipes` and :doc:`overview` first.

NNSVS supports `Wavehax <https://github.com/chomeyama/wavehax>`_, an STFT/ConvNeXt-based neural
vocoder, as a fourth ``vocoder_type`` alongside ``pwg``, ``usfgan`` (which also covers SiFiGAN),
and plain WORLD.

.. warning::

    Unlike PWG/uSFGAN/SiFiGAN, Wavehax has no official singing-voice-synthesis recipe, checkpoint,
    or nnsvs-specific fork. Only a JVS (speech) recipe exists upstream. The configs and 48kHz
    ``sample_rate``/``n_fft`` values shipped with NNSVS are project-local tuning choices made
    without an upstream 48kHz precedent, not validated defaults. Training convergence and audio
    quality on singing data have not been verified end-to-end; only the nnsvs-side wiring
    (config parsing, data conversion, inference-time integration) has been tested.

Install Wavehax
----------------

.. code::

    pip install git+https://github.com/chomeyama/wavehax --no-build-isolation

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

Checkpoints are written under ``exp/${speaker name}/${vocoder config name}/checkpoints/``
(note the extra ``checkpoints/`` subdirectory, unlike PWG/uSFGAN/SiFiGAN which write checkpoints
flat in the output directory). ``pack_model.sh`` and ``run_common_steps_dev.sh``'s stage 99 both
account for this when locating the latest checkpoint.

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
