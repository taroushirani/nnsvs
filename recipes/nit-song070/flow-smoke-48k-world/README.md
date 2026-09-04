# nit-song070 flow matching smoke test

A short sanity-check run for the flow matching-based acoustic model
(`nnsvs.techsinger.FlowMatching` / `FlowMatchingF0`). It is identical to
`../flow-48k-world` except that it uses `train=flow_smoke`, which trains the
acoustic model for 3 epochs, saves a checkpoint every epoch, disables AMP and
evaluates a single utterance per epoch.

The model config is exactly the same as the one used for the real training run,
so this is a wiring test (model instantiation, loss computation, evaluation and
checkpointing), not a quality test.

## Usage

Feature extraction is identical to `../dev-48k-world` and `../flow-48k-world`,
so the dumped features can be shared:

```bash
ln -s ../dev-48k-world/data data
ln -s ../dev-48k-world/dump dump
```

Otherwise, run stage 0 to 3 as usual. Then train the acoustic model only:

```bash
./run.sh --stage 5 --stop-stage 5
```

On a CPU-only machine the per-epoch evaluation is the bottleneck (each
utterance requires K_step ODE steps; measured at roughly 50 seconds for 7.5
seconds of audio with K_step=100). K_step only affects inference, so it can be
lowered for the smoke test by editing the model config or by appending hydra
overrides to `nnsvs/bin/train_acoustic.py`:

```
model.netG.lf0_model.K_step=20 model.netG.mgc_model.K_step=20 model.netG.bap_model.K_step=20
```

## What to check

- The loss is finite and decreases.
- `params_without_grad` is 0 from the second iteration onwards. It is non-zero
  on the very first iteration because `DiffNet.output_projection.weight` is
  zero-initialized, which is expected (`GaussianDiffusion` behaves the same way).
- In TensorBoard, the predicted log-F0 stays close to the log-F0 of the musical
  score. `FlowMatchingF0` predicts the residual against the score and clips the
  sampled endpoint to +-`clip_cent` (600 cent by default), so a prediction far
  away from the score indicates a configuration error (typically a wrong
  `in_lf0_idx`, or log-F0 statistics that were not injected).
