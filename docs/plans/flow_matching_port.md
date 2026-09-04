# TechSinger Flow Matching を NNSVS v5 音響モデルへ移植する実装仕様

作成日: 2026-09-03 / 更新: 2026-09-04 / 対象ブランチ: `dev20260812` (base commit `b9d6e2c2`)
ステータス: **実装済み (未学習)**。lk plan エントリ id 15 (user scope) と対応。

## 実装済みファイル (2026-09-04)

| ファイル | 内容 |
|---|---|
| `nnsvs/techsinger/flow.py` | `FlowMatching`, `FlowMatchingF0` |
| `nnsvs/techsinger/__init__.py` | 上記 2 クラスを export |
| `nnsvs/techsinger/README.md` | TechSinger 由来である旨の出典表記 |
| `tests/test_flow.py` | 単体 + multistream 統合テスト (18 件) |
| `recipes/_common/conf/jp_dev_48k_nodyn/train_acoustic/model/acoustic_nnsvs_world_multi_flow_f0_flow_mgcbap.yaml` | model 設定 |
| `recipes/_common/conf/jp_dev_48k_nodyn/train_acoustic/train/flow.yaml` | train 設定 |

既存コードへの変更は不要 (新規ディレクトリ `nnsvs/techsinger/` の追加のみ)。

**設計から変えた点** (以下の本文は実装に合わせて更新済み):

1. 推論時の clamp 境界を `self._clip_bounds` というインスタンス状態で持たず、
   `FlowMatching._sample(cond, lengths, clip_bounds=None)` の引数として渡す。
   `FlowMatchingF0.inference` は `_sample` を直接呼ぶ。副作用のない実装にするため。
2. `residual_scale` は `__init__` で計算した属性ではなく `@property`
   (`residual_scale_cent` を後から書き換えても整合する)。
3. `FlowMatchingF0` に `out_lf0_idx=0` を追加 (0 以外は assert で弾く)。
   既存の `BiLSTMResF0NonAttentiveDecoder` の yaml と設定の形を揃えるため。
4. `stream_weights` を `[1.0, 1.0, 1.0, 1.0]` ではなく `[0.25, 0.25, 0.25, 0.25]`
   にした。重みの総和が 1 になり (= stream_sizes 比の既定と同じ)、
   `clip_norm` の意味が既定と揃うため。
5. `FlowMatching` に `solver` の検証 (`euler`/`midpoint` 以外は `ValueError`) を追加。

**この環境でのテスト実行に関する注意**: `tests/test_model_configs.py` は
インストール済み isort 5.0.0 の `entry_points.txt` が壊れている
(`isort=isort=isort.pylama_isort:Linter`) ため mlflow のプラグイン走査が
`ValueError` になり、collection 段階で失敗する。本実装とは無関係の既存問題。
`nnsvs.bin.train_acoustic` の import も同じ理由で失敗するので、
train_step まで通す確認は `importlib_metadata.Distribution.entry_points` を
一時的に包むスクリプト経由で行った。

この文書は、別の Claude インスタンス (または人間) がこの文書だけを読んで
実装に着手できることを目的にしている。背景の実験記録は
`~/works/qiita-content/public/nnsvs_v5_failure.md` を参照。

---

## 0. 要約

- **何を作るか**: `nnsvs/techsinger/flow.py` に Rectified Flow (flow matching) ベースの
  2 クラスを追加する。
  - `FlowMatching`: mgc / bap 用。既存 `GaussianDiffusion` の置き換え。
  - `FlowMatchingF0`: lf0 用。**スコア lf0 に対する residual を flow で生成する**
    (NNSVS の ResF0 慣例を踏襲)。既存 `BiLSTMResF0NonAttentiveDecoder` の置き換え。
- **なぜ**: v5 の lf0 (自己回帰 LSTM) と mgc/bap (DDPM DiffNet) は最適な lr / grad clip が
  2 桁違い、単一 optimizer での共同学習が壊れる。全ストリームを同種の非自己回帰
  DiffNet + flow matching にして学習レジームを均質化し、exposure bias も除去する。
- **既存コードへの影響**: `train_acoustic.py` / `train_util.py` / `multistream.py` /
  `gen.py` / `pad_inference` は **無変更**で動く設計 (後述 §2)。追加は新規ファイル、
  `__init__.py` の export、テスト、recipe の yaml のみ。
- **外部依存の追加なし** (TechSinger は torchdyn を使うが Euler ループを自前実装する)。
- **ライセンス**: TechSinger リポジトリには LICENSE ファイルが無い。flow 部分は一般的な
  数式なので参照実装を写さず自前で書く。DiffNet は既に `nnsvs/diffsinger/denoiser.py`
  (DiffSinger 由来、MIT) にあるものをそのまま使う。

---

## 1. 参照元: TechSinger の該当実装

リポジトリ: https://github.com/gwx314/techsinger (論文 arXiv:2502.12572)。
2026-09-03 に clone して確認した内容。

### 1.1 `modules/TechSinger/flow/flow_f0.py` `ReflowF0`

```
学習:
  x1 = 正規化 F0 (B,1,1,T),  x0 = randn_like(x1)
  t  = randint(0, num_timesteps=1000) (B,)      # 整数。DiffNet の step 埋め込みに渡す
  tt = t / num_timesteps                          # (B,1,1,1)
  xt = tt * x1 + (1 - tt) * x0
  v_pred = denoise_fn(xt, t, cond)               # DiffNet, cond は (B,H,T)
  loss = L1(v_pred, x1 - x0) を nonpadding & voiced フレームでマスク平均

推論 (Wrapper + torchdyn NeuralODE, solver="euler", t_span=linspace(0,1,K_step+1), K_step=100):
  各 step で
    ut = denoise_fn(x, t*num_timesteps, cond)
    if f0_sample_clip:
        x1_hat = (1 - t) * ut + x                 # 1-step の終点推定
        x1_hat.clamp_(lower, upper)               # スコア MIDI ±3 半音 (dyn_clip)
        ut = (x1_hat - x) / (1 - t)
    x = x + dt * ut
```

- F0 の正規化は log2(f0) を [6, 10] → [−1, 1] に minmax。dyn_clip の境界も同じ正規化で作る。
- `F0DiffNet` は `nnsvs.diffsinger.DiffNet` と同一構造。stage1.yaml の値:
  `f0_residual_layers: 10`, `f0_residual_channels: 192`, `f0_dilation_cycle_length: 4`,
  `f0_timesteps: 1000`, `f0_K_step: 100`, `hidden_size: 256` (cond 次元)。
- cond は phoneme encoder 出力 + note encoder + speaker + technique embedding を
  フレーム長に展開したもの。**スコア音高は cond 経由でのみ入り、residual 予測はしていない**。
- U/V は別の小さな conv 予測器 (`PitchPredictor`, BCE)。

### 1.2 `modules/TechSinger/flow/flow.py` `FlowMel`

- mel 用。数式は ReflowF0 と同じ (clip は無し、CFG 任意)。
- `residual_layers: 20`, `residual_channels: 256`, `timesteps: 1000`, `K_step: 100`。
- cond = `Linear(cat(FS2 の coarse mel (detach), decoder_inp))`。stage1 を凍結した 2 段学習。
- mel の正規化は spec_min/spec_max による [−1, 1] minmax。

### 1.3 学習ハイパーパラメータ (`egs/egs_bases/tts/base.yaml`, `egs/stage{1,2}.yaml`)

| 項目 | stage1 (F0 flow + FS2) | stage2 (mel flow) |
|---|---|---|
| optimizer | Adam lr 5e-4, betas (0.9, 0.98), wd 0 | Adam lr 1e-3 |
| clip_grad_norm | 1 | 1 |
| scheduler | StepLR 50k step ×0.5 | 同 |
| max_updates | 200k | 160k |
| loss | L1 | L1 |

---

## 2. NNSVS 側の接続点 (変更不要な理由)

実装前に以下のファイルを読むこと。

| ファイル | 見るべき箇所 |
|---|---|
| `nnsvs/base.py` | `BaseModel`, `PredictionType.DIFFUSION` |
| `nnsvs/diffsinger/diffusion.py` | `GaussianDiffusion.forward/inference` (返り値の形が新クラスの規約) |
| `nnsvs/diffsinger/denoiser.py` | `DiffNet.forward(spec (B,1,M,T), step (B,), cond (B,H,T))` |
| `nnsvs/acoustic_models/multistream.py` | `NPSSMDNMultistreamParametricModel.forward/_set_lf0_params` |
| `nnsvs/acoustic_models/tacotron_f0.py` | `BiLSTMResF0NonAttentiveDecoder.forward` の residual 計算 (L380〜) |
| `nnsvs/acoustic_models/util.py` | `predict_lf0_with_residual`, `pad_inference` |
| `nnsvs/bin/train_acoustic.py` | `train_step` の MULTISTREAM_HYBRID 分岐 |
| `nnsvs/train_util.py` | `check_resf0_config`, `eval_model`, `get_stream_weight`, `build_grad_clip_groups` |

### 2.1 学習時の loss (`train_step`)

`prediction_type == MULTISTREAM_HYBRID` のとき、各ストリームの出力が
`tuple and len == 2` なら `(noise, x_recon)` とみなし
`criterion(noise.masked_select(mask), x_recon.masked_select(mask))` を取る。
**新クラスの `forward` が `(u_t, v_pred)` を同じ形 `(B, T, out_dim)` で返せば、この分岐が
そのまま flow matching の loss になる。** `criterion` は `train.feats_criterion`
(mse / l1) で全ストリーム共通。

### 2.2 推論時の連結 (`NPSSMDNMultistreamParametricModel.forward`, is_inference)

```python
lf0 = self.lf0_model.inference(x, lengths)      # (B,T,1) を返せばよい
lf0_cond = lf0                                  # PROBABILISTIC でなければそのまま
mgc = self.mgc_model.inference(torch.cat([x, lf0_cond], -1), lengths)
bap = self.bap_model.inference(torch.cat([x, lf0_cond], -1), lengths)
vuv = self.vuv_model.inference(cat(x, [mgc], [lf0], [bap]), lengths)
out = torch.cat([mgc, lf0, vuv, bap], -1); return out, out
```

`lf0_model.has_residual_lf0_prediction()` が True の場合、学習時は
`lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)` と 2 要素で受け取り、
`return (mgc, lf0, vuv, bap), lf0_residual` となる。`train_step` はこの
`lf0_residual` で `pitch_reg_weight * |residual|` を計算する。

### 2.3 lf0 統計値の注入

- `check_resf0_config` (`train_util.py`) が学習開始時に、netG (トップレベル) の
  `in_lf0_min/in_lf0_max/out_lf0_mean/out_lf0_scale` が None なら scaler から埋め、
  `config.model.netG[...]` に書き戻す。`save_configs` がその値を `model.yaml` に保存するので
  合成時も値が入る。
- `NPSSMDNMultistreamParametricModel._set_lf0_params` は毎 forward で
  `hasattr(self.lf0_model, "out_lf0_mean")` なら lf0_model の
  `in_lf0_min/in_lf0_max/out_lf0_mean/out_lf0_scale` を上書きする。
  **`in_lf0_idx` は伝播されない**ので lf0_model の yaml に明示する。
- 従って新クラスはこれら 4 属性を持つだけでよく、yaml では null にしておける。

### 2.4 その他

- `eval_model` (train_util.py) は 2 要素 tuple のストリームを diffusion とみなし、
  forward 出力の可聴化をスキップして inference 出力のみログする。flow でも同じ挙動でよい。
- `compute_distortions` に渡る forward 出力は `x_recon` 位置の値 (flow では `v_pred`) なので
  TensorBoard の teacher-forcing F0 RMSE 等は無意味になる。DDPM でも既にそうなので許容。
  評価は inference サンプルと §7 の外部スクリプトで行う。
- `pad_inference(mdn=True)` は inference 出力が `(out, out)` の tuple であることだけ要求する。
  multistream が満たすので変更不要。
- `reduction_factor`: 新クラスはフレーム単位 (r=1)。netG の `reduction_factor: 4` は
  `pad_inference` が長さを 4 の倍数に padding するだけなので影響なし。

---

## 3. `nnsvs/techsinger/flow.py` の設計

### 3.1 `FlowMatching(BaseModel)` — mgc / bap 用

```python
class FlowMatching(BaseModel):
    """Rectified-flow (flow matching) generator for a single feature stream.

    Drop-in replacement for GaussianDiffusion: same constructor shape,
    forward() returns a 2-tuple consumed by train_acoustic.train_step as
    (target, prediction), inference() returns (B, T, out_dim).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        denoise_fn,          # nnsvs.diffsinger.DiffNet, in_dim == out_dim
        encoder=None,        # e.g. nnsvs.model.FFConvLSTM, encoder.in_dim == in_dim
        num_timesteps=1000,  # 時刻埋め込みの離散化数 (TechSinger: timesteps)
        K_step=100,          # 推論の ODE step 数 (TechSinger: K_step)
        norm_scale=1.0,      # x1 = y / norm_scale。標準化済み特徴量なので既定 1.0
        solver="euler",      # "euler" | "midpoint"
    ):
        super().__init__()
        assert out_dim == denoise_fn.in_dim
        if encoder is not None:
            assert encoder.in_dim == in_dim
        ...

    def prediction_type(self):
        return PredictionType.DIFFUSION
```

**forward(cond, lengths=None, y=None)** — 学習専用。

```python
B = cond.shape[0]
if self.encoder is not None:
    cond = self.encoder(cond, lengths)          # (B,T,H)
cond = cond.transpose(1, 2)                     # (B,H,T)

t = torch.randint(0, self.num_timesteps, (B,), device=cond.device).long()
x1 = (y / self.norm_scale).transpose(1, 2)[:, None]   # (B,1,M,T)
x0 = torch.randn_like(x1)
tt = (t.float() / self.num_timesteps).view(B, 1, 1, 1)
xt = tt * x1 + (1.0 - tt) * x0
v_pred = self.denoise_fn(xt, t, cond)           # (B,1,M,T)
u = x1 - x0
return u.squeeze(1).transpose(1, 2), v_pred.squeeze(1).transpose(1, 2)   # 各 (B,T,M)
```

- 返り値順は `(target, prediction)`。`train_step` は順序に依存しない (対称な loss) が、
  `eval_model` は `[1]` を "x_recon 相当" として扱うので `v_pred` を 2 番目にする。
- TechSinger の `flow_qsample == "sig"` (微小ノイズ加算) は使わない ("direct")。

**inference(cond, lengths=None)** — ODE 積分。

```python
@torch.no_grad()
def inference(self, cond, lengths=None):
    B = cond.shape[0]
    if self.encoder is not None:
        cond = self.encoder(cond, lengths)
    cond = cond.transpose(1, 2)
    x = torch.randn((B, 1, self.out_dim, cond.shape[2]), device=cond.device)
    x = self._integrate(x, cond)
    return x[:, 0].transpose(1, 2) * self.norm_scale       # (B,T,M)

def _velocity(self, x, cond, t_float):
    t = torch.full((x.shape[0],), int(t_float * self.num_timesteps),
                   device=x.device, dtype=torch.long)
    v = self.denoise_fn(x, t, cond)
    return self._clip_velocity(x, v, t_float)              # hook (既定 no-op)

def _integrate(self, x, cond):
    dt = 1.0 / self.K_step
    for i in range(self.K_step):
        t = i * dt
        if self.solver == "euler":
            x = x + dt * self._velocity(x, cond, t)
        elif self.solver == "midpoint":
            v1 = self._velocity(x, cond, t)
            v2 = self._velocity(x + 0.5 * dt * v1, cond, t + 0.5 * dt)
            x = x + dt * v2
        else:
            raise ValueError(self.solver)
    return x

def _clip_velocity(self, x, v, t_float):
    """Sub-classes may clamp the 1-step endpoint estimate here."""
    return v
```

- `int(t_float * num_timesteps)` は TechSinger の `Wrapper` と同じ (`t * num_timesteps` を long)。
  t=1 には到達しない (最後の step は t = 1 − dt) ので (1 − t) の 0 除算は起きない。
- 既存 `GaussianDiffusion.inference` の `tqdm` は不要 (lf0 は軽い)。mgc 用に付けるなら
  `tqdm(range(K_step), desc="flow step", leave=False)` 程度。

### 3.2 `FlowMatchingF0(FlowMatching)` — lf0 用 (residual 既定)

```python
class FlowMatchingF0(FlowMatching):
    """Flow-matching log-F0 generator with residual-to-score parameterization.

    predict_residual=True (default):
        x1 = (lf0_target_denorm - lf0_score_denorm) / residual_scale
    predict_residual=False (TechSinger-faithful):
        x1 = y / norm_scale, endpoint clamped to score +- clip_cent at sampling
    """

    def __init__(
        self,
        in_dim,
        out_dim=1,
        denoise_fn=None,
        encoder=None,
        num_timesteps=1000,
        K_step=100,
        norm_scale=1.0,
        solver="euler",
        # ResF0 parameters (same names as BiLSTMResF0NonAttentiveDecoder)
        in_lf0_idx=51,
        in_lf0_min=None,
        in_lf0_max=None,
        out_lf0_idx=0,             # 0 以外は非対応
        out_lf0_mean=None,
        out_lf0_scale=None,
        # residual parameterization
        predict_residual=True,
        residual_scale_cent=600,   # x1 = +-1 が +-600 cent に対応
        clip_cent=600,             # 推論時の endpoint clamp。None で無効
    ):
        assert out_dim == 1
        assert out_lf0_idx == 0
        super().__init__(in_dim, out_dim, denoise_fn, encoder,
                         num_timesteps, K_step, norm_scale, solver)
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.predict_residual = predict_residual
        self.residual_scale_cent = residual_scale_cent
        self.clip_cent = clip_cent

    def has_residual_lf0_prediction(self):
        return True

    @property
    def residual_scale(self):
        # log-Hz 単位。x1 = 1 が residual_scale_cent cent に対応
        return self.residual_scale_cent * np.log(2) / 1200
```

補助メソッド:

```python
def _lf0_score_denorm(self, x):
    # x: (B,T,in_dim), in_feats は MinMax [0,1] 正規化済み (ResF0 decoder と同じ式)
    lf0_score = x[:, :, self.in_lf0_idx].unsqueeze(-1)              # (B,T,1)
    return lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min

def _to_normalized_lf0(self, lf0_denorm):
    return (lf0_denorm - self.out_lf0_mean) / self.out_lf0_scale
```

**forward(x, lengths=None, y=None)**

```python
assert y is not None, "target log-F0 must be given at training time"
score = self._lf0_score_denorm(x)                                    # (B,T,1)
if self.predict_residual:
    target_denorm = y * self.out_lf0_scale + self.out_lf0_mean
    y_flow = (target_denorm - score) / self.residual_scale           # (B,T,1)
else:
    y_flow = y

# (B,1,1,T), (B,1,1,T), (B,1,1,T), (B,1,1,1)
u, v_pred, xt, tt = self._forward_flow(x, lengths, y_flow)

# 1-step endpoint estimate for pitch regularization (B,T,1), log-Hz residual
x1_hat = self._denorm(_bcmt_to_btc(xt + (1.0 - tt) * v_pred))
if self.predict_residual:
    lf0_residual = x1_hat * self.residual_scale
else:
    lf0_residual = x1_hat * self.out_lf0_scale + self.out_lf0_mean - score
return (_bcmt_to_btc(u), _bcmt_to_btc(v_pred)), lf0_residual
```

- `FlowMatching` に `_forward_flow(cond, lengths, y) -> (u, v_pred, xt, tt)` を置き
  (すべて (B,1,out_dim,T)、`tt` のみ (B,1,1,1))、`FlowMatching.forward` はそれを呼んで
  先頭 2 つを `(B,T,out_dim)` に直して返す。F0 版は `xt`/`tt` も使うのでこの分割が必要。
- `_bcmt_to_btc` / `_btc_to_bcmt` は `flow.py` のモジュール関数
  ((B,1,C,T) ⇄ (B,T,C) の変換)。
- `lf0_residual` は `train_step` の `pitch_reg_weight * |lf0_residual|` に使われる。
  t が小さいときの endpoint 推定はノイズが大きいので、最初は `pitch_reg_weight: 0.0` で学習する。
  必要なら `(tt ** 2)` 等で重み付けするオプションを後で足す。

**inference(x, lengths=None)**

```python
def _compute_clip_bounds(self, lf0_score_denorm):
    """(B,T,1) のスコア lf0 から flow の内部空間での (lower, upper) を作る"""
    if self.clip_cent is None:
        return None
    max_lf0_ratio = self.clip_cent * np.log(2) / 1200            # log-Hz
    if self.predict_residual:
        upper = torch.full_like(lf0_score_denorm, max_lf0_ratio / self.residual_scale)
        lower = -upper
    else:
        upper = self._to_normalized_lf0(lf0_score_denorm + max_lf0_ratio)
        lower = self._to_normalized_lf0(lf0_score_denorm - max_lf0_ratio)
    # (B,T,1) -> (B,1,1,T)
    return _btc_to_bcmt(self._norm(lower)), _btc_to_bcmt(self._norm(upper))

@torch.no_grad()
def inference(self, x, lengths=None):
    score = self._lf0_score_denorm(x)                                # (B,T,1)
    clip_bounds = self._compute_clip_bounds(score)
    x1 = self._sample(x, lengths, clip_bounds=clip_bounds)           # (B,T,1)
    if self.predict_residual:
        return self._to_normalized_lf0(score + x1 * self.residual_scale)
    return x1
```

clamp 自体は基底クラスの `_clip_velocity(x, v, t_float, clip_bounds)` が行う
(`clip_bounds is None` なら no-op)。Euler の最終ステップは `t = 1 - dt` なので
`x_new = x + dt * (x1_hat - x)/dt = x1_hat` となり、clamp 後の終点にちょうど着地する
= 生成結果は厳密に clip_cent 以内になる (`midpoint` では近似的)。

- residual 既定では clamp 境界が定数になり、TechSinger の `dyn_clip` (毎 step スコア参照) と
  等価な効果を単純に得られる。rest フレームの特別扱いは不要 (スコア lf0 は前処理で補間済み、
  `BiLSTMResF0NonAttentiveDecoder` も特別扱いしていない)。
- `residual_scale_cent=600`, `clip_cent=600` で ResF0 decoder の `scaled_tanh` (±600 cent) と同じ範囲。
- 学習時の residual 分布は大半が ±100 cent (x1 で ±0.17) に集中する。N(0,1) prior との釣り合いが
  悪ければ `residual_scale_cent` を 200〜300 に下げる (clip_cent は 600 のままでよい)。

### 3.3 `nnsvs/techsinger/__init__.py`

```python
from .flow import FlowMatching, FlowMatchingF0
__all__ = ["DiffNet", "GaussianDiffusion", "FlowMatching", "FlowMatchingF0"]
```

hydra の `_target_` は `nnsvs.techsinger.FlowMatching` / `nnsvs.techsinger.FlowMatchingF0`。
`nnsvs/acoustic_models/__init__.py` への追加は不要。

---

## 4. recipe 設定

### 4.1 model yaml

実装済み:
`recipes/_common/conf/jp_dev_48k_nodyn/train_acoustic/model/acoustic_nnsvs_world_multi_flow_f0_flow_mgcbap.yaml`
(同ディレクトリの `acoustic_nnsvs_world_multi_ar_f0_diff_mgcbap.yaml` をベースにしている)。
`tests/test_model_configs.py::test_train_acoustic_model_config_recipes` が
`**/_common/conf/**/train_acoustic/model/*.yaml` を glob するので、この場所に置けば
自動でスモークテストの対象になる。
自分のデータセットの recipe に持っていく場合、`in_dim: 86`, `in_lf0_idx: 51`,
`in_ph_start_idx: 3`, `in_ph_end_idx: 50` は hed 依存なので既存 yaml の値を引き継ぐこと。

```yaml
stream_sizes: [60, 1, 1, 5]
has_dynamic_features: [false, false, false, false]
num_windows: 1
# stream_wise_loss: true のとき使われる。null だと stream_sizes 比 (60:1:1:5) になり
# lf0 が 1/67 に埋もれるので明示的に等重みにする。
# 総和が 1 になるよう 0.25 ずつにして clip_norm の意味を既定と揃える
stream_weights: [0.25, 0.25, 0.25, 0.25]

netG:
  _target_: nnsvs.acoustic_models.NPSSMDNMultistreamParametricModel
  in_dim: 86
  out_dim: 67
  stream_sizes: [60, 1, 1, 5]
  reduction_factor: 4          # pad_inference の padding 単位。そのままでよい
  in_rest_idx: 0
  in_lf0_idx: 51
  out_lf0_idx: 60
  in_lf0_min: null
  in_lf0_max: null
  out_lf0_mean: null
  out_lf0_scale: null
  vuv_model_bap_conditioning: false
  vuv_model_bap0_conditioning: false
  vuv_model_lf0_conditioning: true
  vuv_model_mgc_conditioning: true

  lf0_model:
    _target_: nnsvs.techsinger.FlowMatchingF0
    in_dim: 86
    out_dim: 1
    in_lf0_idx: 51             # netG.in_lf0_idx と同じ (伝播されないので必須)
    in_lf0_min: null           # 以下 4 つは _set_lf0_params が毎 forward で注入
    in_lf0_max: null
    out_lf0_mean: null
    out_lf0_scale: null
    predict_residual: true
    residual_scale_cent: 600
    clip_cent: 600
    num_timesteps: 1000
    K_step: 100
    solver: euler
    encoder:
      _target_: nnsvs.model.FFConvLSTM
      in_dim: 86
      in_ph_start_idx: 3
      in_ph_end_idx: 50
      embed_dim: 256
      ff_hidden_dim: 256
      conv_hidden_dim: 128
      lstm_hidden_dim: 64
      num_lstm_layers: 2
      bidirectional: true
      dropout: 0.0
      out_dim: 256             # = denoise_fn.encoder_hidden_dim
    denoise_fn:
      _target_: nnsvs.diffsinger.DiffNet
      in_dim: 1
      encoder_hidden_dim: 256
      residual_layers: 10      # TechSinger f0_residual_layers
      residual_channels: 192   # TechSinger f0_residual_channels
      dilation_cycle_length: 4

  mgc_model:
    _target_: nnsvs.techsinger.FlowMatching
    in_dim: 87                 # (x, lf0)
    out_dim: 60
    norm_scale: 1.0            # DDPM の 10 は [-1,1] clip 前提。flow では 1〜3 で実験
    num_timesteps: 1000
    K_step: 100
    solver: euler
    encoder:                   # 既存 DDPM 設定と同じ
      _target_: nnsvs.model.FFConvLSTM
      in_dim: 87
      in_ph_start_idx: 3
      in_ph_end_idx: 50
      embed_dim: 256
      ff_hidden_dim: 512
      conv_hidden_dim: 256
      lstm_hidden_dim: 128
      num_lstm_layers: 2
      bidirectional: true
      dropout: 0.0
      out_dim: 256
    denoise_fn:
      _target_: nnsvs.diffsinger.DiffNet
      in_dim: 60
      encoder_hidden_dim: 256
      residual_layers: 20
      residual_channels: 256
      dilation_cycle_length: 4

  bap_model:
    _target_: nnsvs.techsinger.FlowMatching
    in_dim: 87
    out_dim: 5
    norm_scale: 1.0
    num_timesteps: 1000
    K_step: 100
    solver: euler
    encoder:                   # 既存 DDPM 設定と同じ (out_dim 128)
      ...
    denoise_fn:
      _target_: nnsvs.diffsinger.DiffNet
      in_dim: 5
      encoder_hidden_dim: 128
      residual_layers: 10
      residual_channels: 128
      dilation_cycle_length: 4

  vuv_model:                   # 変更なし (FFConvLSTM)
    ...
```

### 4.2 train yaml

既存の v5 用 train yaml から以下を変更する。

```yaml
nepochs: 300                  # データ量に応じて
feats_criterion: l1           # TechSinger は L1。vuv も L1 になる点は許容 (後述 §8)
pitch_reg_weight: 0.0         # まず無効。endpoint 推定のノイズを避ける
stream_wise_loss: true        # model.stream_weights [1,1,1,1] と組で等重み
optim:
  optimizer:
    name: Adam
    params:
      lr: 0.0005              # TechSinger stage1
      betas: [0.9, 0.98]
      weight_decay: 0.0
  lr_scheduler:
    name: StepLR
    params:
      step_size: 100          # epoch 単位。TechSinger の 50k step 相当を換算
      gamma: 0.5
  clip_norm: 1.0
  # param_groups は最初は指定しない (均質化で単一設定が成立するかを見る)。
  # うまくいかない場合のフォールバック例:
  # param_groups:
  #   - modules: [lf0_model]
  #     lr: 0.0005
  #     clip_norm: 1.0
  #   - modules: [mgc_model, bap_model]
  #     lr: 0.001
  #     clip_norm: 1.0
```

`stream_wise_loss: true` の意図: false だと全ストリーム誤差の総和を総要素数 (≈67·B·T) で割るため
lf0 の loss 寄与が 1/67 になる。各サブモジュールは teacher forcing で独立なので Adam では
gradient のスケール自体は概ね正規化されるが、`clip_norm` との相互作用で lf0 側が
一切 clip されない/され過ぎる事態を避けるため等重みにする。

---

## 5. テスト (`tests/test_flow.py`)

実装済み。18 件すべて green
(`pytest tests/test_flow.py tests/test_diffusion.py tests/test_acoustic_models.py`
→ 317 passed)。`tests/test_diffusion.py` と `tests/util.py::_test_model_impl` の
流儀に合わせ、極小の `DiffNet` (2 layers / 4 ch) と `LSTMEncoder` を使う。

| テスト | 検証内容 |
|---|---|
| `test_flow_matching[solver, norm_scale]` | `forward` の 2-tuple の形、`inference` の形と有限性、`prediction_type() == DIFFUSION` |
| `test_flow_matching_without_encoder` | `encoder=None` で encoder 出力を直接渡す経路 |
| `test_flow_matching_target_velocity` | `y = 0` のとき目標速度 `u = -x0 ~ N(0,1)` になること (mean≈0, std≈1) |
| `test_flow_matching_f0[solver, predict_residual]` | `((u, v), lf0_residual)` の形、`has_residual_lf0_prediction() == True` |
| `test_flow_matching_f0_residual_roundtrip` | residual 変換 → 逆変換で正規化 lf0 が復元されること |
| `test_flow_matching_f0_endpoint_clip[predict_residual]` | 生成 lf0 がスコア ±`clip_cent` (=300 cent) 以内 (Euler なので厳密) |
| `test_flow_matching_f0_no_clip` | `clip_cent: null` で clamp 無効 |
| `test_flow_matching_f0_invalid_out_dim` / `test_flow_matching_invalid_solver` | 設定ミスが assert / ValueError になること |
| `test_flow_multistream_parametric_model[reduction_factor]` | lf0/mgc/bap を flow にした `NPSSMDNMultistreamParametricModel` の学習 forward (mgc/lf0/bap が 2-tuple、vuv が tensor、`lf0_residual` が (B,T,1)) と `inference` の `(mu, sigma)`。lf0 統計が `_set_lf0_params` で伝播することも確認。`reduction_factor=4` で `pad_inference` の padding 経路も通る |

`_test_model_impl` は MULTISTREAM_HYBRID を TODO にしているので統合テストは自前で書いている。

---

## 6. 実装順序とチェックリスト

- [x] `nnsvs/techsinger/flow.py` に `FlowMatching` / `FlowMatchingF0`
- [x] `nnsvs/techsinger/__init__.py` の export
- [x] `tests/test_flow.py` (18 件 green)、既存テストの回帰なし
- [x] recipe yaml (§4) の作成と `hydra.utils.instantiate` + forward/inference のスモーク
      (`tests/test_model_configs.py` はこの環境では collection 不能なので、
      同等の処理をスクリプトで実行して確認)
- [x] `train_step` を直接呼ぶスモーク (loss 有限、backward、grad_norm、
      `stream_wise_loss` の true/false、`pitch_reg_weight` の 0/0.1 の 4 通り)
- [ ] 数 epoch の smoke train (`max_num_eval_utts: 2`) で TensorBoard の inference
      サンプルの lf0 がスコア付近になることを確認
- [ ] 本学習 → §7 で評価

**注**: `DiffNet` は `output_projection.weight` をゼロ初期化しているため、
最初の 1 step だけ `denoise_fn` 内部のパラメータの勾配が厳密に 0 になる
(DDPM の `GaussianDiffusion` でも同じ)。2 step 目以降は流れるので問題ない。

コミットは Conventional Commits、本文英語。例:
`feat: add flow matching acoustic streams (FlowMatching, FlowMatchingF0)`。
自動 commit / push はしない (ユーザー確認必須)。

---

## 7. 評価手順

- 記事の実験 5 (v4 lf0/vuv warm-start + freeze、DDPM mgc/bap) を比較対象にする。
- `utils/f0_stability_eval.py` (記事 §3.1) で合成 wav と楽譜を突き合わせ、
  `drift_cent_per_sec` median(abs)、`detrended_sigma_cent`、`off_note_rate`、`dev_p95_cent`、
  フレーム毎 |score − pred| median、mgc[0] `diff_std` を実験 5 と横並びにする。
- 実験 5 の値 (記事より): drift 98.6, σ 60.6, off_note 0.5 %, dev_p95 27.5, frame med 12.1,
  mgc0 diff_std 0.483。
- 段階:
  - Stage A: 全ストリーム同時学習 (単一 lr/clip)。本命の仮説検証。
  - Stage B (A が不十分な場合): `train.optim.param_groups` で lf0 と mgc/bap の lr/clip を分ける。
  - Stage C (任意): `K_step` を 25/50 に下げて推論コストと品質のトレードオフを見る。
    `solver: midpoint` は step 数半分で同等精度になり得る。

---

## 8. 既知のギャップ・注意点

- **loss の種類**: `feats_criterion` は全ストリーム共通。L1 にすると vuv も L1 になる。
  問題になれば `train_step` にストリーム別 criterion を追加する (本計画の範囲外)。
- **TensorBoard 指標**: forward 出力が `v_pred` なので teacher-forcing の F0 RMSE / VUV err は
  無意味。inference サンプルと §7 の外部評価を見る。
- **TechSinger との差分**:
  - TechSinger は有声フレームのみ loss。NNSVS の lf0 は補間済み連続値なので全フレームで loss を取る。
  - TechSinger の mel flow は FS2 の coarse mel を条件に取る。NNSVS は encoder のみ。
    品質不足なら encoder 出力に補助回帰ヘッド (MSE) を付け、その出力を cond に連結する (phase 2)。
  - TechSinger は absolute F0 + dyn_clip。本実装の既定は residual + 定数 clamp。
    `predict_residual: false` で TechSinger 忠実版に切り替えて比較できる。
- **norm_scale**: DDPM の `norm_scale: 10` は [−1, 1] clip 前提だった。flow では clip が無いので
  標準化データをそのまま (1.0) 使う。1〜3 でスイープ候補。
- **mgc[0] ジッター**: ODE は決定論的で DDPM の ancestral noise 注入が無いため改善が期待できるが保証はない。
- **vuv の条件 mismatch**: 学習時は正解 mgc/lf0、推論時は flow サンプルに条件付け。現状 v5 と同じ。
- **DDPM checkpoint からの warm-start**: DiffNet の構造は同じだが ε 予測と v 予測で目的が異なる。
  encoder (FFConvLSTM) の重みのみ流用する価値はあるかもしれない。
- **`_set_lf0_params` は `in_lf0_idx` を伝播しない**。lf0_model の yaml に必ず書く。
- **`has_residual_lf0_prediction()` を True にする影響**: multistream / train_step / eval_model /
  pad_inference すべて `(outs, lf0_residual)` の 2 要素返しに対応済み。
  `residual` の形は `(B, T, 1)` で `mask (B, T, 1)` とブロードキャスト可能。

---

## 9. 範囲外 (やらないこと)

- torchdyn / CFG (classifier-free guidance) / `flow_qsample: sig` の移植。
- TechSinger の 2 段学習 (stage1 凍結 → stage2)。まず同時学習で検証する。
- TechSinger の technique embedding / NoteEncoder。NNSVS は hed 由来の x で代替。
- mel 系 (`MultistreamSeparateF0MelModel`) への適用。WORLD 系 (mgc/lf0/vuv/bap) のみ。
