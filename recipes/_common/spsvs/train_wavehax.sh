# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ ! -z ${pretrained_vocoder_checkpoint} ]; then
    ckpt_basename=$(basename "${pretrained_vocoder_checkpoint}")
    if [[ "$ckpt_basename" =~ ^checkpoint-([0-9]+)steps\.pkl$ ]]; then
        resume_steps="${BASH_REMATCH[1]}"
    else
        echo "ERROR: pretrained_vocoder_checkpoint must be named checkpoint-<steps>steps.pkl (got: $ckpt_basename)"
        exit 1
    fi
    if [ ! -e "$expdir/$vocoder_model/checkpoints/checkpoint-${resume_steps}steps.pkl" ]; then
        echo "ERROR: expected checkpoint not found at $expdir/$vocoder_model/checkpoints/checkpoint-${resume_steps}steps.pkl"
        echo "Resume only supports continuing training within the same experiment directory (out_dir=$expdir/$vocoder_model)."
        exit 1
    fi
    extra_args="train.resume=${resume_steps}"
else
    extra_args=""
fi
if [[ -z "${vocoder_model}" ]]; then
    echo "ERROR: vocoder_model is not specified."
    echo "Please specify a vocoder config name"
    echo "Note that conf/train_wavehax/generator/\${vocoder_model}.yaml must exist."
    exit 1
fi

if [[ ${acoustic_features} == *"melf0"* ]]; then
    feature_type="melf0"
else
    feature_type="world"
fi

if [ -z "${RUNNING_TEST_RECIPES+x}" ]; then
    wavehax_data_config=nnsvs_${feature_type}_sr48k
    wavehax_train_config=nnsvs_wavehax
    wavehax_discriminator_config=nnsvs_univnet
else
    # If we are running tests, use a config for testing purpose
    wavehax_data_config=nnsvs_${feature_type}_sr48k_test
    wavehax_train_config=nnsvs_wavehax_test
    wavehax_discriminator_config=nnsvs_univnet
fi

# Convert NNSVS's data to wavehax's format
if [ ! -d dump_wavehax ]; then
    python $NNSVS_ROOT/utils/nnsvs2wavehax.py config.yaml dump_wavehax --feature_type $feature_type
fi

# NOTE: copy normalization stats to expdir for convenience
mkdir -p $expdir/$vocoder_model
cp -v $dump_norm_dir/in_vocoder*.npy $expdir/$vocoder_model

# NOTE: To get the maximum performance, it is highly recommended to configure
# training options in detail
# NOTE: conf/train_wavehax/generator/${vocoder_model}.yaml must exist
cmdstr="wavehax-train --config-dir conf/train_wavehax/ \
    data=$wavehax_data_config \
    discriminator=$wavehax_discriminator_config \
    train=$wavehax_train_config \
    generator=$vocoder_model \
    data.train_audio=dump_wavehax/scp/${spk}_sr${sample_rate}_train_no_dev.scp \
    data.train_feat=dump_wavehax/scp/${spk}_sr${sample_rate}_train_no_dev.list \
    data.valid_audio=dump_wavehax/scp/${spk}_sr${sample_rate}_dev.scp \
    data.valid_feat=dump_wavehax/scp/${spk}_sr${sample_rate}_dev.list \
    data.eval_audio=dump_wavehax/scp/${spk}_sr${sample_rate}_eval.scp \
    data.eval_feat=dump_wavehax/scp/${spk}_sr${sample_rate}_eval.list \
    data.stats=dump_wavehax/stats/scaler.joblib \
    out_dir=$expdir/$vocoder_model $extra_args"
echo $cmdstr
eval $cmdstr
