#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../../
NNSVS_COMMON_ROOT=$NNSVS_ROOT/egs/_common/spsvs
NO2_ROOT=$NNSVS_ROOT/egs/_common/no2
. $NNSVS_ROOT/utils/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($dev_set $eval_set)

dumpdir=dump
dump_org_dir=$dumpdir/$spk/org
dump_norm_dir=$dumpdir/$spk/norm

stage=0
stop_stage=0

. $NNSVS_ROOT/utils/parse_options.sh || exit 1;

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}
else
    expname=${spk}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e $db_root ]; then
	cat<<EOF
stage -1: Downloading

This recipe does not download ONIKU_KURUMI_UTAGOE_DB.zip automatically to 
provide you the opportunity to read the original license.

Please visit http://onikuru.info/db-download/ and read the term of services,
and then download the singing voice database manually.
EOF
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    sh $NO2_ROOT/utils/data_prep.sh ./config.yaml
    mkdir -p data/list

    echo "train/dev/eval split"
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | sort > data/list/utt_list.txt
    grep haruga_kita_ data/list/utt_list.txt > data/list/$eval_set.list
    grep kagome_kagome_ data/list/utt_list.txt > data/list/$dev_set.list
    grep -v haruga_kita_ data/list/utt_list.txt | grep -v kagome_kagome_ > data/list/$train_set.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation"
    . $NNSVS_COMMON_ROOT/feature_generation.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training time-lag model"
    . $NNSVS_COMMON_ROOT/train_timelag.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training duration model"
    . $NNSVS_COMMON_ROOT/train_duration.sh
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training acoustic model"
    . $NNSVS_COMMON_ROOT/train_acoustic.sh
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generate features from timelag/duration/acoustic models"
    . $NNSVS_COMMON_ROOT/generate.sh
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    . $NNSVS_COMMON_ROOT/synthesis.sh
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    if [ ! -e $nsf_root_dir ]; then
	echo "stage 7: Downloading NSF"
        mkdir -p downloads
        cd downloads
	git clone https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
	cd $script_dir
    fi
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Data preparation for NSF"
    out_dir=$expdir/nsf
    mkdir -p $out_dir
    for s in ${datasets[@]};
    do
        if [ $s = $eval_set ]; then
	    xrun python local/prepare_nsf_data.py in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir test_set=true
        else
	    xrun python local/prepare_nsf_data.py in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir
	fi
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Training NSF model"
    if [ ! -e $nsf_root_dir ]; then
	echo "No NSF files found. Please set nsf_root_dir properly or run stage 7."
	exit 1
    fi

    input_dirs=$expdir/nsf/input_dirs
    output_dirs=$expdir/nsf/output_dirs
    mkdir -p $output_dirs
    mkdir -p $nsf_save_model_dir
    xrun python local/train_nsf.py \
	 nsf_root_dir=$nsf_root_dir \
	 nsf_type=hn-sinc-nsf \
	 nsf.args.batch_size=1 \
	 nsf.args.epochs=100 \
	 nsf.args.no_best_epochs=5 \
	 nsf.args.lr=0.00003 \
	 nsf.args.save_model_dir=$nsf_save_model_dir \
	 nsf.args.trained_model=$nsf_pretrained_model \
	 nsf.model.input_dirs=["$input_dirs","$input_dirs","$input_dirs"]\
	 nsf.model.output_dirs=["$output_dirs"]
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Evaluating NSF model"
    if [ ! -e $nsf_root_dir ]; then
	echo "No NSF files found. Please set nsf_root_dir properly or run stage 7."
	exit 1
    fi

    # for inference
    test_input_dirs=$expdir/nsf/test_input_dirs
    test_output_dirs=$expdir/nsf/test_output_dirs
    mkdir -p $test_output_dirs
    xrun python local/train_nsf.py \
	 nsf_root_dir=$nsf_root_dir \
	 nsf_type=hn-sinc-nsf \
	 nsf.args.batch_size=1 \
	 nsf.args.save_model_dir=$nsf_save_model_dir \
	 nsf.args.trained_model=$nsf_pretrained_model \
	 nsf.args.inference=true \
	 nsf.model.test_input_dirs=["$test_input_dirs","$test_input_dirs","$test_input_dirs"]\
	 nsf.model.test_output_dirs=$test_output_dirs

fi
