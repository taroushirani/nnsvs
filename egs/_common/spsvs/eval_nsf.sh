# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directry.

if [ ! -e $nsf_root_dir ]; then
    echo "No NSF files found. Please set nsf_root_dir properly or run stage 7."
    exit 1
fi

if [ -d conf/train_nsf ]; then
    ext="--config-dir conf/train_nsf"
else
    ext=""
fi

# for inference
mkdir -p $nsf_test_output_dirs
echo $nsf_save_model_dir
xrun nnsvs-train-nsf $ext \
     nsf=$nsf_config \
     nsf.root_dir=$nsf_root_dir \
     nsf.args.save_model_dir=$nsf_save_model_dir \
     nsf.args.inference=true  \
     nsf.model.test_input_dirs=["$nsf_test_input_dirs","$nsf_test_input_dirs","$nsf_test_input_dirs"]\
     nsf.model.test_output_dirs=$nsf_test_output_dirs

