# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directry.

if [ -d conf/prepare_nsf_data ]; then
    ext="--config-dir conf/prepare_nsf_data"
else
    ext=""
fi

out_dir=$expdir/nsf
mkdir -p $out_dir
for s in ${datasets[@]};
do
    if [ $s = $eval_set ]; then
	xrun nnsvs-prepare-nsf-data $ext question_path=$question_path in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir test_set=true
    else
	xrun nnsvs-prepare-nsf-data $ext question_path=$question_path in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir
    fi
done
