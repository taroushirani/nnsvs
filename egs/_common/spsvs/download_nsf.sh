# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directry.

if [ ! -e $nsf_root_dir ]; then
    mkdir -p downloads
    cd downloads
    git clone https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
    cd $script_dir
fi
