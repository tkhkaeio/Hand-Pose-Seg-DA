source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

python run_joint.py \
    exp=config_single_domain \
    exp.base.mode=val_test \
    exp.base.dataset=ho3d \
    exp.model.load_joint_model=$1 \
    # exp.base.debug=True \