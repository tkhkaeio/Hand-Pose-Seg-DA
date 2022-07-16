source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

len=$#

# NOTE: add your args to overwrite default
if [ ${len} -eq 0 ]; then
    python run_joint.py \
        exp=config_single_domain \
        #exp.base.debug=True \
elif [ ${len} -eq 0 ]; then
    python run_joint.py \
        exp=config_single_domain \
        exp.model.load_joint_model=$1 \
        exp.base.resume=$2
        #exp.base.debug=True \
fi
