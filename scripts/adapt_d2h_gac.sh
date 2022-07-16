source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

len=$#
# NOTE: add your args to overwrite
if [ ${len} -eq 0 ]; then
    python run_joint.py \
        exp=config_dexycb_to_ho3d_gac \
        # exp.base.debug=True
elif [ ${len} -eq 1 ]; then
    python run_joint.py \
        exp=config_dexycb_to_ho3d_gac \
        exp.base.adapt_mode=${1} \
        # exp.base.debug=True
elif [ ${len} -eq 2 ]; then 
    # NOTE: when you retrain model
    python run_joint.py \
        exp=config_dexycb_to_ho3d_gac \
        exp.base.adapt_mode=${1} \
        exp.model.load_joint_model=${2} \
        exp.base.resume=1 \
        # exp.base.debug=True
fi 
