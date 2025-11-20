#! /usr/bin/env bash

set -eu

if [ "$#" -ne 7 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} EXPERIMENT_ID EA_PARAMS_FILE STRATEGY FIT DRUG SWIFT_FILE (e.g. ${script_name} experiment_1 data/ga_params.json)"
  # echo "Usage: ${script_name} EXPERIMENT_ID EA_PARAMS_FILE (e.g. ${script_name} experiment_1 data/ga_params.json)"
  exit 1
fi

# uncomment to turn on swift/t logging. Can also set TURBINE_LOG,
# TURBINE_DEBUG, and ADLB_DEBUG to 0 to turn off logging
# export TURBINE_LOG=1 TURBINE_DEBUG=1 ADLB_DEBUG=1
export EMEWS_PROJECT_ROOT=$( cd $( dirname $0 )/.. ; /bin/pwd )
# source some utility functions used by EMEWS in this script
source "${EMEWS_PROJECT_ROOT}/etc/emews_utils.sh"


export EXPID=$1
export TURBINE_OUTPUT=$EMEWS_PROJECT_ROOT/experiments/$EXPID
check_directory_exists



# CFG_PROCS=$(( NODES * CFG_PPN ))    

# TODO edit QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME
# as required. Note that QUEUE, WALLTIME, PPN, AND TURNBINE_JOBNAME will
# be ignored if MACHINE flag (see below) is not set


##########################################################################

# RUNNING THE EQPY IN MN5

##########################################################################

# TODO edit the number of processes as required.
export PROCS=112
# 225 PROCS is 16 nodes in MN5
# 128 PROCS is 10 nodes in MN5
# 112 PROCS is 8 nodes in MN5


# export QUEUE=gp_bscls # for MN5
export QUEUE=gp_resa # for MN5
# export QUEUE=gp_debug
export PROJECT=cns119 # it's bsc08 for my own user, cns119 for the RES project
export WALLTIME=72:00:00 
TOTAL_CPU_PER_NODE=112
export PPN=14
export CPUS_PER_TASK=$(($TOTAL_CPU_PER_NODE / $PPN))
export TURBINE_JOBNAME="${EXPID}_job"
# export TURBINE_LAUNCHER="srun -c $CPUS_PER_TASK"
export TURBINE_LAUNCHER="srun -c $CPUS_PER_TASK"
#export TURBINE_SBATCH_ARGS="-N 1 --ntasks=28 --cpus-per-task=4"



##########################################################################

# RUNNING THE EQPY EMEWS IN NORD4

##########################################################################

# IF USING NORD4
# export PROCS=48 # 96 is 2 full nodes
# export QUEUE=debug # QoS name for MN4
# export PROJECT=bsc08
# export WALLTIME=02:00:00 
# TOTAL_CPU_PER_NODE=48 # 48 cores per node is the default for MN4
# export PPN=6
# export CPUS_PER_TASK=$(($TOTAL_CPU_PER_NODE / $PPN))
# export TURBINE_JOBNAME="${EXPID}_job"
# export TURBINE_LAUNCHER="srun -c $CPUS_PER_TASK"
# # Add both partition and QoS, using 'main' as the default compute partition
# export TURBINE_SBATCH_ARGS="--qos=bsc_ls --account=$PROJECT --partition=main --nodes=8 --ntasks-per-node=$PPN"

# if R cannot be found, then these will need to be
# uncommented and set correctly.
# export R_HOME=/path/to/R
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib
# export PYTHONHOME=/path/to/python
# export PYTHONPATH=$EMEWS_PROJECT_ROOT/python:$EMEWS_PROJECT_ROOT/ext/EQ-Py
export PYTHONPATH=$EMEWS_PROJECT_ROOT/ext/EQ-Py:$EMEWS_PROJECT_ROOT/python
export PYTHONPATH=/apps/GPP/EQPy/INTEL/:$PYTHONPATH
export PCDL_PYTHON=/apps/PYTHON/3.9.10/bin/python


# Resident task workers and ranks
export TURBINE_RESIDENT_WORK_WORKERS=1
export RESIDENT_WORK_RANKS=$(( PROCS - 2 ))

# EQ/Py location
EQPY=$EMEWS_PROJECT_ROOT/ext/EQ-Py

mkdir -p $TURBINE_OUTPUT

ALGO_PARAMS_FILE_SOURCE=$2
EXE_SOURCE=$EMEWS_PROJECT_ROOT/model/PhysiBoSS/physiboss-drugs-synergy-model
# EXE_SOURCE=$EMEWS_PROJECT_ROOT/model_mdr/drugs_synergy_model
# SETTINGS_SOURCE=$EMEWS_PROJECT_ROOT/data/settings_drug_template_2D.xml 
SETTINGS_SOURCE=$7 

EXE_OUT=$TURBINE_OUTPUT/`basename $EXE_SOURCE`
SETTINGS_OUT=$TURBINE_OUTPUT/settings.xml
ALGO_PARAMS_FILE_OUT=$TURBINE_OUTPUT/`basename $2`
# EXPERIMENTAL_PATH=`realpath $3` # path to the Flobak et al. experimental data


cp $EXE_SOURCE $EXE_OUT
cp $SETTINGS_SOURCE $SETTINGS_OUT
cp $ALGO_PARAMS_FILE_SOURCE $ALGO_PARAMS_FILE_OUT

cp -r $EMEWS_PROJECT_ROOT/model/PhysiBoSS/config $TURBINE_OUTPUT
# cp -r $EMEWS_PROJECT_ROOT/data/boolean_network $TURBINE_OUTPUT

SEED=1998
ITER=500 # was 15
REP=100 # was 3
POP=1000 # was 300
SIGMA=1 # This was the default, if it's too high it does not explore the landscape

STRATEGY=$3
METRIC_FITTING=$4
DRUG=$5

# -experimental_path=$EXPERIMENTAL_PATH 

CMD_LINE_ARGS="$* -strategy=$STRATEGY -metric=$METRIC_FITTING -drug=$DRUG -sigma=$SIGMA -seed=$SEED -ni=$ITER -nv=$REP -np=$POP -exe=$EXE_OUT -settings=$SETTINGS_OUT -ea_params=$ALGO_PARAMS_FILE_OUT"

# Uncomment this for the BG/Q:
#export MODE=BGQ QUEUE=default

# set machine to your schedule type (e.g. pbs, slurm, cobalt etc.),
# or empty for an immediate non-queued unscheduled run
MACHINE="slurm"
# MACHINE=""

if [ -n "$MACHINE" ]; then
  MACHINE="-m $MACHINE"
fi

# Add any script variables that you want to log as
# part of the experiment meta data to the USER_VARS array,
# for example, USER_VARS=("VAR_1" "VAR_2")
USER_VARS=()

# log variables and script to to TURBINE_OUTPUT directory
log_script

# module load python/3.6.1 java intel openmp mkl R/3.4.0 java/8u131 swiftt/1.5.0 gcc/12.1.0_binutils # Running on Nord4
module load swig zsh/5.9 java-jdk/8u131 ant/1.10.14 hdf5 python/3.12.1 intel impi mkl R  swiftt/1.6.2-python-3.12.1   # Running on MN5


# ---------------------------------------------------------------------------------------------------------------------- /apps/modules/modulefiles/tools -----------------------------------------------------------------------------------------------------------------------
#    ANACONDA/5.0.1_python2    python/2-intel-2019.2    python/2.7.13    python/2.7.16            python/3-intel-2019.2    python/3.6.1_spack        python/3.6.4_ML_grad_cam    python/3.6.6_gdb                   python/3.7.4_ES_test    python/3.7.4    python/3.9.10
#    python/2-intel-2018.2     python/2.7.13_ML         python/2.7.14    python/3-intel-2018.2    python/3-intel-2021.3    python/3.6.1       (D)    python/3.6.4_ML             python/3.7.4_ES_test_jupyterlab    python/3.7.4_ES         python/3.8.2    python/3.10.2


# activate your virtual environment with your dependencies
# activate () {
#   . ../python/venv/bin/activate
# }

#export PYTHONPATH=${PYTHONPATH}:$EMEWS_PROJECT_ROOT/python
export PYTHONPATH=$EMEWS_PROJECT_ROOT/python/:$EMEWS_PROJECT_ROOT/ext/EQ-Py


# echo's anything following this to standard out
set -x

# DEFAULT
SWIFT_FILE=$6
swift-t -n $PROCS $MACHINE -p -I $EQPY -I $EMEWS_PROJECT_ROOT/swift/ -r $EQPY $EMEWS_PROJECT_ROOT/swift/$SWIFT_FILE $CMD_LINE_ARGS
