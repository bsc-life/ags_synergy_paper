#! /usr/bin/env bash

set -eu

if [ "$#" -ne 5 ]; then
  script_name=$(basename $0)
  echo "Usage: ${script_name} EXPERIMENT_ID INPUT SETTINGS_XML (e.g. ${script_name} exp_1 data/input.txt data/settings_template_3D.xml)"
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

##########################################################################
# RUNNING THE EQPY IN MN5
##########################################################################

# TODO edit the number of processes as required.
export PROCS=112 
# 225 PROCS is 16 nodes in MN5
# 128 PROCS is 10 nodes in MN5
# 112 PROCS is 8 nodes in MN5


export QUEUE=gp_bscls # for MN5
# export QUEUE=gp_debug
export PROJECT=bsc08
export WALLTIME=24:00:00 
TOTAL_CPU_PER_NODE=112
export PPN=14
export CPUS_PER_TASK=$(($TOTAL_CPU_PER_NODE / $PPN))
export TURBINE_JOBNAME="${EXPID}_job"
# export TURBINE_LAUNCHER="srun -c $CPUS_PER_TASK"
export TURBINE_LAUNCHER="srun -c $CPUS_PER_TASK"
#export TURBINE_SBATCH_ARGS="-N 1 --ntasks=28 --cpus-per-task=4"

# TODO edit command line arguments as appropriate
# for your run. Note that the default $* will pass all of this script's
# command line arguments to the swift script.
mkdir -p $TURBINE_OUTPUT

PARAMS_FILE_SOURCE=`realpath $2`
SETTINGS_SOURCE=`realpath $3`
# EXPERIMENTAL_PATH=`realpath $4` # path to the Flobak et al. experimental data

# SETTINGS_SOURCE=$EMEWS_PROJECT_ROOT/data/settings_template_3D.xml


# EXE_SOURCE=$EMEWS_PROJECT_ROOT/model_mdr/drugs_synergy_model
EXE_SOURCE=$EMEWS_PROJECT_ROOT/model/PhysiBoSS/physiboss-drugs-synergy-model
EXE_OUT=$TURBINE_OUTPUT/`basename $EXE_SOURCE`
SETTINGS_OUT=$TURBINE_OUTPUT/settings.xml
PARAMS_FILE_OUT=$TURBINE_OUTPUT/`basename $PARAMS_FILE_SOURCE`

cp $EXE_SOURCE $EXE_OUT
cp $SETTINGS_SOURCE $SETTINGS_OUT
cp $PARAMS_FILE_SOURCE $PARAMS_FILE_OUT

# @oth: Copy all contect of the model config folder into the experiment directory
# This makes everything a bit easier

cp -r $EMEWS_PROJECT_ROOT/model/PhysiBoSS/config $TURBINE_OUTPUT
# cp -r $EMEWS_PROJECT_ROOT/data/simulation_initial_setup $TURBINE_OUTPUT

REP=5
METRIC_FITTING=$5
DRUG_NAME=$4

# -experimental_path=$EXPERIMENTAL_PATH

CMD_LINE_ARGS="$* -nv=$REP -metric=$METRIC_FITTING -drug=$DRUG_NAME -exe=$EXE_OUT -settings=$SETTINGS_OUT -parameters=$PARAMS_FILE_OUT"

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

# module load python java R/3.4.0 swiftt/1.5.0
# module load python java intel/2021.4.0 impi/2021.4.0 mkl/2021.4.0 R java/8u131 swiftt/1.5.0 # Running on Nord III
# module load python java intel impi mkl R/3.4.0 java/8u131 swiftt/1.5.0 # Running on MN4
module load swig zsh/5.9 java-jdk/8u131 ant/1.10.14 hdf5 python/3.12.1 intel impi mkl R  swiftt/1.6.2-python-3.12.1   # Running on MN5

# module load ANACONDA/2021.11 java R/3.4.0 
# source activate SwiftT

#export PYTHONPATH=${PYTHONPATH}:$EMEWS_PROJECT_ROOT/python
export PYTHONPATH=$EMEWS_PROJECT_ROOT/python
# export R_HOME=${R_HOME}:$EMEWS_PROJECT_ROOT/R
export R_HOME=$EMEWS_PROJECT_ROOT/R



# echo's anything following this standard out
set -x
# SWIFT_FILE=swift_run_sweep.swift
SWIFT_FILE=swift_run_sweep.swift
swift-t -n $PROCS $MACHINE -p -I $EMEWS_PROJECT_ROOT/swift/ $EMEWS_PROJECT_ROOT/swift/$SWIFT_FILE $CMD_LINE_ARGS
