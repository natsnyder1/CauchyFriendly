#!/bin/bash

REPO_DIR=$PWD
#Configure the sbatch session
source $REPO_DIR/../../xsede_init.sh
#SBATCH parameters
ACCOUNT="cla299"
PARTITION=compute
TIME=02:00:00
NODES=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=128
MEM_REQ=100GB
CAUCHY_EXEC="test_leo5.py"
CAUCHY_EXEC_CMD_LINE_ARGS="" #"0 4 5500000 200000 64 2.0 0.0013 1.0 muc1e6_tr1e7" # rand_seed, win_size, max_num_terms, sat alt km, sat surf area, gamma_gauss, beta_1.3, beta_scale
JOB_NAME="leo5-200km-debug-h2boost2"

# JOB PARAM CHECKING
# 1.) Make sure we have a job name less than MAX_JOB_NAME_LENGTH
MAX_JOB_NAME_LENGTH=60
SQUEUE_FORMAT="%.18i %.9P %.${MAX_JOB_NAME_LENGTH}j %.12u %.8T %.10M %.10l %.6D %R"
WHITE_SPACE_COUNT=$( expr length "$JOB_NAME" - length `echo "$JOB_NAME" | sed "s/ //g"` )
if [ ${#JOB_NAME} -gt $MAX_JOB_NAME_LENGTH ] || [ $WHITE_SPACE_COUNT -gt 0 ]; then 
  echo "[ERROR:] JOB NAME: $JOB_NAME MUST BE LESS THAN 30 CHARACTERS IN LENGTH WITH NO SPACES! Please rename job! Exiting!"
  exit 1
fi
# The job name should be unique
JOB_NAME_DUPLICATE_COUNT=$( squeue --format="$SQUEUE_FORMAT" --me | awk '{ print $3 }' | grep -v NAME | grep -c -w $JOB_NAME )
if [ $JOB_NAME_DUPLICATE_COUNT -gt 0 ]; then 
  echo "[ERROR:] JOB NAME NOT UNIQUE! A JOB BY THE NAME OF $JOB_NAME is already running! CANNOT SUBMIT THIS JOB! PLEASE GIVE THE JOB A NEW NAME!"
  exit 1
fi

# Make a cfg sub-directory which stores this job's job_name.out and job_name.err. Name of the directory is job_name. Check for continue or restart logic for safety
JOB_CFG_DIR=$REPO_DIR/cfg/$JOB_NAME
JOB_SHELL_SCRIPT=${JOB_CFG_DIR}/rerun_job.sh
if [ -d $JOB_CFG_DIR ]; then
  echo "[WARNING:] JOB_CFG_DIR $JOB_CFG_DIR already exists, you may be corupting the previous launch params. Do you really wish to continue?"
  echo "Enter (y/n) below to continue (y) or abort this script (n)"
  read USER_INPUT
  if [ $USER_INPUT == "y" ]; then
    echo "User entered: $USER_INPUT. Continuing the launch!" 
  elif [ $USER_INPUT == "n" ]; then 
    echo "User entered: $USER_INPUT. Aborting the launch!" 
    exit 1
  else
    echo "User entered: $USER_INPUT. This is not an option (y=continue or n=abort), so aborting anyways! Please try again!"
    exit 1 
  fi
else
  mkdir $JOB_CFG_DIR
  touch ${JOB_CFG_DIR}/history_of_${JOB_NAME}.err
  touch $JOB_SHELL_SCRIPT
  chmod +x $JOB_SHELL_SCRIPT
fi

#SBATCH LAUNCH -- DO NOT MODIFY
echo "submitting job to slurm queue"
sbatch --job-name=$JOB_NAME --partition=$PARTITION --nodes=$NODES --ntasks-per-node=$NTASKS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --mem=$MEM_REQ --account=$ACCOUNT --export=ALL --no-requeue -t $TIME -e $JOB_CFG_DIR/$JOB_NAME.err -o $JOB_CFG_DIR/$JOB_NAME.out $REPO_DIR/run_job.sh $REPO_DIR/$CAUCHY_EXEC
if [ $? -eq 0 ]; then
  echo "job was submitted successfully!"
else
  echo "job was not submitted successfully!"
  exit 1
fi

#RESTORING CONFIGURATION RESULTS IN: JOB_CFG_DIR -- DO NOT MODIFY
echo "#!/bin/bash" > $JOB_SHELL_SCRIPT
echo "source $REPO_DIR/scripts/xsede_init.sh" >> $JOB_SHELL_SCRIPT
echo "sbatch --job-name=$JOB_NAME --partition=$PARTITION --nodes=$NODES --ntasks-per-node=$NTASKS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --mem=$MEM_REQ --account=$ACCOUNT --export=ALL --no-requeue -t $TIME -e $JOB_CFG_DIR/$JOB_NAME.err -o $JOB_CFG_DIR/$JOB_NAME.out $REPO_DIR/run_job.sh $REPO_DIR/$CAUCHY_EXEC $CAUCHY_EXEC_CMD_LINE_ARGS" >> $JOB_SHELL_SCRIPT


