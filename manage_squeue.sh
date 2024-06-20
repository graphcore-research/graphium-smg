#!/bin/bash

username="blazejb"
stuck_jobs=()  # Array to store job IDs of stuck jobs
declare -A job_models  # Associative array to store model names of stuck jobs
declare -A models

# Get the list of active job IDs for the specified user
active_jobs=$(squeue | grep "$username" | awk '{print $1}')
all_active_jobs=$(squeue | awk 'NR>1 {print $1}')

# Count the number of active jobs
my_job_count=$(echo "$active_jobs" | grep -c '^')
all_job_count=$(echo "$all_active_jobs" | grep -c '^')

echo "Active SLURM jobs associated with ${username}:"
echo $active_jobs
echo "Total number of my active jobs: $my_job_count"
echo "Total number of all active jobs: $all_job_count"
echo "----------"


# Loop through all slurm-*.out files in the current directory
for logfile in slurm-*.out; do
    if [[ -f "$logfile" ]]; then
        # Extract job ID from the filename
        jobid=${logfile#slurm-}
        jobid=${jobid%.out}
        model_name=$(grep -m 1 -oP '"model_name": "\K[^"]+' "$logfile")

        # Check if this job ID is in the list of active jobs
        if grep -q "$jobid" <<< "$active_jobs"; then
            # Check if the last line of the log file indicates a stuck job
            last_line=$(tail -n 1 "$logfile")
            if [[ "$last_line" == "wandb: Terminating and syncing runs. Press ctrl-c to kill." ]]; then
                # Add job ID to the stuck_jobs array
                stuck_jobs+=("$jobid")
                job_models["$jobid"]=$model_name
            fi

            # Update models array
            if [ -z "${models[$model_name]}" ]; then
                models["$model_name"]="$jobid"
            else
                models["$model_name"]+=", $jobid"
            fi
        else
            echo "Deleting log file for inactive job: $logfile"
            rm "$logfile"
        fi
    fi
done

# Print job IDs of stuck jobs and their corresponding dataset and model
if [ ${#stuck_jobs[@]} -ne 0 ]; then
    echo "Stuck job IDs with their corresponding dataset and model:"
    for job in "${stuck_jobs[@]}"; do
        echo "Job ID: $job, Model: ${job_models[$job]}"
        # Cancel the stuck job
        scancel "$job"
        echo "Cancelled stuck job ID: $job"
    done
else
    echo "No stuck jobs found."
fi

echo "----------"
echo "Models and their associated active job IDs:"
for model in "${!models[@]}"; do
    echo "$model: ${models[$model]}"
done