# This is a placeholder script that you can modify to submit batch jobs to
# your computing cluster, if you have access to one.
# The first argument is a unique job name to identify the job.
# The second argument is a directory where additional outputs can be written,
# such as the stdout of the command being run. Use <job-name> to make the name
# of output files unique.
# The third argument is "gpu" if the job should be run on a GPU, and "cpu"
# otherwise.
# The rest of the arguments are the command that should be run. It will include
# `poetry run`.

set -e
set -o pipefail

usage() {
  echo "Usage: $0 <job-name> <output-dir> <device> <command>...

Queues batch jobs.
"
}

job_name=$1
output_dir=$2
device=$3
shift 3 || true
args=("$@")

if [[ ! $job_name || ! $output_dir || ! $device ]]; then
  usage >&2
  exit 1
fi

# You may want to replace this with some sort of call to `docker exec` or
# `singularity exec`.
echo "[$job_name] [$output_dir] [$device] ${args[@]}"
