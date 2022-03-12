# Open a shell in the Docker container, optionally building or pulling the
# Docker image first.
bash scripts/docker-exec.bash "$@" -- bash
