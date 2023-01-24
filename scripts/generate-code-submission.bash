set -e
set -u
set -o pipefail

rm -f code.zip
zip -r code.zip . \
  --include \
    'Dockerfile*' \
    '.docker*' \
    'experiments/*' \
    'poetry.lock' \
    'pyproject.toml' \
    'README.md' \
    'scripts/*' \
    'src/*' \
    'tests/*' \
  --exclude \
    '*/__pycache__/*' \
    '*.swp' \
    '*/.pytest_cache/*'
