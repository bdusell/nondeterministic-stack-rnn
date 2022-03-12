# Generate a zip file to be uploaded as supplementary material for the paper.
set -e
rm -f code.zip
zip -r code.zip . \
  --include \
    'data/syntaxgym/*' \
    'data/circuits.json' \
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
    '*.swp'
