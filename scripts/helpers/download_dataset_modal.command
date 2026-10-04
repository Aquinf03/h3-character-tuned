#!/bin/zsh
# Compatibility wrapper. Canonical path: src/data/download_dataset_modal.command
exec "$(cd "$(dirname "$0")/../.." && pwd)/src/data/download_dataset_modal.command"
