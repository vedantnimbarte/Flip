#!/bin/sh
# dlm updater — re-run the installer to fetch the latest release.
#
#   curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/update.sh | sh
#
# Honors the same env as install.sh (DLM_INSTALL_DIR, DLM_CPU). It just
# reinstalls over the existing binary, so updating == installing again.
set -eu

REPO="vedantnimbarte/dlm"
u="https://raw.githubusercontent.com/${REPO}/main/install.sh"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$u" | sh
elif command -v wget >/dev/null 2>&1; then
  wget -qO- "$u" | sh
else
  printf 'error: need curl or wget installed\n' >&2
  exit 1
fi
