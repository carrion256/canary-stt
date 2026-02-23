#!/usr/bin/env bash
# Install canary-stt: symlink systemd service and bin scripts into place.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing from $REPO_DIR"

# Bin scripts → ~/.local/bin/
mkdir -p "$HOME/.local/bin"
ln -sfv "$REPO_DIR/bin/canary-dictate" "$HOME/.local/bin/canary-dictate"
ln -sfv "$REPO_DIR/bin/canary-dictate-cancel" "$HOME/.local/bin/canary-dictate-cancel"

# Systemd service → ~/.config/systemd/user/
mkdir -p "$HOME/.config/systemd/user"
ln -sfv "$REPO_DIR/systemd/canary-dictate.service" "$HOME/.config/systemd/user/canary-dictate.service"

# Reload and enable
systemctl --user daemon-reload
systemctl --user enable canary-dictate.service

echo "Done. Start with: systemctl --user start canary-dictate"
