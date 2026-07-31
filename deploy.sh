#!/usr/bin/env bash
# Push this project to the UNO Q.
#
# App Lab reads its files directly from the board and keeps no local copy, so
# this is the only sync step. rsync is not installed on the board, hence tar
# over ssh.
set -euo pipefail

HOST="${SMARTBIN_HOST:-unoq}"
APP="${SMARTBIN_APP:-smartbin}"
DEST="ArduinoApps/$APP"

ssh "$HOST" "mkdir -p ~/$DEST"

# Never ship .cache: it holds the sketch build artifacts (~490 MB) and lives
# only on the board. Never ship captures or the event store either - they are
# runtime output, and copying stale ones over would be misleading.
tar cf - \
    --exclude='.git' \
    --exclude='.cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.db' \
    --exclude='captures' \
    app.yaml requirement.txt python sketch \
  | ssh "$HOST" "tar xf - -C ~/$DEST"

echo "deployed to $HOST:~/$DEST"
