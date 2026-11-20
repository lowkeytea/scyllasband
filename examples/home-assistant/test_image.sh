#!/bin/sh
# Image-level checks for the PR review hardening requests.
# Usage: sh test_image.sh <image-tag>
set -eu
IMAGE=${1:-scyllasband-wyoming:test}
echo "== non-root user =="
id_out=$(docker run --rm --entrypoint id "$IMAGE")
echo "$id_out"
case "$id_out" in *uid=0*) echo "FAIL runs as root"; exit 1;; esac
echo "== no git in image =="
if docker run --rm --entrypoint sh "$IMAGE" -c 'command -v git' >/dev/null 2>&1; then
  echo "FAIL git present"; exit 1
fi
echo "== pinned packages present =="
docker run --rm --entrypoint python "$IMAGE" -c 'from importlib.metadata import version; print("scyllasband", version("scyllasband")); print("wyoming", version("wyoming"))'
echo "== all image checks passed =="
