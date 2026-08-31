#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="$repo_root/.baseline-work/upstream"
mkdir -p "$work_root"

prepare_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local target="$work_root/$name"

  if [[ ! -d "$target/.git" ]]; then
    git clone "$url" "$target"
  fi

  if ! git -C "$target" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$target" fetch origin "$commit"
  fi

  # Recover a checkout left empty by an interrupted clone before applying the
  # ordinary dirty-worktree check.
  if [[ -z "$(find "$target" -mindepth 1 -maxdepth 1 ! -name .git -print -quit)" ]]; then
    git -C "$target" checkout --detach "$commit"
  fi

  # Upstream imports create untracked __pycache__ directories. Protect tracked
  # source edits, while allowing those disposable ignored-work-area artifacts.
  if [[ -n "$(git -C "$target" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to change dirty upstream checkout: $target" >&2
    exit 1
  fi
  git -C "$target" checkout --detach "$commit"
  printf '%s %s\n' "$name" "$(git -C "$target" rev-parse HEAD)"
}

prepare_repo \
  LILAC \
  https://github.com/logpai/LILAC.git \
  163d199f95ddf419044c53681f2b474a9c2d45ce
prepare_repo \
  LogBatcher \
  https://github.com/LogIntelligence/LogBatcher.git \
  7d4768f1ec30a25c6581f4423c03643f6fbd266c
