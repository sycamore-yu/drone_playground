#!/usr/bin/env bash
set -euo pipefail

# Publish an isolated static gh-pages branch without changing the source
# checkout, the current main branch, or any historical results/checkpoints.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="https://github.com/sycamore-yu/drone_playground.git"
SOURCE_BRANCH="feat/github-pages-release-20261018"
OUTPUT_ROOT="${PAGES_BUILD_ROOT:-$ROOT/tmp/pages-deployment}"
SITE_DIR="$OUTPUT_ROOT/drone_playground"

command -v gh >/dev/null || { echo "GitHub CLI is required."; exit 2; }
gh auth status >/dev/null 2>&1 || {
  echo "GitHub CLI is not authenticated. Run: gh auth login --web -h github.com --git-protocol https --scopes repo"
  exit 2
}
gh auth setup-git
if [[ -e "$SITE_DIR" ]]; then
  echo "The release directory already exists: $SITE_DIR"
  echo "Use a fresh PAGES_BUILD_ROOT directory to avoid replacing saved artifacts."
  exit 2
fi
if git ls-remote --exit-code "$REMOTE" refs/heads/gh-pages >/dev/null 2>&1; then
  echo "Remote gh-pages already exists. This first-publication script refuses a non-fast-forward update."
  exit 2
fi
if git ls-remote --exit-code "$REMOTE" "refs/heads/$SOURCE_BRANCH" >/dev/null 2>&1; then
  echo "Remote source release branch already exists. Refusing to overwrite it."
  exit 2
fi

npm run build --prefix "$ROOT/docs/viewer"
python3 "$ROOT/tools/build_pages_demo.py" --output "$SITE_DIR" --base-path /drone_playground/

git init --initial-branch=gh-pages "$SITE_DIR"
git -C "$SITE_DIR" config user.name "$(git -C "$ROOT" config user.name)"
git -C "$SITE_DIR" config user.email "$(git -C "$ROOT" config user.email)"
git -C "$SITE_DIR" add -A
git -C "$SITE_DIR" commit -m "Publish selected real Drone Playground flight recordings"
git -C "$SITE_DIR" remote add origin "$REMOTE"

git -C "$SITE_DIR" push origin HEAD:refs/heads/gh-pages
git -C "$ROOT" push "$REMOTE" HEAD:refs/heads/"$SOURCE_BRANCH"

# Classic GitHub CLI login with repo scope can create the Pages configuration.
if ! gh api "repos/sycamore-yu/drone_playground/pages" > /dev/null 2>&1; then
  gh api -X POST "repos/sycamore-yu/drone_playground/pages" \
    -f 'source[branch]=gh-pages' -f 'source[path]=/' > /dev/null
fi

echo "Published the static gh-pages branch and source feature branch."
echo "Site URL: https://sycamore-yu.github.io/drone_playground/"
echo "GitHub Pages status:"
gh api "repos/sycamore-yu/drone_playground/pages" --jq '{status,html_url,source}'
