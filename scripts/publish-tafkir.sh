#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Publishing Tafkir to GitHub Packages..."

if [[ -z "${GITHUB_ACTOR:-}" || -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_ACTOR and GITHUB_TOKEN environment variables must be set."
  echo "You can generate a classic Personal Access Token with 'read:packages' and 'write:packages' scopes."
  exit 1
fi

cd "$PROJECT_ROOT"

# This publishes to the GitHubPackages repository configured in build.gradle.kts
./gradlew --no-daemon publishAllPublicationsToGitHubPackagesRepository

echo "✅ Successfully published to GitHub Packages."
