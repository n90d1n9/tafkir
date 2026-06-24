#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing Tafkir VS Code extension..."


cd "$PROJECT_ROOT"

code --install-extension "$PROJECT_ROOT/integration/vscode/tafkir-vscode/out/tafkir-vscode-0.1.0.vsix"

echo "✅ Successfully installed Tafkir VS Code extension."
