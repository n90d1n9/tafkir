#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/target"
BIN="$BIN_DIR/tafkir"

if [[ ! -x "$BIN" ]]; then
  echo "tafkir native binary not found at $BIN" >&2
  echo "Build it with: mvn -f inference-tafkir/ui/tafkir-cli/pom.xml -Pnative -DskipTests clean package" >&2
  exit 1
fi

# Avoid deprecated console env keys leaking into native runtime.
unset QUARKUS_LOG_CONSOLE_JSON
unset QUARKUS_LOG_CONSOLE_ENABLE

# Default to disabling file logging in sandboxed runs.
: "${TAFKIR_CLI_FILE_LOG_ENABLED:=false}"
export TAFKIR_CLI_FILE_LOG_ENABLED

exec "$BIN" "$@"
