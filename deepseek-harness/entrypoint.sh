#!/bin/sh
set -eu

export DSH_HOME="${DSH_HOME:-/data/dsh}"
export NPM_CONFIG_REGISTRY="${NPM_CONFIG_REGISTRY:-https://registry.npmmirror.com}"
mkdir -p "$DSH_HOME" /workspace
cd /workspace

DSH_SPEC="@deepseek-ai/dsh@${DSH_VERSION}"
PROFILE_MANIFEST="$DSH_HOME/profiles/web/package.json"

if [ ! -f "$PROFILE_MANIFEST" ] || ! grep -q 'dsh-ui-auth' "$PROFILE_MANIFEST"; then
  npx --yes "$DSH_SPEC" plugin --profile web add -w "dsh-ui-auth@${UI_AUTH_VERSION}"
fi

BIND="$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -n 1)"
if [ -z "$BIND" ]; then
  echo "容器没有可用的 IPv4 地址，无法绑定 Web UI" >&2
  exit 1
fi

set -- npx --yes "$DSH_SPEC" web --no-open --host "$BIND" --port 3080

if [ -n "${DSH_TRUSTED_HOSTS:-}" ]; then
  for host in $DSH_TRUSTED_HOSTS; do
    set -- "$@" --trusted-host "$host"
  done
fi

if [ -n "${DSH_PUBLIC_URL:-}" ]; then
  set -- "$@" --public-url "$DSH_PUBLIC_URL"
fi

echo "Web UI 绑定 ${BIND}:3080，对外地址 ${DSH_PUBLIC_URL:-未设置}。"
exec "$@"
