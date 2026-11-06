FROM docker.1ms.run/node:22-bookworm-slim

ARG DSH_VERSION=0.2.1-rc.5
ARG UI_AUTH_VERSION=0.7.1

ENV DSH_VERSION=${DSH_VERSION} \
    UI_AUTH_VERSION=${UI_AUTH_VERSION} \
    DSH_HOME=/data/dsh \
    NPM_CONFIG_REGISTRY=https://registry.npmmirror.com \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false \
    CI=true \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    DSH_TRUSTED_HOSTS=139.196.93.191 \
    DSH_PUBLIC_URL=http://139.196.93.191/deepseek-harness/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 make g++ iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g pnpm@9 \
    && pnpm config set registry https://registry.npmmirror.com \
    && mkdir -p /data/dsh /workspace \
    && chown -R node:node /data /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

USER node
WORKDIR /workspace

RUN npx --yes "@deepseek-ai/dsh@${DSH_VERSION}" plugin --profile web add -w "dsh-ui-auth@${UI_AUTH_VERSION}"

EXPOSE 3080

ENTRYPOINT ["/entrypoint.sh"]
