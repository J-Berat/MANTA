FROM julia:1.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        fonts-dejavu-core \
        git \
        libdbus-1-3 \
        libegl1 \
        libegl-mesa0 \
        libfontconfig1 \
        libfreetype6 \
        libgl1 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libxrandr2 \
        libxrender1 \
        libxxf86vm1 \
        mesa-utils \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000

RUN groupadd --gid "${GID}" manta \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash manta \
    && mkdir -p /app /opt/julia-depot \
    && chown -R manta:manta /app /opt/julia-depot

ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PROJECT=/app \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    PATH="/usr/local/julia/bin:${PATH}"

WORKDIR /app
USER manta

COPY --chown=manta:manta Project.toml ./
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'

COPY --chown=manta:manta src ./src
COPY --chown=manta:manta demo ./demo
COPY --chown=manta:manta scripts ./scripts
COPY --chown=manta:manta test ./test
COPY --chown=manta:manta manta ./
RUN chmod +x /app/manta

RUN xvfb-run -a julia --project=. -e 'import Pkg; Pkg.precompile()'

COPY --chown=manta:manta README.md DOCKER.md ./

USER root
COPY docker-entrypoint.sh /usr/local/bin/manta-docker-entrypoint
RUN chmod 0755 /usr/local/bin/manta-docker-entrypoint \
    && chown -R manta:manta /app /opt/julia-depot

USER manta
ENV JULIA_PKG_PRECOMPILE_AUTO=1

ENTRYPOINT ["manta-docker-entrypoint"]
CMD []
