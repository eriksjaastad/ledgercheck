# Ledger Check container image: the offline judge gate by default.
#
# CI builds this image, smoke-tests it and publishes it to
# ghcr.io/eriksjaastad/ledgercheck from main (.github/workflows/image.yml).
#
# Build and run on your own machine (Docker is not needed for tests):
#
#     docker build -t ledgercheck:dev .
#     docker run --rm ledgercheck:dev                 # ledgercheck judge (offline, exits 0/1/2)
#     docker run --rm ledgercheck:dev --help          # any other subcommand
#
# Two stages: "build" turns the source tree into a wheel; the runtime stage
# installs only that wheel. The package has no runtime dependencies. The
# optional ``langfuse`` extra is left out unless you build with
# ``--build-arg WITH_LANGFUSE=1``; LANGFUSE_* keys set without the SDK make
# tracing fail at start (ledgercheck/observability.py).
#
# ``ledgercheck serve`` is loopback-only with no login (ledgercheck/web.py),
# so this image does not expose it. Azure wiring lives in terraform/ and is
# a write-only reference: nothing in this repo applies it (scripts/deploy.py
# refuses).

FROM python:3.11-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY ledgercheck ./ledgercheck
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.11-slim
ARG WITH_LANGFUSE=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY --from=build /wheels /wheels
RUN wheel="$(ls /wheels/ledgercheck-*.whl)" \
    && if [ "$WITH_LANGFUSE" = "1" ]; then wheel="${wheel}[langfuse]"; fi \
    && pip install --no-cache-dir "$wheel" \
    && useradd --create-home --uid 10001 ledgercheck
USER ledgercheck
WORKDIR /home/ledgercheck
ENTRYPOINT ["ledgercheck"]
CMD ["judge"]
