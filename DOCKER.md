# Docker

Build the CartaJulia image:

```bash
docker build -t carta-julia .
```

Run the test suite inside the container:

```bash
docker run --rm carta-julia julia --project=. -e 'import Pkg; Pkg.test()'
```

Run a headless GLMakie smoke test:

```bash
docker run --rm carta-julia xvfb-run -a julia --project=. -e 'using GLMakie; display(GLMakie.Figure()); sleep(2)'
```

## Interactive window on Linux

Allow local Docker containers to use your X11 server, then run the demo:

```bash
xhost +local:docker
docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD/demo/output:/app/demo/output" \
  carta-julia
```

Open a FITS file by mounting its directory:

```bash
docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$PWD/data:/data:ro" \
  carta-julia /data/cube.fits
```

## Interactive window on macOS with XQuartz

Install and start XQuartz, enable "Allow connections from network clients" in
XQuartz settings, then restart XQuartz. In a terminal:

```bash
xhost +localhost
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD/demo/output:/app/demo/output" \
  carta-julia
```

For a FITS file on macOS:

```bash
docker run --rm -it \
  -e DISPLAY=host.docker.internal:0 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD/data:/data:ro" \
  carta-julia /data/cube.fits
```

The image uses Julia 1.11, matching the package compatibility range in
`Project.toml`. `LIBGL_ALWAYS_SOFTWARE=1` favors Mesa software rendering, which
is usually more predictable for GLMakie in containers.
