# EasiGP Docker images

Ready-made images of EasiGP are published on Docker Hub: https://hub.docker.com/r/shunichirot/easigp. They are several gigabytes each, which is why they are not stored in this repository. The Docker Hub tags page lists the exact tag for each version.

| Version | Tag | Includes |
|---|---|---|
| Full | `<version>` | Everything, including the Claude Code integration that generates a gene-interaction network with FLASH-P (needs a Claude access token) |
| Light | `<version>_light` | The same EasiGP, without the Claude Code / FLASH-P integration |

## How to use an image

Follow **Option A** in the [main README](../README.md): Docker on a local PC/laptop, or Apptainer on an HPC (which can build from Docker Hub directly or from a `.tar` file). Inside the image, EasiGP lives in `workspace/EasiGP`, and a folder from your own computer can be mounted at `/workspace/data`.

## Building the images yourself

The `Dockerfile` in this folder builds the images. Its own comments give the exact `docker build` commands, including `--build-arg INSTALL_FLASHP=false`, which produces the Light version.

## ASReml-R is not included

ASReml-R (used by the optional mixed-model adjustment) is commercial software that needs its own paid licence, so it is not part of these images. To use it, set EasiGP up from source code (**Option B** in the main README) and follow the ASReml-R section there.
