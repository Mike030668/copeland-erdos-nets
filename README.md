# copeland-erdos-nets

Research project: deterministic neural network weight initialization
via Copeland–Erdős digit streams and prime-block codebooks.

## Research Questions

1. Can Copeland–Erdős digit streams replace random seed-based initialization
   (Xavier/He) with equivalent or better convergence?
2. Can prime-number-based weight codebooks provide effective weight clustering
   for compression without accuracy loss?

## Setup

```bash
make install
make check-env
```

## Usage

```bash
make test
make lint
```

## Structure

- `src/copeland_erdos_nets/` — core library
- `tests/` — unit tests
- `configs/` — experiment configurations
- `scripts/` — reproducible experiment scripts
