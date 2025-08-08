# ./muininn/AGENTS.md

The project directory structure is as follows:

muininn/
├─ muininn/
│  ├─ c/
│  │  ├─ cudaIpcShim.c
│  │  └─ meson.build
│  ├─ cython/
│  │  ├─ headers/
│  │  │  └─ cudaIpcShim.h
│  │  ├─ __init__.py
│  │  └─ setupCython.py
│  ├─ docs/
│  │  ├─ api.md
│  │  └─ architecture.md
│  ├─ examples/
│  │  └─ minimalClient.py
│  ├─ scripts/
│  │  ├─ dev.ps1
│  │  └─ dev.sh
│  └─ src/
│     └─ muininn/
│        ├─ config/
│        │  ├─ __init__.py
│        │  ├─ defaults.toml
│        │  ├─ loader.py
│        │  └─ schema.json
│        ├─ core/
│        │  ├─ policy/
│        │  │  ├─ __init__.py
│        │  │  ├─ kvStore.py
│        │  │  └─ placementPolicy.py
│        │  ├─ __init__.py
│        │  ├─ daemon.py
│        │  ├─ deviceDiscovery.py
│        │  ├─ scheduler.py
│        │  └─ topology.py
│        ├─ gpu/
│        │  ├─ __init__.py
│        │  ├─ events.py
│        │  ├─ streams.py
│        │  └─ worker.py
│        ├─ integration/
│        │  ├─ __init__.py
│        │  ├─ grendelApi.py
│        │  └─ huginnApi.py
│        ├─ ipc/
│        │  ├─ __init__.py
│        │  ├─ client.py
│        │  ├─ messages.py
│        │  └─ server.py
│        ├─ memory/
│        │  ├─ bigMode/
│        │  │  ├─ __init__.py
│        │  │  ├─ mappedHostHuge.pxd
│        │  │  └─ mappedHostHuge.pyx
│        │  ├─ fastMode/
│        │  │  ├─ __init__.py
│        │  │  ├─ vramShared.pxd
│        │  │  └─ vramShared.pyx
│        │  ├─ __init__.py
│        │  ├─ dedupe.py
│        │  ├─ grendelRam.py
│        │  ├─ nvmeSpill.py
│        │  ├─ pager.py
│        │  ├─ prefetcher.py
│        │  └─ slice.py
│        ├─ observability/
│        │  ├─ __init__.py
│        │  ├─ logging.py
│        │  ├─ metrics.py
│        │  └─ tracing.py
│        ├─ tools/
│        │  ├─ __init__.py
│        │  └─ muininnCtl.py
│        ├─ __init__.py
│        ├─ __main__.py
│        ├─ cli.py
│        └─ version.py
├─ src/
│  └─ muininn/
│     ├─ __init__.py
│     ├─ __main__.py
│     └─ cli.py
├─ .editorconfig
├─ AGENTS.md
├─ CHANGELOG.md
├─ LICENSE
├─ pyproject.toml
├─ README.md
└─ setup.py