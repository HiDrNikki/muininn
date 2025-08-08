# Muninn Roadmap

> **Muninn** — the shared-memory broker for GPUs — is a **GPU implementation of _rememory on steroids_**.
> It prioritises **capacity over latency**, exposes **shared buffers** to multiple processes,
> and integrates with **Grendel** (execution/tiering) and **Huginn** (plans/hints).

_Last updated: 2025-08-08 17:39:02 UTC_

---

## Table of Contents

- [0. Vision](#0-vision)
- [1. Scope & Non-Goals](#1-scope--non-goals)
- [2. Architecture Overview](#2-architecture-overview)
- [3. Modes & Semantics](#3-modes--semantics)
- [4. Interfaces](#4-interfaces)
- [5. Detailed Milestones](#5-detailed-milestones)
- [6. Performance Targets & Budgets](#6-performance-targets--budgets)
- [7. Observability & Operations](#7-observability--operations)
- [8. Reliability, Safety, and Recovery](#8-reliability-safety-and-recovery)
- [9. Security, Multi-Tenancy, and Threat Model](#9-security-multi-tenancy-and-threat-model)
- [10. Testing Strategy](#10-testing-strategy)
- [11. Compatibility Matrix](#11-compatibility-matrix)
- [12. Configuration & Policy](#12-configuration--policy)
- [13. CLI & Tooling](#13-cli--tooling)
- [14. Developer Guidelines](#14-developer-guidelines)
- [15. API Specifications (Non-Normative)](#15-api-specifications-non-normative)
- [16. Error Codes & UX](#16-error-codes--ux)
- [17. Examples & Workflows](#17-examples--workflows)
- [18. Acceptance Criteria](#18-acceptance-criteria)
- [19. Open Questions & Decisions Log](#19-open-questions--decisions-log)
- [20. Roadmap Timeline](#20-roadmap-timeline)
- [21. Future Work](#21-future-work)
- [22. Glossary](#22-glossary)
- [23. Changelog for this Document](#23-changelog-for-this-document)

---

## 0. Vision

Muninn exists to make **big, shared, device-accessible memory** practical and safe.

- It offers **two complementary modes**:
  - **fastMode (`vramShared`)** — true VRAM sharing on the **same GPU** via CUDA IPC memory + IPC events; cross-GPU hand-offs via P2P or staged copies.
  - **bigMode (`mappedHostHuge`)** — a **single, giant, contiguous** buffer backed by **pinned host RAM** with optional **NVMe spill**; device-mapped so kernels can dereference directly.
- It manages **leases, refcounts, and synchronization** across processes.
- It is **topology-aware**, preferring **P2P** routes where possible, and falling back to **pinned host** staging when necessary.
- It integrates cleanly with:
  - **Grendel** (execution/tiering, promotion engines, watchdogs, metrics),
  - **Huginn** (plans/hints, hotRange annotations, lifetime windows).

**Guiding slogan:** _“The goal isn’t fast, it’s big — and correct.”_

### Primary Outcomes
1. Stable, predictable **shared buffer semantics** for multi-process GPU workloads.
2. **Capacity-first** memory via bigMode, with optional per-GPU **promotion** for hot slices.
3. **Zero-copy** intra-GPU sharing (fastMode) with robust fencing via **IPC events**.
4. **Operational** visibility: metrics, logs, and debuggability from day one.
5. **Safety**: lease TTLs, revocation, and no-dangling-handle guarantees.

### Non-Goals (v1, reiterated)
- True VRAM pooling into a single flat heap across devices.
- Cross-machine shared page mapping (GPUDirect RDMA transfers are not shared mappings).
- Windows support.
- cudaMallocManaged/UVA-driven automatic migration and coherence.
- Kernel-mode driver changes.

---

## 1. Scope & Non-Goals

### In Scope
- Shared VRAM on same GPU (CUDA IPC mem + events).
- P2P hand-offs between GPUs (NVLink/PCIe peerable) and staged fallbacks.
- Big contiguous mapped host buffer for **capacity-first** workloads.
- Promotion/writeback pipeline to per-GPU scratch VRAM.
- Leases/refcounts, quotas, observability, and deterministic routing.
- NVMe-backed spill for bigMode (optional M5).

### Explicit Non-Goals (v1)
- Flattening multiple VRAMs into a single linear address space.
- Transparent cross-GPU pointer dereference.
- Cross-node shared mappings (RDMA is explicit copy path only).
- Automatic, invisible auto-promotion that changes semantics without a hint/flag.

---

## 2. Architecture Overview

```
(App) ── LD_PRELOAD shim ──► Huginn Plan (hints/leases) ─► Muninn (broker) ─► Grendel (exec/tiering)
                                         │                        │
                                         │                        ├─► VRAM (per-GPU scratch / fastMode allocations)
                                         │                        ├─► Pinned Host RAM (Grendel RAM; bigMode backing)
                                         │                        └─► NVMe spill (optional)
                                         │
                                     Telemetry ◄────────────────────────────── Metrics/Logs
```

**Core components**
- **allocator** — Creates bigMode (mappedHostHuge) and fastMode (vramShared) buffers, returns opaque handles.
- **syncManager** — CUDA IPC events, host-side control blocks (`rememory`-style atomics, futex).
- **promotionManager** — Promotes `[offset, length]` windows from bigMode into per-GPU scratch VRAM; manages writeback.
- **leaseManager** — Tracks refcounts, lease TTLs, renewals, and revocations.
- **policyEngine** — Quotas, placement, topology-aware routing; admission control for mapped windows.
- **topologyManager** — Discovers P2P pairs, NUMA nodes, PCIe switch locality, BAR aperture health.
- **observability** — Prometheus metrics, structured logs, tracing spans.
- **cli** — `muninctl` for humans and automation.

**Key invariants**
- Opaque handles are **capabilities** that must be broker-validated per process and per tenant.
- **Owner must outlive importers** for fastMode; broker enforces & cleans up.
- Promotion never changes the **caller-visible pointer** for bigMode; it supplies a **scratch pointer** for the launch window.

---

## 3. Modes & Semantics

### 3.1 fastMode (`vramShared`)
- **Backed by:** VRAM on a single GPU (the **owner** GPU).
- **Share across processes:** via CUDA IPC memory handles and **IPC events** for fencing.
- **Cross-GPU consumption:** copy-based (P2P if eligible, staged via pinned host otherwise).
- **Lifetime:** exporting/broker process owns allocation; importers hold leases.
- **Best for:** same-GPU multi-process zero-copy, moderate-sized shared tensors, latency-sensitive reads on owner GPU.

**Semantics**
- Multiple importers can map the same memory region in their contexts.
- Writers must `mnnSignal(evt)` after kernel completes; readers `mnnWait(evt)` before consumption.
- Broker cleans up when owner exits and all importers drop leases.

### 3.2 bigMode (`mappedHostHuge`)
- **Backed by:** Pinned host RAM (very large), with optional NVMe spill.
- **Pointer model:** Each process maps a **single contiguous device pointer** (`mapped host`) that kernels can dereference.
- **Performance:** PCIe/NVLink-to-host bandwidth; higher latency than VRAM; **capacity-first**.
- **Promotion:** Optional, explicit promotion of hot windows to per-GPU scratch VRAM for launch-time speedups.
- **Best for:** datasets/models/streams much larger than VRAM, read-mostly or streaming access.

**Semantics**
- The contiguous pointer is stable; promotion uses **separate scratch pointers** for kernels during the promoted launch.
- Writeback depends on policy (`readOnly`, `readMostly`, `readWrite`).

---

## 4. Interfaces

### 4.1 Shim ⇄ Muninn (UDS/IPC) — C, camelCase

```c
typedef struct { unsigned char bytes[64]; } mnnMemHandle;
typedef struct { unsigned char bytes[64]; } mnnEvtHandle;

int mnnAlloc(size_t bytes, int preferredGpu, const char* mode,
             mnnMemHandle* outMem, mnnEvtHandle* outEvt, int* outGpu, uint64_t* outSize);

int mnnOpen(const mnnMemHandle* mem, unsigned int flags, void** outDevPtr);
int mnnWait(const mnnEvtHandle* evt, cudaStream_t stream);
int mnnSignal(const mnnEvtHandle* evt, cudaStream_t stream);

int mnnPromote(uint64_t offset, uint64_t length, int targetGpu, void** outScratchPtr);
int mnnWriteback(uint64_t offset, uint64_t length, int sourceGpu, const void* scratchPtr);

int mnnDescribe(const mnnMemHandle* mem, char* jsonBuf, size_t jsonBufLen);
int mnnClose(void* devPtr);
int mnnFree(const mnnMemHandle* mem);
```

**Notes**
- `mnnPromote` guarantees a valid scratch pointer only for the **caller’s next launch window**; after that it may be recycled.
- `mnnWriteback` is implicit for `readWrite` but callable explicitly for `readMostly` when required.

### 4.2 Huginn ⇄ Muninn (REST/gRPC) — Non-normative schemas

`POST /buffers`
```json
{
  "size": 17179869184,
  "mode": "bigMode",
  "preferredGpu": 0,
  "name": "weightsA",
  "access": "readMostly",
  "tenant": "org/acct/job123",
  "hints": {
    "hotRanges": [[0, 268435456], [536870912, 134217728]],
    "lifetimeMs": 600000,
    "priority": "normal"
  }
}
```

`POST /buffers/{id}/open`
```json
{ "process": 43210, "tenant": "org/acct/job123", "purpose": "reader" }
```

`POST /buffers/{id}/promote`
```json
{ "offset": 1073741824, "length": 268435456, "targetGpu": 1 }
```

`DELETE /buffers/{id}` — revokes on last refcount, or forces if lease expired.

### 4.3 Muninn ⇄ Grendel
- Topology queries, reservation of per-GPU scratch VRAM pools, copy orchestration (P2P vs staged), thermal/backoff hooks.
- Promotion scheduling to overlap H2D with compute where safe.

---

## 5. Detailed Milestones

The following milestones chart a path from **capacity-first MVP** to **production-ready broker** with observability and optional cross-node capabilities.

---

### **M0 — bigMode Core (mappedHostHuge)**
**Goals**

- Pinned host region allocator with contiguous virtual address space (NUMA-aware).
- Device mapping (mapped host) and per-process open/close with admission control.
- Control blocks in host memory (`rememory`-style): version, flags, seq counters, ring cursors, refcounts.
- Lease model: TTLs, renewals on use, broker-authoritative refcounts.
- Basic quotas for pinnedHostBytes per tenant.
- CLI: `muninctl alloc|open|close|free|describe`.
- Docs: trade-offs, access patterns (coalesced vs random), expected bandwidth.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M1 — fastMode (vramShared) on Same GPU**
**Goals**

- VRAM allocation via broker; export/import **CUDA IPC** memory handles.
- IPC events for GPU-side synchronization; helpers in shim for `mnnWait`/`mnnSignal`.
- Owner-must-outlive-importers enforcement; watchdog cleanup.
- Crash-safety tests: importer death; owner death after importer close.
- Metrics: open latency histogram; active importers gauge.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M2 — Cross-GPU (P2P + Staged Fallback)**
**Goals**

- Topology discovery (NVLink/PCIe) and peerability matrix caching.
- Route selection with clear logs: `route=p2p|staged`, reason codes.
- Peer enablement management per process; staged pipeline overlaps H2D/D2H.
- Throughput/latency benchmarks; acceptance delta between P2P and staged.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M3 — Promotion & Writeback**
**Goals**

- `mnnPromote(offset,length,targetGpu)` returns launch-window scratch pointer.
- Dirty-range tracking selectable granularity (e.g., 64 KiB slices).
- Writeback policy: `readOnly` (skip), `readMostly` (lazy/explicit), `readWrite` (eager).
- Huginn hint integration: hotRanges, lifetime windows.
- Promotion overlap with compute via Grendel copy streams.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M4 — Quotas, Leases, and Observability**
**Goals**

- Per-tenant quotas: VRAM, pinned host, NVMe (if enabled).
- Lease renewals & revocation path with broker-side reason codes.
- Prometheus metrics, structured logs with correlation IDs.
- `muninctl`: list/describe/promote/evict/drain; human-readable reports.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M5 — NVMe Spill & Optional GPUDirect RDMA**
**Goals**

- NVMe spill tier for bigMode with aligned IO, prefetch, and backpressure.
- Admission control when RAM pressure high; hot/cold classification.
- Optional GPUDirect RDMA transfer path for cross-node hand-offs.
- Sustained throughput validation; per-device pacing.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---
### **M6 — Auto-Promotion Heuristics & Tuning**
**Goals**

- Opt-in heuristics based on access counters and recent hints.
- BAR windowing controls; import limiters; backpressure thresholds.
- Configurable promotion budgets per GPU, per tenant.
- A/B evaluation harness; deterministic mode to disable heuristics.
**Deliverables**

- Demos, docs, and benchmarks demonstrating correctness and expected performance.
**Exit Criteria**

- All features above implemented with tests; soak stable; metrics in dashboards.

---

## 6. Performance Targets & Budgets

> Targets are **order-of-magnitude** guides. Real numbers depend on topology and workload.

### bigMode (mappedHostHuge)
- **Bandwidth:** PCIe Gen4 x16 ~11–13 GB/s one way; NVLink higher where present.
- **Latency:** Higher than VRAM; favour coalesced, streaming reads.
- **Goal:** Stable throughput under multi-process reads; avoid regressions >10% after feature additions.

### fastMode (vramShared)
- **Same-GPU:** VRAM-class bandwidth for kernels; open latency < 2 ms p50.
- **Cross-GPU:** P2P preferred; staged route aims for ≥80% of theoretical H2D/D2H aggregate.

### Promotion
- **Hit Ratio:** ≥70% useful-byte ratio for hinted ranges.
- **Writeback Overhead:** ≤30% of promotion time for readWrite workloads (amortised).

### Broker Overheads
- **Shim call latency:** < 100 µs budget for alloc/open where cached; < 1 ms otherwise.
- **Refcount operations:** lock-free fast path; bounded retries under contention.

---

## 7. Observability & Operations

### Metrics (Prometheus)
| Metric | Type | Labels | Description |
|-------|------|--------|-------------|
| `muninn_buffers_total` | Counter | `mode` | Total buffers allocated by mode. |
| `muninn_bytes_total` | Counter | `tier` | Bytes allocated cumulatively per tier (`vram`, `host`, `nvme`). |
| `muninn_opens_total` | Counter |  | Number of successful opens (per process). |
| `muninn_open_latency_ms` | Histogram | `mode` | Open call latency distribution. |
| `muninn_promotions_total` | Counter | `gpu` | Number of promotions executed. |
| `muninn_bytes_promoted_total` | Counter | `gpu` | Total bytes promoted. |
| `muninn_route_selection_total` | Counter | `route` | Chosen routes: `p2p` vs `staged`. |
| `muninn_leases_active` | Gauge | `tenant` | Active leases per tenant. |
| `muninn_revocations_total` | Counter | `reason` | Revocations by reason code. |
| `muninn_errors_total` | Counter | `code` | Errors by code. |
| `muninn_bar_pressure` | Gauge | `gpu` | BAR aperture utilisation estimate. |
| `muninn_nvme_q_depth` | Gauge | `device` | NVMe queue depth (if enabled). |

### Logs
- Structured JSON with fields: `ts`, `level`, `bufferId`, `leaseId`, `tenant`, `gpu`, `route`, `bytes`, `offset`, `durationMs`, `errorCode`.
- Correlate with copy/launch spans via shared `traceId`/`spanId` (OpenTelemetry).

### Dashboards
- **Capacity View:** bytes by tier; buffers by mode; top tenants.  
- **Hot Path View:** open latency; promotion throughput; error rate; route selection.  
- **Health:** BAR pressure; NVMe backlog; watchdog resets; lease revocations.

### Alarms
- High BAR pressure sustained > 60% for 5 min.
- Promotion failure rate > 1% over 15 min window.
- Lease expiry misuses; revocation spikes.
- NVMe queue depth > threshold (configurable).

---

## 8. Reliability, Safety, and Recovery

- **Refcounts & Leases:** Broker authoritative; shim updates are hints validated server-side.
- **Watchdogs:** Detect abandoned imports; reclaim resources after grace period.
- **Thermal/Backoff:** Coordinate with Grendel to throttle promotions/copies under thermal or power constraints.
- **Graceful Degradation:** When P2P unavailable, automatically fall back to staged routes with explicit reason codes.
- **Crash Recovery:** On broker restart, reconstruct state from durable journals and active control blocks; orphan GC reclaims unreachable allocations.
- **Consistency Invariants:**
  - Promotion never changes base bigMode pointer; scratch pointers are ephemeral.
  - Owner must outlive importers in fastMode.
  - Writes are fenced before reads (`signal` → `wait`).

---

## 9. Security, Multi-Tenancy, and Threat Model

- **Capabilities:** Handles are unguessable and scoped by tenant/job; short TTLs; renewal on use.
- **UDS Permissions:** `0700` by default; per-tenant sockets if required.
- **Input Validation:** All IPC payloads validated; length checks; mode gating.
- **Quotas:** Hard & soft quotas per tenant for VRAM, pinned host, NVMe.
- **Audit Logs:** Handle issuance, promotion, revocation, and route selection decisions.
- **Threats & Mitigations:**
  - **Handle leakage:** leases expire; broker revokes on misuse or process exit detection.
  - **BAR exhaustion:** admission control; windowed mapping; backpressure.
  - **Noisy neighbour:** per-tenant bandwidth/promotion budgets; fair scheduling.
  - **TOCTOU on control blocks:** versioned atoms; sequence checks; broker arbitration.

---

## 10. Testing Strategy

### Test Types
1. **Unit Tests** — allocators, control blocks, refcounts, lease renewal/expiry, error codes.
2. **Integration Tests** — CUDA IPC mem/event flows; P2P vs staged correctness; promotion API.
3. **Performance Tests** — bigMode throughput under contention; promotion ROI; P2P vs staged bandwidth.
4. **Stress/Soak** — long-running multi-process readers/writers; lease churn; BAR windowing.
5. **Fault Injection** — owner death; importer death mid-flight; forced revocation; NVMe device hiccups.
6. **Security Tests** — handle forgery attempts; tenant isolation; quota enforcement.
7. **Determinism Tests** — plan → result reproducibility given same hints/topology.

### CI Matrix (illustrative)
- CUDA driver/runtime versions: `n-1, n` where feasible.
- GPU classes: single GPU; dual GPU (peerable); dual GPU (non-peerable).
- NUMA configurations: 1-socket/2-socket hosts.
- Optional: NVMe present/absent; RDMA present/absent.

### Bench Harness
- Python+Cython microbenchmarks for open/close/promote; C-based kernel loops for realistic access patterns.
- Emit CSV; gate PRs on p50/p95 thresholds not regressing beyond 10% without a waiver.

---

## 11. Compatibility Matrix

| Area | v1 Support | Notes |
|------|------------|-------|
| OS | Linux x86-64 | UDS + LD_PRELOAD; Windows out of scope v1 |
| GPU Vendor | NVIDIA | CUDA Runtime + Driver APIs |
| CUDA IPC Mem | Yes | Same-GPU VRAM sharing |
| CUDA IPC Events | Yes | Fencing between processes |
| P2P | Yes | Topology-aware; staged fallback |
| Mapped Host | Yes | bigMode backbone |
| NVMe Spill | M5 | Optional |
| GPUDirect RDMA | M5 (optional) | Transfers only, not shared mappings |
| Huginn Integration | Yes | Hints/leases; hotRanges |
| Grendel Integration | Yes | Promotion scheduling; topology; watchdogs |

---

## 12. Configuration & Policy

```toml
# muninn/config/defaults.toml

[server]
udsPath = "/var/run/muninn/muninn.sock"
logLevel = "info"
metricsPort = 9618
enableTracing = true

[memory]
bigModeMaxBytes = "512G"         # admission cap for mappedHostHuge
fastModeMaxBytes = "64G"          # VRAM broker cap (broker-managed)
nvmeEnabled = false
nvmePath = "/var/lib/muninn/spill"
nvmePrefetch = "64M"
nvmeAioDepth = 64

[promotion]
enablePromotion = true
defaultSliceBytes = "64K"
defaultPolicy = "readMostly"
perGpuScratchBytes = "8G"
maxConcurrentPromotions = 2

[topology]
enablePeerProbing = true
preferP2P = true

[tenancy]
defaultPinnedHostQuota = "256G"
defaultVramQuota = "16G"
defaultNvmeQuota = "1T"

[security]
leaseTtlMs = 120000
handleBytes = 64
udsPerm = "0700"
```

**Policy rules**
- Tenants exceeding soft quotas trigger warnings; hard quotas block allocations.
- Promotion budgets enforced per GPU and per tenant; over-budget requests are queued or rejected.

---

## 13. CLI & Tooling

### `muninctl`
```
muninctl alloc --size 128G --mode bigMode --name datasetA --tenant org/acct/job123
muninctl describe --name datasetA
muninctl promote --name datasetA --offset 4G --length 512M --gpu 0
muninctl list --mode bigMode
muninctl evict --name datasetA --tenant org/acct/job123
muninctl drain --reason maintenance
```

**Output examples**
```
BUFFER  MODE      SIZE     TENANT            GPU  IMPORTERS  LEASE  AGE
datasetA bigMode  128G     org/acct/job123   -    3          ok     12m
weightsX fastMode 4G       org/acct/job222   0    1          ok     3m
```

### Developer Tools
- `muninnd --debug` (verbose logs, trace spans).  
- `muninn-top` (TUI: bytes by tier, routes, promotions).  
- `muninn-prof` (bench harness wrapper).

---

## 14. Developer Guidelines

- **Language:** Python 3.11+ for daemon; Cython for hot paths; C for shim only.
- **Style:** **camelCase** for all internal identifiers; follow repository lint rules.
- **Docs:** Update `docs/` and this roadmap with any surface change.
- **Reviews:** Require at least one maintainer approval; green CI.
- **Comment discipline:** Explain non-obvious synchronization, memory ordering, and topology decisions.
- **Determinism:** Provide switches to disable heuristics; make runs reproducible with the same config/hints.

---

## 15. API Specifications (Non-Normative)

### 15.1 Shim C Headers (selected)
```c
int mnnAlloc(size_t bytes, int preferredGpu, const char* mode,
             mnnMemHandle* outMem, mnnEvtHandle* outEvt, int* outGpu, uint64_t* outSize);
int mnnOpen(const mnnMemHandle* mem, unsigned int flags, void** outDevPtr);
int mnnPromote(uint64_t offset, uint64_t length, int targetGpu, void** outScratchPtr);
int mnnWriteback(uint64_t offset, uint64_t length, int sourceGpu, const void* scratchPtr);
int mnnWait(const mnnEvtHandle* evt, cudaStream_t stream);
int mnnSignal(const mnnEvtHandle* evt, cudaStream_t stream);
int mnnDescribe(const mnnMemHandle* mem, char* jsonBuf, size_t jsonBufLen);
int mnnClose(void* devPtr);
int mnnFree(const mnnMemHandle* mem);
```

### 15.2 Python Client (sketch)
```python
class MuninnBuffer:
    def __init__(self, size: int, mode: str = "bigMode", preferredGpu: int | None = None,
                 access: str = "readOnly", name: str | None = None, tenant: str | None = None): ...
    def open(self) -> int: ...
    def wait(self, stream) -> None: ...
    def signal(self, stream) -> None: ...
    def promote(self, offset: int, length: int, targetGpu: int) -> int: ...
    def writeback(self, offset: int, length: int, sourceGpu: int, scratchPtr: int) -> None: ...
    def describe(self) -> dict: ...
    def close(self) -> None: ...
    def free(self) -> None: ...
```

### 15.3 Message Schemas (informal)
```json
{
  "bufferId": "abc123",
  "mode": "bigMode",
  "gpu": 0,
  "size": 34359738368,
  "access": "readMostly",
  "tenant": "org/acct/job123",
  "leaseTtlMs": 120000,
  "peerable": true,
  "topology": { "p2p": ["0-1", "1-2"], "staged": ["0-2"] }
}
```

---

## 16. Error Codes & UX

| Code | Name | Meaning | Typical Fix |
|------|------|---------|-------------|
| MNN-0001 | InvalidConfig | Config parse/validation failed | Fix config; see logs for key/line |
| MNN-0002 | UnsupportedMode | Mode string not recognised | Use `bigMode` or `fastMode` |
| MNN-0003 | IpcHandleMismatch | Imported handle doesn’t match exporter | Ensure same driver/runtime; re-open |
| MNN-0004 | LeaseExpired | Lease TTL exceeded | Renew lease; reduce idle time |
| MNN-0005 | TopologyNotPeerable | GPUs lack P2P path | Accept staged route or relocate |
| MNN-0006 | PromotionDenied | Over budget / not allowed | Adjust policy or request smaller window |
| MNN-0007 | NvmeSpillUnavailable | NVMe disabled or path invalid | Enable NVMe; check permissions |
| MNN-0008 | PermissionDenied | Tenant not authorised | Fix tenant/ACLs |
| MNN-0009 | BarPressureHigh | BAR window exhausted | Reduce mapped windows; enable windowing |
| MNN-0010 | OwnerDied | Exporter exited with live importers | Recreate buffer; broker cleanup in progress |

**UX Principles**
- Clear reason codes and next-step hints.
- Human-friendly `muninctl` messages + structured logs for machines.

---

## 17. Examples & Workflows

### 17.1 Big, Not Fast (bigMode)
```bash
muninctl alloc --size 128G --mode bigMode --name datasetA --tenant org/team/job7
producer --buffer datasetA --tenant org/team/job7 &
consumer --buffer datasetA --tenant org/team/job7 &
# Promote a hot window for a heavy kernel on GPU0
muninctl promote --name datasetA --offset 4G --length 512M --gpu 0
```

### 17.2 Same-GPU Zero-Copy (fastMode)
```bash
muninctl alloc --size 4G --mode fastMode --gpu 0 --name weightsX
producer --buffer weightsX & consumer --buffer weightsX
```

### 17.3 Cross-GPU with Route Selection
```bash
muninctl alloc --size 2G --mode fastMode --gpu 0 --name tensorY
consumer --buffer tensorY --target-gpu 1
# Broker chooses P2P if possible; else staged; see logs/metrics for route chosen
```

### 17.4 Programmatic (Python)
```python
buf = MuninnBuffer(size=128<<30, mode="bigMode", access="readMostly", name="datasetA", tenant="org/x")
ptr = buf.open()
# Before a heavy kernel:
scratch = buf.promote(offset=4<<30, length=512<<20, targetGpu=0)
# launch kernel with scratch pointer...
buf.close()
buf.free()
```

---

## 18. Acceptance Criteria

- **Demo A (bigMode):** Map ≥128 GB; 3 processes perform streaming reads; throughput stable within PCIe-class expectations; open p50 < 2 ms.
- **Demo B (fastMode same-GPU):** Producer writes; consumer reads; no host copies; IPC events fence correctly.
- **Demo C (Cross-GPU):** Route selection logs reasoned choice; P2P outperforms staged when eligible.
- **Promotion:** Hinted ranges improve kernel time measurably; writeback policy respected.
- **Ops:** Metrics populated; dashboards live; alarms fire on synthetic pressure; logs actionable.
- **Safety:** Forced revocation works; orphan GC reclaims; no leaks in soak (≥12h).

---

## 19. Open Questions & Decisions Log

### Open Questions
- Scratch VRAM partitioning: static vs elastic per GPU?  
- Dirty-range granularity: 64 KiB vs 256 KiB vs 1 MiB?  
- How aggressive should admission control be under high BAR pressure?  
- Should auto-promotion be enabled by default or off until hinted?

### Decisions (log)
- **D-0001:** _Capacity over latency_ is the top priority for bigMode.  
- **D-0002:** Promotion returns a **scratch pointer**, not a remapped base pointer.  
- **D-0003:** Handles are **capabilities** with short TTLs; broker validates per process/tenant.  
- **D-0004:** Cross-node shared mapping is **out of scope**; RDMA is transfer-only.

---

## 20. Roadmap Timeline

| Milestone | Target (indicative) | Notes |
|----------|----------------------|-------|
| M0 | Month 1 | bigMode MVP + control blocks |
| M1 | Month 2 | fastMode (same-GPU) + IPC events |
| M2 | Month 3 | Cross-GPU P2P/staged with routing metrics |
| M3 | Month 4 | Promotion + writeback |
| M4 | Month 5 | Quotas/leases/observability; `muninctl` |
| M5 | Month 6 | NVMe spill; optional RDMA |
| M6 | Month 7 | Auto-promotion; tuning; admission control polish |

_Actual dates depend on hardware availability and upstream dependencies._

---

## 21. Future Work

- **Cross-node caching** with RDMA-aware prefetch daemons.
- **GPU-side memcpy kernels** tuned for specific architectures.
- **Integration with allocator hints** from compilers/runtimes.
- **Pluggable policy engines** (Lua/Python rules for placement).
- **Adaptive writeback strategies** (journaling; copy-on-write slices).

---

## 22. Glossary

- **rememory** — User’s Python package enabling shared memory across processes/CPUs. Muninn is its **GPU-grade cousin**.
- **bigMode (`mappedHostHuge`)** — Device-mapped pinned host memory forming a single large contiguous pointer; capacity-first.
- **fastMode (`vramShared`)** — Shared VRAM via CUDA IPC memory + events; same-GPU zero-copy.
- **Promotion** — Temporarily staging hot sections of bigMode into per-GPU scratch VRAM for faster kernels.
- **Writeback** — Copying modified scratch ranges back to bigMode backing.
- **P2P** — Peer-to-peer GPU copies across NVLink/PCIe.
- **Staged** — Copy path via pinned host memory when P2P unavailable.
- **BAR** — Base Address Register aperture used for device-mapped host memory.
- **Lease** — Time-limited grant to use a buffer; renewable.
- **Refcount** — Process-safe count of imports; broker authoritative.

---

## 23. Changelog for this Document

- **v0.1 (initial):** End-to-end roadmap drafted with milestones M0–M6, APIs, metrics, and acceptance criteria. Emphasised **GPU implementation of rememory on steroids**.
- **v0.1.1:** Minor clarifications on promotion semantics and quotas.
- **v0.1.2:** Added example CLI flows, security notes, and BAR pressure alarms.