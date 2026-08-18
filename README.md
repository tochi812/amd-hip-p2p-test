# AMD HIP P2P Diagnostic Tool

Standalone diagnostic tool to verify HIP Peer-to-Peer transfers between AMD GPUs on Windows ROCm 7.14.

Purpose: Determine whether peer-copy corruption occurs at PCIe/platform, ROCm/HIP runtime/driver, or llama.cpp level.

## Requirements

- Windows 11
- AMD ROCm 7.14 (TheRock) runtime
- 2 or more AMD GPUs visible to HIP
- PowerShell

## Usage

Set which GPUs to use before running:

```powershell
$env:HIP_VISIBLE_DEVICES = "1,2"
.\amd-hip-p2p-test.exe
```

### Options

| Option | Description |
|--------|-------------|
| `--quick` | Reduced sizes and iterations |
| `--iterations N` | Override stress test iteration count |
| `--size-mib N` | Override stress/bandwidth test size (MiB) |
| `--no-bandwidth` | Skip bandwidth measurement |

## Tests

| Test | Description |
|------|-------------|
| **A** Device Information | GPU names, arch, VRAM, PCI info |
| **B** hipDeviceCanAccessPeer | Bidirectional P2P accessibility |
| **C** hipDeviceEnablePeerAccess | Bidirectional P2P enable |
| **D** hipMemcpyPeer correctness | Sync P2P copy, byte-verified |
| **E** hipMemcpyPeerAsync correctness | Async P2P copy, byte-verified |
| **F** Host-staged comparison | GPU->Host->GPU without P2P |
| **G** Stress Test | 100 iterations of async P2P copy |
| **H** Current Device dependency | Tests effect of current device on P2P |
| **I** Bandwidth | P2P vs host-staged throughput |
| **J** Direct peer kernel | SKIPPED (see below) |

## Test Details

### Correctness (D, E, F)

Each test runs at sizes: 4 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB.

Two patterns per size:
1. Sequential (byte = (seed + offset) & 0xFF)
2. Pseudo-random (xorshift, seed = 0xDEADBEEF)

On failure, first 10 mismatches are reported with offset, expected, actual.

### Stress (G)

64 MiB x 100 iterations per direction. Each iteration uses a different pseudo-random seed.

### Bandwidth (I)

64 MiB, 5 warmup + 20 timed iterations using hipEvent. Reports:
- hipMemcpyPeer (sync)
- hipMemcpyPeerAsync
- Host staged (GPU->Host->GPU)

## Interpretation

### Case 1: hipDeviceCanAccessPeer = NO

P2P not available. Check:
- PCIe topology
- BIOS settings (Above 4G, Resizable BAR)
- Driver version

### Case 2: EnablePeerAccess fails

HIP runtime, driver, or platform issue.

### Case 3: Peer copy = corruption, Host staged = PASS

Strongly suggests peer-transfer-path-specific issue.

### Case 4: Sync = PASS, Async = FAIL

Issue in async/stream/runtime path.

### Case 5: Current device affects result

Device/stream management issue.

### Case 6: All HIP tests PASS, llama.cpp fails

Investigate llama.cpp's peer copy usage, scheduler, device management.

## Test J: Direct Peer Kernel Access

SKIPPED. Requires HIP device-side peer pointer access (`__attribute__((address_space(1)))`).
On Windows ROCm 7.14, the correct implementation is not yet verified.
Implementing this requires:
- A minimal HIP kernel that reads/writes peer memory directly
- Verification that the kernel compiled for both gfx1101 and gfx1201
- No undefined behavior or spec violations

## Build

Requires TheRock ROCm 7.14 on Windows. Primary build validation is via GitHub Actions (Windows 2022 runner).

```powershell
cmake -S . -B build -G Ninja -DROCM_PATH=<path-to-rocm> -DAMDGPU_TARGETS="gfx1101;gfx1201"
cmake --build build --config Release
```
