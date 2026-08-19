# AMD HIP P2P Diagnostic Tool

Standalone diagnostic tool to verify HIP Peer-to-Peer transfers between AMD GPUs on Windows ROCm 7.14.

Purpose: Determine whether peer-copy corruption occurs at PCIe/platform, ROCm/HIP runtime/driver, or llama.cpp level.

## Requirements

- Windows 11
- AMD ROCm 7.14 (TheRock) runtime
- 2 or more AMD GPUs visible to HIP
- PowerShell

## Usage

```powershell
$env:HIP_VISIBLE_DEVICES = "1,2"
.\amd-hip-p2p-test.exe
```

## Tests

| Test | Description | Runs when canAccess=0? |
|------|-------------|------------------------|
| **A** Device Information | GPU names, arch, VRAM, PCI info, runtime/driver version | Yes |
| **B** hipDeviceCanAccessPeer | Bidirectional P2P accessibility | Yes |
| **C** hipDeviceEnablePeerAccess | Bidirectional P2P enable | Yes |
| **D** hipMemcpyPeer correctness | Sync P2P copy, byte-verified | Yes (forced) |
| **E** hipMemcpyPeerAsync correctness | Async P2P copy, both src+dst stream, byte-verified | Yes (forced) |
| **F** Host-staged comparison | GPU→Host→GPU without P2P, byte-verified | Yes |

### Key behavior

- **All tests always run**, even when `hipDeviceCanAccessPeer=0`
- P2P API calls are forced to reproduce the same code path as llama.cpp b10453
- Test E tries both dst-stream and src-stream (llama.cpp uses src-stream)
- Every byte is verified via `compareBuffers`
- Exit code is non-zero if any correctness test fails

### Correctness tests (D, E, F)

Each test runs at 4 MiB with:
1. Sequential pattern: `byte = (seed + offset) & 0xFF`
2. Pseudo-random pattern: xorshift, seed = 0xDEADBEEF

On failure, first 10 mismatches are reported with offset, expected, actual.

## Interpretation

### Case 1: hipDeviceCanAccessPeer = NO

Direct P2P memory access not available on this platform.

If peer copy APIs still fail and host-staged passes → matches llama.cpp `NO_PEER_COPY=ON` behavior.

### Case 2: Peer copy = data corruption, Host staged = PASS

Strongly suggests peer-transfer-path-specific issue.

### Case 3: All tests PASS

Investigate llama.cpp peer copy usage, scheduler, device management.

## Build

GitHub Actions workflow is the primary build method. Uses TheRock ROCm 7.14 pip package.

```powershell
cmake -S . -B build -G "Unix Makefiles" `
  -DCMAKE_PREFIX_PATH="$env:HIP_PATH" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$clang" `
  -DCMAKE_CXX_COMPILER="$clangxx" `
  -DCMAKE_HIP_COMPILER="$clang" `
  -DHIP_PATH="$env:HIP_PATH" `
  "-DAMDGPU_TARGETS=gfx1101;gfx1201"
```
