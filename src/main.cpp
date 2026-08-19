#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_HIP(expr) checkHip(expr, #expr, __FILE__, __LINE__)
#define CHECK_HIP_FATAL(expr) checkHipFatal(expr, #expr, __FILE__, __LINE__)

static int g_errors = 0;

static void checkHip(hipError_t e, const char* expr, const char* file, int line) {
    if (e != hipSuccess) {
        fprintf(stderr, "  [WARN] %s failed: %s (%d) at %s:%d\n",
                expr, hipGetErrorString(e), e, file, line);
        g_errors++;
    }
}

static void checkHipFatal(hipError_t e, const char* expr, const char* file, int line) {
    if (e != hipSuccess) {
        fprintf(stderr, "  [FATAL] %s failed: %s (%d) at %s:%d\n",
                expr, hipGetErrorString(e), e, file, line);
        exit(1);
    }
}

static const char* errStr(hipError_t e) {
    return hipGetErrorString(e);
}

static void fillSequential(uint8_t* buf, size_t size, uint64_t seed) {
    for (size_t i = 0; i < size; i++) {
        buf[i] = (uint8_t)((seed + i) & 0xFF);
    }
}

static void fillPseudoRandom(uint8_t* buf, size_t size, uint64_t seed) {
    uint32_t s = (uint32_t)seed;
    for (size_t i = 0; i < size; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        buf[i] = (uint8_t)(s & 0xFF);
    }
}

struct MismatchInfo {
    size_t offset;
    uint8_t expected;
    uint8_t actual;
};

static int compareBuffers(const uint8_t* expected, const uint8_t* actual, size_t size,
                          MismatchInfo* mismatches, int maxMismatch, int* mismatchCount) {
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        if (expected[i] != actual[i]) {
            if (count < maxMismatch) {
                mismatches[count].offset = i;
                mismatches[count].expected = expected[i];
                mismatches[count].actual = actual[i];
            }
            count++;
        }
    }
    *mismatchCount = count;
    return count;
}

struct TestResult {
    const char* status; // "PASS", "FAIL", "SKIPPED", "N/A"
    int mismatchCount;
    hipError_t apiErr;
};

struct PeerTestResult {
    TestResult canAccess[2];     // [0]=0->1, [1]=1->0
    TestResult enablePeer[2];
    TestResult syncCopy[2];
    TestResult asyncCopyDstStream[2];
    TestResult asyncCopySrcStream[2];
    TestResult hostStaged[2];
    bool canAccessAny;
    bool enableOkAny;
};

struct DeviceInfo {
    int id;
    char name[256];
    char archName[256];
    size_t totalMem;
    int pciBus;
    int pciDevice;
    int pciDomain;
    int multiProcessorCount;
};

static bool queryDeviceInfo(int id, DeviceInfo* info) {
    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, id);
    if (e != hipSuccess) return false;
    info->id = id;
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    strncpy(info->archName, prop.gcnArchName, sizeof(info->archName) - 1);
    info->totalMem = prop.totalGlobalMem;
    info->pciBus = prop.pciBusID;
    info->pciDevice = prop.pciDeviceID;
    info->pciDomain = prop.pciDomainID;
    info->multiProcessorCount = prop.multiProcessorCount;
    return true;
}

static TestResult makeTestResult(const char* status, int mismatches, hipError_t e) {
    TestResult r;
    r.status = status;
    r.mismatchCount = mismatches;
    r.apiErr = e;
    return r;
}

static TestResult makeSkipped() { return makeTestResult("SKIPPED", 0, hipSuccess); }
static TestResult makePass() { return makeTestResult("PASS", 0, hipSuccess); }
static TestResult makeFail(int mismatches, hipError_t e) { return makeTestResult("FAIL", mismatches, e); }
static TestResult makeNaA() { return makeTestResult("N/A", 0, hipSuccess); }

// ============================================================
// Test A: Device Information + Runtime/Driver version
// ============================================================
static void testDeviceInformation(int count, DeviceInfo* infos) {
    int runtimeVer = 0, driverVer = 0;
    hipRuntimeGetVersion(&runtimeVer);
    hipDriverGetVersion(&driverVer);

    printf("\n=== DEVICE INFORMATION ===\n\n");
    printf("HIP Runtime version : %d\n", runtimeVer);
    printf("HIP Driver version  : %d\n", driverVer);
    printf("\n");

    for (int i = 0; i < count; i++) {
        printf("Device %d\n", i);
        printf("  Name             : %s\n", infos[i].name);
        printf("  Arch             : %s\n", infos[i].archName);
        printf("  VRAM             : %zu bytes (%.1f MiB)\n", infos[i].totalMem,
               (double)infos[i].totalMem / (1024.0 * 1024.0));
        printf("  PCI Bus          : %d\n", infos[i].pciBus);
        printf("  PCI Device       : %d\n", infos[i].pciDevice);
        printf("  PCI Domain       : %d\n", infos[i].pciDomain);
        printf("  MultiProcessors  : %d\n", infos[i].multiProcessorCount);
        printf("\n");
    }
}

// ============================================================
// Test B: hipDeviceCanAccessPeer (always runs)
// ============================================================
static void testCanAccessPeer(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST B: hipDeviceCanAccessPeer ===\n\n");

    int can01 = 0, can10 = 0;
    hipError_t e01 = hipDeviceCanAccessPeer(&can01, dev0, dev1);
    hipError_t e10 = hipDeviceCanAccessPeer(&can10, dev1, dev0);

    result->canAccess[0] = can01 ? makePass() : makeFail(0, e01);
    result->canAccess[1] = can10 ? makePass() : makeFail(0, e10);
    result->canAccessAny = (can01 != 0 || can10 != 0);

    printf("  %d -> %d : canAccess=%d, err=%s\n", dev0, dev1, can01, errStr(e01));
    printf("  %d -> %d : canAccess=%d, err=%s\n", dev1, dev0, can10, errStr(e10));
    printf("\n");
}

// ============================================================
// Test C: hipDeviceEnablePeerAccess (always runs, records result)
// ============================================================
static void testEnablePeerAccess(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST C: hipDeviceEnablePeerAccess ===\n\n");

    CHECK_HIP(hipSetDevice(dev0));
    hipError_t e01 = hipDeviceEnablePeerAccess(dev1, 0);
    bool ok01 = (e01 == hipSuccess || e01 == hipErrorPeerAccessAlreadyEnabled);
    result->enablePeer[0] = ok01 ? makePass() : makeFail(0, e01);

    CHECK_HIP(hipSetDevice(dev1));
    hipError_t e10 = hipDeviceEnablePeerAccess(dev0, 0);
    bool ok10 = (e10 == hipSuccess || e10 == hipErrorPeerAccessAlreadyEnabled);
    result->enablePeer[1] = ok10 ? makePass() : makeFail(0, e10);
    result->enableOkAny = ok01 || ok10;

    printf("  %d -> %d : %s (err=%s)\n", dev0, dev1, ok01 ? "OK" : "FAIL", errStr(e01));
    printf("  %d -> %d : %s (err=%s)\n", dev1, dev0, ok10 ? "OK" : "FAIL", errStr(e10));
    printf("\n");
}

// ============================================================
// Core correctness check helper
// ============================================================
static TestResult runCorrectnessCheck(int srcDev, int dstDev, size_t testSize,
                                       void (*fillFn)(uint8_t*, size_t, uint64_t),
                                       uint64_t seed, const char* label) {
    uint8_t* hSrc = nullptr;
    uint8_t* hDst = nullptr;
    uint8_t* dSrc = nullptr;
    uint8_t* dDst = nullptr;

    CHECK_HIP_FATAL(hipHostMalloc((void**)&hSrc, testSize));
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hDst, testSize));
    fillFn(hSrc, testSize, seed);
    memset(hDst, 0, testSize);

    CHECK_HIP_FATAL(hipSetDevice(srcDev));
    CHECK_HIP_FATAL(hipMalloc((void**)&dSrc, testSize));
    CHECK_HIP_FATAL(hipMemcpy(dSrc, hSrc, testSize, hipMemcpyHostToDevice));

    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMalloc((void**)&dDst, testSize));

    hipError_t e = hipMemcpyPeer(dDst, dstDev, dSrc, srcDev, testSize);

    if (e == hipSuccess) {
        CHECK_HIP_FATAL(hipSetDevice(dstDev));
        CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

        MismatchInfo mismatches[10];
        int mismatchCount = 0;
        int total = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

        printf("    %s %zu bytes: %s (mismatches=%d)\n", label, testSize,
               total == 0 ? "PASS" : "FAIL", total);
        if (total > 0) {
            for (int i = 0; i < mismatchCount && i < 10; i++) {
                printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                       mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
            }
        }
        TestResult r = (total == 0) ? makePass() : makeFail(total, hipSuccess);
        hipHostFree(hSrc); hipHostFree(hDst); hipFree(dSrc); hipFree(dDst);
        return r;
    } else {
        printf("    %s %zu bytes: API error=%s\n", label, testSize, errStr(e));
        TestResult r = makeFail(0, e);
        hipHostFree(hSrc); hipHostFree(hDst); hipFree(dSrc); hipFree(dDst);
        return r;
    }
}

// ============================================================
// Test D: hipMemcpyPeer correctness (always runs)
// ============================================================
static void testMemcpyPeer(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST D: hipMemcpyPeer correctness ===\n\n");

    size_t testSize = 4 * 1024 * 1024; // 4 MiB

    printf("  %d -> %d :\n", dev0, dev1);
    TestResult r01_seq = runCorrectnessCheck(dev0, dev1, testSize, fillSequential, 0, "sequential");
    TestResult r01_prn = runCorrectnessCheck(dev0, dev1, testSize, fillPseudoRandom, 0xDEADBEEF, "pseudo-random");
    result->syncCopy[0] = (r01_seq.status[0] == 'F' || r01_prn.status[0] == 'F') ? r01_prn : r01_seq;

    printf("\n  %d -> %d :\n", dev1, dev0);
    TestResult r10_seq = runCorrectnessCheck(dev1, dev0, testSize, fillSequential, 0, "sequential");
    TestResult r10_prn = runCorrectnessCheck(dev1, dev0, testSize, fillPseudoRandom, 0xDEADBEEF, "pseudo-random");
    result->syncCopy[1] = (r10_seq.status[0] == 'F' || r10_prn.status[0] == 'F') ? r10_prn : r10_seq;
    printf("\n");
}

// ============================================================
// Test E: hipMemcpyPeerAsync correctness (always runs, src+dst stream)
// ============================================================
static TestResult runAsyncCorrectness(int srcDev, int dstDev, size_t testSize,
                                       hipStream_t stream, const char* streamLabel) {
    uint8_t* hSrc = nullptr;
    uint8_t* hDst = nullptr;
    uint8_t* dSrc = nullptr;
    uint8_t* dDst = nullptr;

    CHECK_HIP_FATAL(hipHostMalloc((void**)&hSrc, testSize));
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hDst, testSize));
    fillPseudoRandom(hSrc, testSize, 0xDEADBEEF);
    memset(hDst, 0, testSize);

    CHECK_HIP_FATAL(hipSetDevice(srcDev));
    CHECK_HIP_FATAL(hipMalloc((void**)&dSrc, testSize));
    CHECK_HIP_FATAL(hipMemcpy(dSrc, hSrc, testSize, hipMemcpyHostToDevice));

    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMalloc((void**)&dDst, testSize));

    hipError_t e = hipMemcpyPeerAsync(dDst, dstDev, dSrc, srcDev, testSize, stream);
    if (e != hipSuccess) {
        printf("    async %s: API error=%s\n", streamLabel, errStr(e));
        TestResult r = makeFail(0, e);
        hipHostFree(hSrc); hipHostFree(hDst); hipFree(dSrc); hipFree(dDst);
        return r;
    }

    CHECK_HIP_FATAL(hipStreamSynchronize(stream));
    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

    MismatchInfo mismatches[10];
    int mismatchCount = 0;
    int total = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

    printf("    async %s: %s (mismatches=%d)\n", streamLabel,
           total == 0 ? "PASS" : "FAIL", total);
    if (total > 0) {
        for (int i = 0; i < mismatchCount && i < 10; i++) {
            printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                   mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
        }
    }
    TestResult r = (total == 0) ? makePass() : makeFail(total, hipSuccess);
    hipHostFree(hSrc); hipHostFree(hDst); hipFree(dSrc); hipFree(dDst);
    return r;
}

static void testMemcpyPeerAsync(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST E: hipMemcpyPeerAsync correctness ===\n\n");

    size_t testSize = 4 * 1024 * 1024;

    for (int dir = 0; dir < 2; dir++) {
        int src = (dir == 0) ? dev0 : dev1;
        int dst = (dir == 0) ? dev1 : dev0;

        printf("  %d -> %d :\n", src, dst);

        // Case A: stream on dst device (current = dst)
        CHECK_HIP_FATAL(hipSetDevice(dst));
        hipStream_t streamDst;
        CHECK_HIP_FATAL(hipStreamCreate(&streamDst));
        TestResult rDst = runAsyncCorrectness(src, dst, testSize, streamDst, "dst_stream");
        hipStreamDestroy(streamDst);

        // Case B: stream on src device (current = src) — matches llama.cpp b10453
        CHECK_HIP_FATAL(hipSetDevice(src));
        hipStream_t streamSrc;
        CHECK_HIP_FATAL(hipStreamCreate(&streamSrc));
        TestResult rSrc = runAsyncCorrectness(src, dst, testSize, streamSrc, "src_stream");
        hipStreamDestroy(streamSrc);

        result->asyncCopyDstStream[dir] = rDst;
        result->asyncCopySrcStream[dir] = rSrc;
        printf("\n");
    }
}

// ============================================================
// Test F: Host-staged comparison (always runs, no P2P)
// ============================================================
static void testHostStaged(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST F: Host-staged comparison (no P2P) ===\n\n");

    size_t testSize = 4 * 1024 * 1024;

    for (int dir = 0; dir < 2; dir++) {
        int src = (dir == 0) ? dev0 : dev1;
        int dst = (dir == 0) ? dev1 : dev0;

        printf("  %d -> %d :\n", src, dst);

        uint8_t* hSrc = nullptr;
        uint8_t* hDst = nullptr;
        uint8_t* hStage = nullptr;
        uint8_t* dSrc = nullptr;
        uint8_t* dDst = nullptr;

        CHECK_HIP_FATAL(hipHostMalloc((void**)&hSrc, testSize));
        CHECK_HIP_FATAL(hipHostMalloc((void**)&hDst, testSize));
        CHECK_HIP_FATAL(hipHostMalloc((void**)&hStage, testSize));
        fillSequential(hSrc, testSize, 0);

        CHECK_HIP_FATAL(hipSetDevice(src));
        CHECK_HIP_FATAL(hipMalloc((void**)&dSrc, testSize));
        CHECK_HIP_FATAL(hipMemcpy(dSrc, hSrc, testSize, hipMemcpyHostToDevice));

        CHECK_HIP_FATAL(hipSetDevice(dst));
        CHECK_HIP_FATAL(hipMalloc((void**)&dDst, testSize));

        // GPU src -> Host
        CHECK_HIP_FATAL(hipSetDevice(src));
        CHECK_HIP_FATAL(hipMemcpy(hStage, dSrc, testSize, hipMemcpyDeviceToHost));

        // Host -> GPU dst
        CHECK_HIP_FATAL(hipSetDevice(dst));
        CHECK_HIP_FATAL(hipMemcpy(dDst, hStage, testSize, hipMemcpyHostToDevice));

        // Read back
        CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

        MismatchInfo mismatches[10];
        int mismatchCount = 0;
        int total = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

        printf("    %s (mismatches=%d)\n", total == 0 ? "PASS" : "FAIL", total);
        if (total > 0) {
            for (int i = 0; i < mismatchCount && i < 10; i++) {
                printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                       mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
            }
        }
        result->hostStaged[dir] = (total == 0) ? makePass() : makeFail(total, hipSuccess);

        hipHostFree(hSrc); hipHostFree(hDst); hipHostFree(hStage);
        hipFree(dSrc); hipFree(dDst);
    }
    printf("\n");
}

// ============================================================
// Final Summary
// ============================================================
static void printFinalSummary(DeviceInfo* infos, PeerTestResult* result, int dev0, int dev1) {
    printf("========================================\n");
    printf("FINAL SUMMARY\n");
    printf("========================================\n\n");

    printf("Device %d: %s / %s\n", dev0, infos[0].name, infos[0].archName);
    printf("Device %d: %s / %s\n\n", dev1, infos[1].name, infos[1].archName);

    printf("CanAccessPeer\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->canAccess[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->canAccess[1].status);

    printf("EnablePeerAccess\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->enablePeer[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->enablePeer[1].status);

    printf("hipMemcpyPeer (sync)\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->syncCopy[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->syncCopy[1].status);

    printf("hipMemcpyPeerAsync (dst stream)\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->asyncCopyDstStream[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->asyncCopyDstStream[1].status);

    printf("hipMemcpyPeerAsync (src stream)\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->asyncCopySrcStream[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->asyncCopySrcStream[1].status);

    printf("Host staged\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->hostStaged[0].status);
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->hostStaged[1].status);

    printf("----------------------------------------\n");
    printf("OBSERVED:\n");
    if (!result->canAccessAny) {
        printf("  hipDeviceCanAccessPeer returned NO for both directions.\n");
        printf("  Direct P2P memory access is not available on this platform.\n\n");
        printf("  Peer copy APIs (hipMemcpyPeer/Async) were still executed.\n");
        bool anyApiError = false;
        for (int i = 0; i < 2; i++) {
            if (result->syncCopy[i].status[0] == 'F' ||
                result->asyncCopyDstStream[i].status[0] == 'F' ||
                result->asyncCopySrcStream[i].status[0] == 'F') {
                anyApiError = true;
            }
        }
        bool allHostStagedPass = (result->hostStaged[0].status[0] == 'P' &&
                                  result->hostStaged[1].status[0] == 'P');
        if (anyApiError) {
            printf("  Peer copy APIs failed as expected (canAccess=0).\n");
        }
        if (allHostStagedPass) {
            printf("  Host-staged transfer passed all checks.\n");
            printf("  llama.cpp NO_PEER_COPY=ON behavior is consistent.\n");
        }
    } else {
        bool anyPeerCorruption = false;
        for (int i = 0; i < 2; i++) {
            if (result->syncCopy[i].status[0] == 'F' ||
                result->asyncCopyDstStream[i].status[0] == 'F' ||
                result->asyncCopySrcStream[i].status[0] == 'F') {
                anyPeerCorruption = true;
            }
        }
        bool allHostStagedPass = (result->hostStaged[0].status[0] == 'P' &&
                                  result->hostStaged[1].status[0] == 'P');
        if (anyPeerCorruption) {
            printf("  Peer copy API returned data corruption.\n");
            if (allHostStagedPass) {
                printf("  Host-staged transfer passed all checks.\n");
                printf("  This strongly suggests a peer-transfer-path-specific issue.\n");
                printf("  This matches llama.cpp NO_PEER_COPY=ON fixing the corruption.\n");
            }
        } else if (allHostStagedPass) {
            printf("  All peer copy and host-staged tests passed.\n");
            printf("  If llama.cpp still shows corruption,\n");
            printf("  investigate llama.cpp peer copy usage, scheduler, device management.\n");
        }
    }
    printf("\n");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    bool quickMode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--quick]\n", argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--quick") == 0) quickMode = true;
    }

    printf("AMD HIP P2P Diagnostic Tool\n");
    printf("Built with: ROCm HIP (TheRock)\n\n");

    int deviceCount = 0;
    CHECK_HIP_FATAL(hipGetDeviceCount(&deviceCount));
    printf("HIP device count: %d\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("ERROR: Need at least 2 HIP devices. Found %d.\n", deviceCount);
        printf("Set HIP_VISIBLE_DEVICES (e.g., $env:HIP_VISIBLE_DEVICES = \"1,2\").\n");
        return 1;
    }

    DeviceInfo infos[8];
    for (int i = 0; i < deviceCount && i < 8; i++) {
        if (!queryDeviceInfo(i, &infos[i])) {
            fprintf(stderr, "Failed to query device %d\n", i);
            return 1;
        }
    }

    int dev0 = 0;
    int dev1 = 1;
    PeerTestResult result = {};

    // A: Device Information (always)
    testDeviceInformation(deviceCount < 2 ? deviceCount : 2, infos);

    // B: CanAccessPeer (always)
    testCanAccessPeer(dev0, dev1, &result);

    // C: EnablePeerAccess (always, just record result)
    testEnablePeerAccess(dev0, dev1, &result);

    // D: hipMemcpyPeer sync (always — force even if canAccess=0)
    testMemcpyPeer(dev0, dev1, &result);

    // E: hipMemcpyPeerAsync (always — force, test both src and dst stream)
    testMemcpyPeerAsync(dev0, dev1, &result);

    // F: Host-staged (always — no P2P dependency)
    testHostStaged(dev0, dev1, &result);

    // Summary
    printFinalSummary(infos, &result, dev0, dev1);

    // Exit code: non-zero if any correctness test failed
    int testFailures = 0;
    for (int i = 0; i < 2; i++) {
        if (result.syncCopy[i].status[0] == 'F') testFailures++;
        if (result.asyncCopyDstStream[i].status[0] == 'F') testFailures++;
        if (result.asyncCopySrcStream[i].status[0] == 'F') testFailures++;
        if (result.hostStaged[i].status[0] == 'F') testFailures++;
    }

    return (g_errors > 0 || testFailures > 0) ? 1 : 0;
}
