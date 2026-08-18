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

struct TestSizes {
    size_t sizes[5];
    int count;
};

static TestSizes getDefaultSizes() {
    return {{4096, 1048576, 16777216, 67108864, 268435456}, 5};
}

struct PeerTestResult {
    bool canAccess;
    bool enableOk;
    bool syncOk;
    bool asyncOk;
    bool hostStagedOk;
    hipError_t canAccessErr;
    hipError_t enableErr;
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

// ============================================================
// Test A: Device Information
// ============================================================
static int testDeviceInformation(int count, DeviceInfo* infos) {
    printf("\n=== DEVICE INFORMATION ===\n\n");
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
    return 0;
}

// ============================================================
// Test B: hipDeviceCanAccessPeer
// ============================================================
static void testCanAccessPeer(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST B: hipDeviceCanAccessPeer ===\n\n");

    int can01 = 0, can10 = 0;
    hipError_t e01, e10;

    e01 = hipDeviceCanAccessPeer(&can01, dev0, dev1);
    printf("  %d -> %d : canAccess=%d, err=%s\n", dev0, dev1, can01, errStr(e01));

    e10 = hipDeviceCanAccessPeer(&can10, dev1, dev0);
    printf("  %d -> %d : canAccess=%d, err=%s\n", dev1, dev0, can10, errStr(e10));

    result->canAccess = (can01 != 0 && can10 != 0);
    result->canAccessErr = (e01 != hipSuccess) ? e01 : e10;
    printf("\n");
}

// ============================================================
// Test C: hipDeviceEnablePeerAccess
// ============================================================
static void testEnablePeerAccess(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST C: hipDeviceEnablePeerAccess ===\n\n");

    hipError_t e01, e10;

    CHECK_HIP(hipSetDevice(dev0));
    e01 = hipDeviceEnablePeerAccess(dev1, 0);
    bool ok01 = (e01 == hipSuccess || e01 == hipErrorPeerAccessAlreadyEnabled);
    printf("  %d -> %d : %s (err=%s)\n", dev0, dev1,
           ok01 ? "OK" : "FAIL", errStr(e01));

    CHECK_HIP(hipSetDevice(dev1));
    e10 = hipDeviceEnablePeerAccess(dev0, 0);
    bool ok10 = (e10 == hipSuccess || e10 == hipErrorPeerAccessAlreadyEnabled);
    printf("  %d -> %d : %s (err=%s)\n", dev1, dev0,
           ok10 ? "OK" : "FAIL", errStr(e10));

    result->enableOk = ok01 && ok10;
    result->enableErr = (e01 != hipSuccess && e01 != hipErrorPeerAccessAlreadyEnabled) ? e01 : e10;
    printf("\n");
}

// ============================================================
// Test D: hipMemcpyPeer correctness
// ============================================================
static bool testMemcpyPeerCorrectness(int srcDev, int dstDev, size_t testSize,
                                       const char* patternName,
                                       void (*fillFn)(uint8_t*, size_t, uint64_t),
                                       uint64_t seed) {
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
    if (e != hipSuccess) {
        printf("    hipMemcpyPeer failed: %s\n", errStr(e));
        hipHostFree(hSrc);
        hipHostFree(hDst);
        hipFree(dSrc);
        hipFree(dDst);
        return false;
    }

    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

    MismatchInfo mismatches[10];
    int mismatchCount = 0;
    int totalMismatches = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

    bool pass = (totalMismatches == 0);
    printf("    %s %zu bytes: %s (mismatches=%d)\n", patternName, testSize,
           pass ? "PASS" : "FAIL", totalMismatches);
    if (!pass) {
        for (int i = 0; i < mismatchCount && i < 10; i++) {
            printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                   mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
        }
    }

    hipHostFree(hSrc);
    hipHostFree(hDst);
    hipFree(dSrc);
    hipFree(dDst);
    return pass;
}

static void testMemcpyPeer(int dev0, int dev1, PeerTestResult* result, bool quick) {
    printf("=== TEST D: hipMemcpyPeer correctness ===\n\n");

    TestSizes ts = getDefaultSizes();
    if (quick) ts.count = 1;

    bool allPass01 = true, allPass10 = true;

    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev0, dev1);
        bool p1 = testMemcpyPeerCorrectness(dev0, dev1, sz, "sequential",
                                             fillSequential, 0);
        bool p2 = testMemcpyPeerCorrectness(dev0, dev1, sz, "pseudo-random",
                                             fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass01 = false;
    }
    printf("\n");
    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev1, dev0);
        bool p1 = testMemcpyPeerCorrectness(dev1, dev0, sz, "sequential",
                                             fillSequential, 0);
        bool p2 = testMemcpyPeerCorrectness(dev1, dev0, sz, "pseudo-random",
                                             fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass10 = false;
    }

    result->syncOk = allPass01 && allPass10;
    printf("\n");
}

// ============================================================
// Test E: hipMemcpyPeerAsync correctness
// ============================================================
static bool testMemcpyPeerAsyncCorrectness(int srcDev, int dstDev, size_t testSize,
                                            const char* patternName,
                                            void (*fillFn)(uint8_t*, size_t, uint64_t),
                                            uint64_t seed) {
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

    hipStream_t stream;
    CHECK_HIP_FATAL(hipStreamCreate(&stream));

    // hipMemcpyPeerAsync must be called with current device = dstDevice
    // and the stream belongs to the current device (dstDev)
    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    hipError_t e = hipMemcpyPeerAsync(dDst, dstDev, dSrc, srcDev, testSize, stream);
    if (e != hipSuccess) {
        printf("    hipMemcpyPeerAsync failed: %s\n", errStr(e));
        hipStreamDestroy(stream);
        hipHostFree(hSrc);
        hipHostFree(hDst);
        hipFree(dSrc);
        hipFree(dDst);
        return false;
    }

    CHECK_HIP_FATAL(hipStreamSynchronize(stream));
    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

    MismatchInfo mismatches[10];
    int mismatchCount = 0;
    int totalMismatches = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

    bool pass = (totalMismatches == 0);
    printf("    %s %zu bytes: %s (mismatches=%d)\n", patternName, testSize,
           pass ? "PASS" : "FAIL", totalMismatches);
    if (!pass) {
        for (int i = 0; i < mismatchCount && i < 10; i++) {
            printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                   mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
        }
    }

    hipStreamDestroy(stream);
    hipHostFree(hSrc);
    hipHostFree(hDst);
    hipFree(dSrc);
    hipFree(dDst);
    return pass;
}

static void testMemcpyPeerAsync(int dev0, int dev1, PeerTestResult* result, bool quick) {
    printf("=== TEST E: hipMemcpyPeerAsync correctness ===\n\n");

    TestSizes ts = getDefaultSizes();
    if (quick) ts.count = 1;

    bool allPass01 = true, allPass10 = true;

    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev0, dev1);
        bool p1 = testMemcpyPeerAsyncCorrectness(dev0, dev1, sz, "sequential",
                                                  fillSequential, 0);
        bool p2 = testMemcpyPeerAsyncCorrectness(dev0, dev1, sz, "pseudo-random",
                                                  fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass01 = false;
    }
    printf("\n");
    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev1, dev0);
        bool p1 = testMemcpyPeerAsyncCorrectness(dev1, dev0, sz, "sequential",
                                                  fillSequential, 0);
        bool p2 = testMemcpyPeerAsyncCorrectness(dev1, dev0, sz, "pseudo-random",
                                                  fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass10 = false;
    }

    result->asyncOk = allPass01 && allPass10;
    printf("\n");
}

// ============================================================
// Test F: Host-staged comparison (no P2P)
// ============================================================
static bool testHostStagedCorrectness(int srcDev, int dstDev, size_t testSize,
                                       const char* patternName,
                                       void (*fillFn)(uint8_t*, size_t, uint64_t),
                                       uint64_t seed) {
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

    // Stage through host: GPU src -> Host -> GPU dst
    uint8_t* hStage = nullptr;
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hStage, testSize));

    // GPU src -> Host
    CHECK_HIP_FATAL(hipSetDevice(srcDev));
    CHECK_HIP_FATAL(hipMemcpy(hStage, dSrc, testSize, hipMemcpyDeviceToHost));

    // Host -> GPU dst
    CHECK_HIP_FATAL(hipSetDevice(dstDev));
    CHECK_HIP_FATAL(hipMemcpy(dDst, hStage, testSize, hipMemcpyHostToDevice));

    // Read back and verify
    CHECK_HIP_FATAL(hipMemcpy(hDst, dDst, testSize, hipMemcpyDeviceToHost));

    MismatchInfo mismatches[10];
    int mismatchCount = 0;
    int totalMismatches = compareBuffers(hSrc, hDst, testSize, mismatches, 10, &mismatchCount);

    bool pass = (totalMismatches == 0);
    printf("    %s %zu bytes: %s (mismatches=%d)\n", patternName, testSize,
           pass ? "PASS" : "FAIL", totalMismatches);
    if (!pass) {
        for (int i = 0; i < mismatchCount && i < 10; i++) {
            printf("      offset=%zu expected=0x%02X actual=0x%02X\n",
                   mismatches[i].offset, mismatches[i].expected, mismatches[i].actual);
        }
    }

    hipHostFree(hSrc);
    hipHostFree(hDst);
    hipHostFree(hStage);
    hipFree(dSrc);
    hipFree(dDst);
    return pass;
}

static void testHostStaged(int dev0, int dev1, PeerTestResult* result, bool quick) {
    printf("=== TEST F: Host-staged comparison (no P2P) ===\n\n");

    TestSizes ts = getDefaultSizes();
    if (quick) ts.count = 1;

    bool allPass01 = true, allPass10 = true;

    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev0, dev1);
        bool p1 = testHostStagedCorrectness(dev0, dev1, sz, "sequential",
                                             fillSequential, 0);
        bool p2 = testHostStagedCorrectness(dev0, dev1, sz, "pseudo-random",
                                             fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass01 = false;
    }
    printf("\n");
    for (int i = 0; i < ts.count; i++) {
        size_t sz = ts.sizes[i];
        printf("  %d -> %d :\n", dev1, dev0);
        bool p1 = testHostStagedCorrectness(dev1, dev0, sz, "sequential",
                                             fillSequential, 0);
        bool p2 = testHostStagedCorrectness(dev1, dev0, sz, "pseudo-random",
                                             fillPseudoRandom, 0xDEADBEEF);
        if (!p1 || !p2) allPass10 = false;
    }

    result->hostStagedOk = allPass01 && allPass10;
    printf("\n");
}

// ============================================================
// Test G: Stress Test
// ============================================================
static void testStress(int dev0, int dev1, bool quick, int iterations, size_t testSizeMiB) {
    printf("=== TEST G: Stress Test ===\n\n");

    size_t testSize = testSizeMiB * 1024 * 1024;
    if (quick) iterations = 10;

    uint8_t* dSrc = nullptr;
    uint8_t* dDst = nullptr;
    uint8_t* hVerify = nullptr;
    uint8_t* hPattern = nullptr;

    CHECK_HIP_FATAL(hipHostMalloc((void**)&hVerify, testSize));
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hPattern, testSize));

    for (int direction = 0; direction < 2; direction++) {
        int s = (direction == 0) ? dev0 : dev1;
        int d = (direction == 0) ? dev1 : dev0;

        CHECK_HIP_FATAL(hipSetDevice(s));
        CHECK_HIP_FATAL(hipMalloc((void**)&dSrc, testSize));

        CHECK_HIP_FATAL(hipSetDevice(d));
        hipError_t allocErr = hipMalloc((void**)&dDst, testSize);
        if (allocErr != hipSuccess) {
            printf("  %d -> %d : SKIPPED (cannot allocate %zu bytes on device %d: %s)\n",
                   s, d, testSize, d, errStr(allocErr));
            hipFree(dSrc);
            continue;
        }

        hipStream_t stream;
        CHECK_HIP_FATAL(hipStreamCreate(&stream));

        int passCount = 0, failCount = 0;
        for (int iter = 0; iter < iterations; iter++) {
            fillPseudoRandom(hPattern, testSize, (uint32_t)(iter * 0x1234567 + direction));

            CHECK_HIP_FATAL(hipSetDevice(s));
            CHECK_HIP_FATAL(hipMemcpy(dSrc, hPattern, testSize, hipMemcpyHostToDevice));

            CHECK_HIP_FATAL(hipSetDevice(d));
            hipError_t e = hipMemcpyPeerAsync(dDst, d, dSrc, s, testSize, stream);
            if (e != hipSuccess) {
                printf("    iter %d: hipMemcpyPeerAsync failed: %s\n", iter, errStr(e));
                failCount++;
                continue;
            }

            CHECK_HIP_FATAL(hipStreamSynchronize(stream));
            CHECK_HIP_FATAL(hipMemcpy(hVerify, dDst, testSize, hipMemcpyDeviceToHost));

            int mismatchCount = 0;
            MismatchInfo mismatches[1];
            compareBuffers(hPattern, hVerify, testSize, mismatches, 1, &mismatchCount);
            if (mismatchCount > 0) {
                failCount++;
                printf("    iter %d: FAIL (mismatches=%d, first at offset=%zu)\n",
                       iter, mismatchCount, mismatches[0].offset);
            } else {
                passCount++;
            }
        }

        printf("  %d -> %d : PASS %d / FAIL %d (iterations=%d, size=%zu MiB)\n",
               s, d, passCount, failCount, iterations, testSize / (1024 * 1024));

        hipStreamDestroy(stream);
        hipFree(dSrc);
        hipFree(dDst);
    }

    hipHostFree(hVerify);
    hipHostFree(hPattern);
    printf("\n");
}

// ============================================================
// Test H: Current Device dependency
// ============================================================
static void testCurrentDeviceDependency(int dev0, int dev1) {
    printf("=== TEST H: Current Device Dependency ===\n\n");

    size_t testSize = 4 * 1024 * 1024; // 4 MiB
    uint8_t* dSrc = nullptr;
    uint8_t* dDst = nullptr;
    uint8_t* hPattern = nullptr;
    uint8_t* hVerify = nullptr;

    CHECK_HIP_FATAL(hipHostMalloc((void**)&hPattern, testSize));
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hVerify, testSize));
    fillSequential(hPattern, testSize, 0xCAFEBABE);

    struct TestCase {
        int srcDev;
        int dstDev;
        int currentDevice;
        const char* desc;
    };

    TestCase cases[] = {
        {dev0, dev1, dev0, "src=current"},
        {dev0, dev1, dev1, "dst=current"},
        {dev1, dev0, dev1, "src=current (reverse)"},
        {dev1, dev0, dev0, "dst=current (reverse)"},
    };

    for (auto& tc : cases) {
        printf("  Case: %s\n", tc.desc);
        printf("    src=%d, dst=%d, currentDevice=%d\n", tc.srcDev, tc.dstDev, tc.currentDevice);

        CHECK_HIP_FATAL(hipSetDevice(tc.srcDev));
        hipError_t allocSrc = hipMalloc((void**)&dSrc, testSize);
        if (allocSrc != hipSuccess) {
            printf("    SKIPPED (cannot alloc on device %d: %s)\n", tc.srcDev, errStr(allocSrc));
            continue;
        }
        CHECK_HIP_FATAL(hipMemcpy(dSrc, hPattern, testSize, hipMemcpyHostToDevice));

        CHECK_HIP_FATAL(hipSetDevice(tc.dstDev));
        hipError_t allocDst = hipMalloc((void**)&dDst, testSize);
        if (allocDst != hipSuccess) {
            printf("    SKIPPED (cannot alloc on device %d: %s)\n", tc.dstDev, errStr(allocDst));
            hipFree(dSrc);
            continue;
        }

        // Set current device as specified in the test case
        CHECK_HIP_FATAL(hipSetDevice(tc.currentDevice));

        hipStream_t stream;
        CHECK_HIP_FATAL(hipStreamCreate(&stream));

        hipError_t e = hipMemcpyPeerAsync(dDst, tc.dstDev, dSrc, tc.srcDev, testSize, stream);
        hipStreamSynchronize(stream);

        if (e != hipSuccess) {
            printf("    hipMemcpyPeerAsync returned: %s\n", errStr(e));
        } else {
            CHECK_HIP_FATAL(hipSetDevice(tc.dstDev));
            CHECK_HIP_FATAL(hipMemcpy(hVerify, dDst, testSize, hipMemcpyDeviceToHost));
            MismatchInfo mismatches[1];
            int mismatchCount = 0;
            compareBuffers(hPattern, hVerify, testSize, mismatches, 1, &mismatchCount);
            printf("    result: %s (mismatches=%d)\n",
                   mismatchCount == 0 ? "PASS" : "FAIL", mismatchCount);
        }

        hipStreamDestroy(stream);
        hipFree(dSrc);
        hipFree(dDst);
    }

    hipHostFree(hPattern);
    hipHostFree(hVerify);
    printf("\n");
}

// ============================================================
// Test I: Bandwidth
// ============================================================
static void testBandwidth(int dev0, int dev1, bool quick, size_t testSizeMiB) {
    printf("=== TEST I: Bandwidth ===\n\n");

    size_t testSize = testSizeMiB * 1024 * 1024;
    int warmup = 5;
    int iterations = 20;
    if (quick) { warmup = 2; iterations = 5; }

    uint8_t* hSrc = nullptr;
    uint8_t* hDst = nullptr;
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hSrc, testSize));
    CHECK_HIP_FATAL(hipHostMalloc((void**)&hDst, testSize));
    fillSequential(hSrc, testSize, 0x1234);

    uint8_t* dSrc0 = nullptr, *dSrc1 = nullptr;
    uint8_t* dDst0 = nullptr, *dDst1 = nullptr;

    CHECK_HIP_FATAL(hipSetDevice(dev0));
    CHECK_HIP_FATAL(hipMalloc((void**)&dSrc0, testSize));
    CHECK_HIP_FATAL(hipMalloc((void**)&dDst0, testSize));
    CHECK_HIP_FATAL(hipMemcpy(dSrc0, hSrc, testSize, hipMemcpyHostToDevice));

    CHECK_HIP_FATAL(hipSetDevice(dev1));
    CHECK_HIP_FATAL(hipMalloc((void**)&dSrc1, testSize));
    CHECK_HIP_FATAL(hipMalloc((void**)&dDst1, testSize));
    CHECK_HIP_FATAL(hipMemcpy(dSrc1, hSrc, testSize, hipMemcpyHostToDevice));

    auto measureAvg = [&](auto&& fn) -> double {
        for (int i = 0; i < warmup; i++) fn();
        double totalMs = 0;
        for (int i = 0; i < iterations; i++) {
            hipEvent_t start, stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start);
            fn();
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            float ms = 0;
            hipEventElapsedTime(&ms, start, stop);
            totalMs += ms;
            hipEventDestroy(start);
            hipEventDestroy(stop);
        }
        return totalMs / iterations;
    };

    // hipMemcpyPeer: dev0 -> dev1
    double peerMs01 = measureAvg([&]() {
        hipMemcpyPeer(dDst1, dev1, dSrc0, dev0, testSize);
    });
    double bwPeer01 = (testSize / (1024.0 * 1024.0)) / (peerMs01 / 1000.0);
    printf("  hipMemcpyPeer      %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev0, dev1, peerMs01, bwPeer01);

    // hipMemcpyPeer: dev1 -> dev0
    double peerMs10 = measureAvg([&]() {
        hipMemcpyPeer(dDst0, dev0, dSrc1, dev1, testSize);
    });
    double bwPeer10 = (testSize / (1024.0 * 1024.0)) / (peerMs10 / 1000.0);
    printf("  hipMemcpyPeer      %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev1, dev0, peerMs10, bwPeer10);

    // hipMemcpyPeerAsync: dev0 -> dev1 (stream owned by dev1)
    CHECK_HIP_FATAL(hipSetDevice(dev1));
    hipStream_t stream01;
    hipStreamCreate(&stream01);
    double peerAsyncMs01 = measureAvg([&]() {
        hipMemcpyPeerAsync(dDst1, dev1, dSrc0, dev0, testSize, stream01);
        hipStreamSynchronize(stream01);
    });
    double bwAsync01 = (testSize / (1024.0 * 1024.0)) / (peerAsyncMs01 / 1000.0);
    printf("  hipMemcpyPeerAsync %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev0, dev1, peerAsyncMs01, bwAsync01);
    hipStreamDestroy(stream01);

    // hipMemcpyPeerAsync: dev1 -> dev0 (stream owned by dev0)
    CHECK_HIP_FATAL(hipSetDevice(dev0));
    hipStream_t stream10;
    hipStreamCreate(&stream10);
    double peerAsyncMs10 = measureAvg([&]() {
        hipMemcpyPeerAsync(dDst0, dev0, dSrc1, dev1, testSize, stream10);
        hipStreamSynchronize(stream10);
    });
    double bwAsync10 = (testSize / (1024.0 * 1024.0)) / (peerAsyncMs10 / 1000.0);
    printf("  hipMemcpyPeerAsync %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev1, dev0, peerAsyncMs10, bwAsync10);
    hipStreamDestroy(stream10);

    // Host staged: dev0 -> Host -> dev1 (peer access enabled, no device switch needed)
    double hostMs01 = measureAvg([&]() {
        hipMemcpy(hDst, dSrc0, testSize, hipMemcpyDeviceToHost);
        hipMemcpy(dDst1, hDst, testSize, hipMemcpyHostToDevice);
    });
    double bwHost01 = (testSize / (1024.0 * 1024.0)) / (hostMs01 / 1000.0);
    printf("  Host staged        %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev0, dev1, hostMs01, bwHost01);

    // Host staged: dev1 -> Host -> dev0 (peer access enabled, no device switch needed)
    double hostMs10 = measureAvg([&]() {
        hipMemcpy(hDst, dSrc1, testSize, hipMemcpyDeviceToHost);
        hipMemcpy(dDst0, hDst, testSize, hipMemcpyHostToDevice);
    });
    double bwHost10 = (testSize / (1024.0 * 1024.0)) / (hostMs10 / 1000.0);
    printf("  Host staged        %d -> %d : %.2f ms  (%.1f MiB/s)\n",
           dev1, dev0, hostMs10, bwHost10);

    printf("\n");

    hipHostFree(hSrc);
    hipHostFree(hDst);
    hipFree(dSrc0);
    hipFree(dSrc1);
    hipFree(dDst0);
    hipFree(dDst1);
}

// ============================================================
// Test J: Direct peer memory kernel access (skipped - P2)
// ============================================================
static void testDirectPeerKernel(int dev0, int dev1, PeerTestResult* result) {
    printf("=== TEST J: Direct Peer Memory Kernel Access ===\n\n");
    printf("  SKIPPED: Requires HIP device-side peer pointer access.\n");
    printf("  On Windows ROCm 7.14, the correct way to implement this\n");
    printf("  is not yet verified. See README for details.\n\n");
}

// ============================================================
// Final Summary
// ============================================================
static void printFinalSummary(int count, DeviceInfo* infos, PeerTestResult* result,
                               int dev0, int dev1) {
    printf("========================================\n");
    printf("FINAL SUMMARY\n");
    printf("========================================\n\n");

    for (int i = 0; i < count; i++) {
        printf("Device %d:\n", i);
        printf("  %s / %s\n\n", infos[i].name, infos[i].archName);
    }

    printf("CanAccessPeer\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->canAccess ? "YES" : "NO");
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->canAccess ? "YES" : "NO");

    printf("EnablePeerAccess\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->enableOk ? "PASS" : "FAIL");
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->enableOk ? "PASS" : "FAIL");

    printf("hipMemcpyPeer (sync)\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->syncOk ? "PASS" : "FAIL");
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->syncOk ? "PASS" : "FAIL");

    printf("hipMemcpyPeerAsync\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->asyncOk ? "PASS" : "FAIL");
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->asyncOk ? "PASS" : "FAIL");

    printf("Host staged\n");
    printf("  %d -> %d : %s\n", dev0, dev1, result->hostStagedOk ? "PASS" : "FAIL");
    printf("  %d -> %d : %s\n\n", dev1, dev0, result->hostStagedOk ? "PASS" : "FAIL");

    printf("----------------------------------------\n");
    printf("OBSERVED:\n");
    if (!result->canAccess) {
        printf("  hipDeviceCanAccessPeer returned NO.\n");
        printf("  P2P may not be available at PCIe/platform/driver level.\n");
    } else if (!result->enableOk) {
        printf("  hipDeviceEnablePeerAccess failed.\n");
        printf("  HIP runtime, driver, or platform issue suspected.\n");
    } else {
        if (!result->syncOk || !result->asyncOk) {
            printf("  Peer copy data corruption detected.\n");
            if (result->hostStagedOk) {
                printf("  Host-staged transfer passed all checks.\n");
                printf("  This strongly suggests a peer-transfer-path-specific issue.\n");
            }
        } else if (!result->hostStagedOk) {
            printf("  Host-staged transfer failed but peer copy passed.\n");
            printf("  Unexpected: host staging should not use P2P.\n");
        } else {
            printf("  All peer copy and host-staged tests passed.\n");
            printf("  If llama.cpp still shows corruption,\n");
            printf("  investigate llama.cpp peer copy usage, scheduler, device management.\n");
        }
    }
    printf("\n");
}

// ============================================================
// CLI
// ============================================================
static void printUsage(const char* prog) {
    printf("Usage: %s [--quick] [--iterations N] [--size-mib N] [--no-bandwidth]\n", prog);
    printf("  --quick          : Run with reduced sizes and iterations\n");
    printf("  --iterations N   : Override stress test iteration count\n");
    printf("  --size-mib N     : Override stress/bandwidth test size in MiB\n");
    printf("  --no-bandwidth   : Skip bandwidth measurement\n");
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    bool quick = false;
    bool noBandwidth = false;
    int stressIterations = 100;
    size_t stressSizeMiB = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (strcmp(argv[i], "--no-bandwidth") == 0) {
            noBandwidth = true;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            stressIterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size-mib") == 0 && i + 1 < argc) {
            stressSizeMiB = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("AMD HIP P2P Diagnostic Tool\n");
    printf("Built with: ROCm HIP (TheRock)\n\n");

    int deviceCount = 0;
    CHECK_HIP_FATAL(hipGetDeviceCount(&deviceCount));
    printf("HIP device count: %d\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("ERROR: Need at least 2 HIP devices. Found %d.\n", deviceCount);
        printf("Make sure HIP_VISIBLE_DEVICES is set (e.g., $env:HIP_VISIBLE_DEVICES = \"1,2\").\n");
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

    // A: Device Information
    testDeviceInformation(deviceCount < 2 ? deviceCount : 2, infos);

    // B: CanAccessPeer
    testCanAccessPeer(dev0, dev1, &result);

    if (!result.canAccess) {
        printf("P2P access not available. Skipping remaining P2P tests.\n");
        testDirectPeerKernel(dev0, dev1, &result);
        printFinalSummary(2, infos, &result, dev0, dev1);
        return 0;
    }

    // C: EnablePeerAccess
    testEnablePeerAccess(dev0, dev1, &result);

    if (!result.enableOk) {
        printf("Peer access enable failed. Skipping copy tests.\n");
        testDirectPeerKernel(dev0, dev1, &result);
        printFinalSummary(2, infos, &result, dev0, dev1);
        return 0;
    }

    // D: hipMemcpyPeer correctness
    testMemcpyPeer(dev0, dev1, &result, quick);

    // E: hipMemcpyPeerAsync correctness
    testMemcpyPeerAsync(dev0, dev1, &result, quick);

    // F: Host-staged comparison
    testHostStaged(dev0, dev1, &result, quick);

    // G: Stress Test
    testStress(dev0, dev1, quick, stressIterations, stressSizeMiB);

    // H: Current Device dependency
    testCurrentDeviceDependency(dev0, dev1);

    // I: Bandwidth
    if (!noBandwidth) {
        testBandwidth(dev0, dev1, quick, stressSizeMiB);
    }

    // J: Direct peer kernel (skipped)
    testDirectPeerKernel(dev0, dev1, &result);

    // Summary
    printFinalSummary(2, infos, &result, dev0, dev1);

    return (g_errors > 0) ? 1 : 0;
}
