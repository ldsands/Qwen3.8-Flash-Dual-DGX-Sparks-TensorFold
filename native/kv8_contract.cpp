// CPU execution of CUDA's actual E4M3 encoder/decoder, without a driver, context or model.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "flashnext_kv8.cuh"

static float reference(unsigned code) {
    unsigned e = (code >> 3) & 15, m = code & 7;
    float x = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + m / 8.0f, int(e) - 7);
    return code & 128 ? -x : x;
}

int main() {
    static_assert(fn_kv8::row_bytes == 260);
    assert(fn_kv8::scale(0.0f) == 1.0f);
    assert(fn_kv8::scale(448.0f) == 1.0f);
    assert(fn_kv8::scale(1e-40f) == 0x1p-126f);
    assert(std::isfinite(fn_kv8::scale(0x1.fffffep127f)));
    for (float scale : {1.0f, 0.125f, 3.25f, 100.0f}) {
        for (unsigned code = 0; code < 256; ++code) {
            if ((code & 127) == 127) {
                assert(std::isnan(fn_kv8::decode(code, scale)));
                continue;
            }
            float want = reference(code) * scale;
            assert(fn_kv8::decode(code, scale) == want);
            assert(fn_kv8::encode(want, scale) == code);
        }
        for (unsigned code = 0; code < 126; ++code) {
            float a = reference(code), b = reference(code + 1);
            float halfway = (a + b) * 0.5f;
            unsigned even = code & 1 ? code + 1 : code;
            assert(fn_kv8::encode(halfway, 1.0f) == even);
            assert(fn_kv8::encode(std::nextafter(halfway, a), 1.0f) == code);
            assert(fn_kv8::encode(std::nextafter(halfway, b), 1.0f) == code + 1);
        }
        assert(fn_kv8::decode(fn_kv8::encode(1e10f, scale), scale) == 448.0f * scale);
        assert(fn_kv8::decode(fn_kv8::encode(-1e10f, scale), scale) == -448.0f * scale);
    }
    // Distinct neighboring rows must retain independent scales and payloads.
    struct Row { uint8_t values[256]; float scale; } rows[2]{};
    static_assert(sizeof(Row) == fn_kv8::row_bytes);
    for (int r = 0; r < 2; ++r) {
        rows[r].scale = (r + 1) * 0.125f;
        for (int d = 0; d < 256; ++d) rows[r].values[d] = fn_kv8::encode(float(d - 128), rows[r].scale);
    }
    for (int r = 0; r < 2; ++r) for (int d = 0; d < 256; ++d) {
        float x = float(d - 128), clipped = std::fmax(-448 * rows[r].scale, std::fmin(448 * rows[r].scale, x));
        assert(std::fabs(fn_kv8::decode(rows[r].values[d], rows[r].scale) - clipped) <= std::fabs(clipped) / 16 + 0.01f);
    }
    std::puts("E4M3 CPU contracts passed: codes, ties, saturation, row layout and independent scales");
}
