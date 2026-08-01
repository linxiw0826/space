#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline int face_label(const int32_t* labels, int a, int b, int c) {
    const int la = labels[a], lb = labels[b], lc = labels[c];
    if (la == lb || la == lc) return la;
    if (lb == lc) return lb;
    // Boundary triangles without a majority remain background but still write
    // depth, so unannotated geometry correctly occludes annotated objects.
    return -1;
}

}  // namespace

extern "C" int rasterize_scannetppv2(
    const float* vertices,
    int64_t vertex_count,
    const int32_t* faces,
    int64_t face_count,
    const int32_t* vertex_labels,
    const double* camera_from_world,
    const double* intrinsic,
    int width,
    int height,
    int32_t* output_labels) {
    if (!vertices || !faces || !vertex_labels || !camera_from_world ||
        !intrinsic || !output_labels || vertex_count <= 0 || face_count <= 0 ||
        width <= 0 || height <= 0) {
        return 1;
    }

    std::vector<float> sx(vertex_count), sy(vertex_count), invz(vertex_count);
    std::vector<uint8_t> valid(vertex_count, 0);

    #pragma omp parallel for schedule(static)
    for (int64_t index = 0; index < vertex_count; ++index) {
        const double x = vertices[index * 3 + 0];
        const double y = vertices[index * 3 + 1];
        const double z = vertices[index * 3 + 2];
        const double cx = camera_from_world[0] * x + camera_from_world[1] * y +
                          camera_from_world[2] * z + camera_from_world[3];
        const double cy = camera_from_world[4] * x + camera_from_world[5] * y +
                          camera_from_world[6] * z + camera_from_world[7];
        const double cz = camera_from_world[8] * x + camera_from_world[9] * y +
                          camera_from_world[10] * z + camera_from_world[11];
        if (!(cz > 1e-4) || !std::isfinite(cz)) continue;
        const double px = intrinsic[0] * cx + intrinsic[1] * cy + intrinsic[2] * cz;
        const double py = intrinsic[3] * cx + intrinsic[4] * cy + intrinsic[5] * cz;
        sx[index] = static_cast<float>(px / cz);
        sy[index] = static_cast<float>(py / cz);
        invz[index] = static_cast<float>(1.0 / cz);
        valid[index] = std::isfinite(sx[index]) && std::isfinite(sy[index]);
    }

    // High 32 bits store positive float(1/z), whose bit order is monotonic.
    // Low 32 bits store instance+1 (zero means background).  A single atomic
    // compare/exchange keeps depth and label transactionally consistent.
    std::vector<std::atomic<uint64_t>> buffer(width * height);
    for (auto& value : buffer) value.store(0, std::memory_order_relaxed);

    #pragma omp parallel for schedule(dynamic, 512)
    for (int64_t face_index = 0; face_index < face_count; ++face_index) {
        const int a = faces[face_index * 3 + 0];
        const int b = faces[face_index * 3 + 1];
        const int c = faces[face_index * 3 + 2];
        if (a < 0 || b < 0 || c < 0 || a >= vertex_count || b >= vertex_count ||
            c >= vertex_count || !valid[a] || !valid[b] || !valid[c]) continue;

        const float ax = sx[a], ay = sy[a];
        const float bx = sx[b], by = sy[b];
        const float cx = sx[c], cy = sy[c];
        const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::abs(area) < 1e-8f) continue;
        int x0 = std::max(0, static_cast<int>(std::floor(std::min({ax, bx, cx}))));
        int x1 = std::min(width - 1, static_cast<int>(std::ceil(std::max({ax, bx, cx}))));
        int y0 = std::max(0, static_cast<int>(std::floor(std::min({ay, by, cy}))));
        int y1 = std::min(height - 1, static_cast<int>(std::ceil(std::max({ay, by, cy}))));
        if (x0 > x1 || y0 > y1) continue;

        const int label = face_label(vertex_labels, a, b, c);
        const uint32_t label_code = label >= 0 ? static_cast<uint32_t>(label + 1) : 0;
        const float inverse_area = 1.0f / area;
        for (int py = y0; py <= y1; ++py) {
            const float sample_y = py + 0.5f;
            for (int px = x0; px <= x1; ++px) {
                const float sample_x = px + 0.5f;
                const float wa = ((bx - sample_x) * (cy - sample_y) -
                                  (by - sample_y) * (cx - sample_x)) * inverse_area;
                const float wb = ((cx - sample_x) * (ay - sample_y) -
                                  (cy - sample_y) * (ax - sample_x)) * inverse_area;
                const float wc = 1.0f - wa - wb;
                if (wa < -1e-6f || wb < -1e-6f || wc < -1e-6f) continue;
                const float depth_value = wa * invz[a] + wb * invz[b] + wc * invz[c];
                if (!(depth_value > 0.0f) || !std::isfinite(depth_value)) continue;
                const uint64_t candidate =
                    (static_cast<uint64_t>(float_bits(depth_value)) << 32) | label_code;
                auto& pixel = buffer[py * width + px];
                uint64_t current = pixel.load(std::memory_order_relaxed);
                while (candidate > current && !pixel.compare_exchange_weak(
                    current, candidate, std::memory_order_relaxed,
                    std::memory_order_relaxed)) {}
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int index = 0; index < width * height; ++index) {
        const uint32_t code = static_cast<uint32_t>(
            buffer[index].load(std::memory_order_relaxed) & 0xffffffffu);
        output_labels[index] = code == 0 ? -1 : static_cast<int32_t>(code - 1);
    }
    return 0;
}
