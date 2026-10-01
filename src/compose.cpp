// compose: splice X-ray / MRI / ultrasound tiles over a whole-body bone scan
// so the anatomy lines up, while each tile keeps a hard rectangular border.
//
// usage: compose [layout.txt] [out.png]
//
// Layout coordinates are percentages of a reference "space" (the padded bone
// scan). Each tile pins a source landmark (fraction of the source image) to a
// target landmark (percent of space), with a width and rotation, e.g. the
// femoral heads of a pelvis film onto the hips of the bone scan.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Img {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
    unsigned char at(int x, int y) const { return px[size_t(y) * w + x]; }
    float sample(float x, float y) const {  // bilinear, clamped
        x = std::clamp(x, 0.f, w - 1.001f); y = std::clamp(y, 0.f, h - 1.001f);
        int x0 = int(x), y0 = int(y); float fx = x - x0, fy = y - y0;
        float a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
    }
};

static Img load(const std::string& path) {
    Img im; int c;
    unsigned char* d = stbi_load(path.c_str(), &im.w, &im.h, &c, 1);
    if (!d) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return im; }
    im.px.assign(d, d + size_t(im.w) * im.h);
    stbi_image_free(d);
    return im;
}

static Img resized(const Img& s, float f) {
    Img o; o.w = std::max(1, int(s.w * f)); o.h = std::max(1, int(s.h * f));
    o.px.resize(size_t(o.w) * o.h);
    stbir_resize_uint8_linear(s.px.data(), s.w, s.h, s.w, o.px.data(), o.w, o.h, o.w, STBIR_1CHANNEL);
    return o;
}

struct Tile {
    std::string file;
    float sx, sy, sw, sh;   // crop, fraction of source
    float ax, ay;           // source anchor, fraction of source
    float dx, dy;           // target anchor, percent of space
    float width;            // crop width on canvas, percent of space width
    float angle;            // degrees, clockwise
    int border = 3;
};

struct Layout {
    float spaceW = 512, spaceH = 1088, scale = 2;
    float vx = 0, vy = 0, vw = 100, vh = 100;
    std::string bg; bool bgInvert = true; float bgGamma = 1, bgGain = 1, bgX = -1, bgY = -1;
    std::vector<Tile> tiles;
};

static Layout readLayout(const std::string& path) {
    Layout L;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string k;
        if (!(ss >> k) || k[0] == '#') continue;
        if (k == "space") ss >> L.spaceW >> L.spaceH;
        else if (k == "scale") ss >> L.scale;
        else if (k == "view") ss >> L.vx >> L.vy >> L.vw >> L.vh;
        else if (k == "bg") { int inv = 1; ss >> L.bg >> inv >> L.bgGain >> L.bgGamma >> L.bgX >> L.bgY; L.bgInvert = inv != 0; }
        else if (k == "tile") {
            Tile t;
            if (ss >> t.file >> t.sx >> t.sy >> t.sw >> t.sh >> t.ax >> t.ay >> t.dx >> t.dy >> t.width >> t.angle) {
                ss >> t.border;
                L.tiles.push_back(t);
            } else std::fprintf(stderr, "bad tile line: %s\n", line.c_str());
        }
    }
    return L;
}

int main(int argc, char** argv) {
    std::string layoutPath = argc > 1 ? argv[1] : "layout.txt";
    std::string out = argc > 2 ? argv[2] : "out/composite.png";
    Layout L = readLayout(layoutPath);
    std::string dir = layoutPath.substr(0, layoutPath.find_last_of('/') + 1);

    // canvas pixels per percent of space
    const float kx = L.spaceW * L.scale / 100.f, ky = L.spaceH * L.scale / 100.f;
    const int W = int(L.vw * kx), H = int(L.vh * ky);
    const float ox = L.vx * kx, oy = L.vy * ky;  // canvas origin in space pixels
    std::vector<unsigned char> canvas(size_t(W) * H, 255);

    if (!L.bg.empty()) {
        Img bg = load(dir + L.bg);
        if (!bg.px.empty()) {
            // background sits in the space at its native pixel size, centred
            // unless an explicit top-left offset (space pixels) is given
            float s = L.scale;
            float bx = (L.bgX >= 0 ? L.bgX : (L.spaceW - bg.w) / 2.f) * s;
            float by = (L.bgY >= 0 ? L.bgY : (L.spaceH - bg.h) / 2.f) * s;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float u = (x + ox - bx) / s, v = (y + oy - by) / s;
                    float val = (u < 0 || v < 0 || u >= bg.w || v >= bg.h) ? 0.f : bg.sample(u, v) / 255.f;
                    val = std::pow(std::min(1.f, val * L.bgGain), 1.f / L.bgGamma);
                    if (L.bgInvert) val = 1.f - val;
                    canvas[size_t(y) * W + x] = (unsigned char)(val * 255.f);
                }
        }
    }

    for (const Tile& t : L.tiles) {
        Img src = load(dir + t.file);
        if (src.px.empty()) continue;
        float cropW = t.sw * src.w;
        float s = t.width * kx / cropW;              // canvas px per source px
        if (s < 0.5f) {                              // pre-shrink to avoid aliasing
            float f = std::min(1.f, 2.f * s);
            Img r = resized(src, f);
            src = std::move(r);
            cropW = t.sw * src.w;
            s = t.width * kx / cropW;
        }
        const float th = t.angle * 3.14159265f / 180.f, c = std::cos(th), sn = std::sin(th);
        const float ax = t.ax * src.w, ay = t.ay * src.h;
        const float Dx = t.dx * kx - ox, Dy = t.dy * ky - oy;
        const float x0 = t.sx * src.w, y0 = t.sy * src.h, x1 = x0 + cropW, y1 = y0 + t.sh * src.h;
        const float bpx = t.border / s;              // border thickness in source px

        // canvas bbox of the rotated crop
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        for (float u : {x0, x1}) for (float v : {y0, y1}) {
            float px = Dx + s * ((u - ax) * c - (v - ay) * sn);
            float py = Dy + s * ((u - ax) * sn + (v - ay) * c);
            minx = std::min(minx, px); maxx = std::max(maxx, px);
            miny = std::min(miny, py); maxy = std::max(maxy, py);
        }
        int bx0 = std::max(0, int(minx)), bx1 = std::min(W - 1, int(maxx) + 1);
        int by0 = std::max(0, int(miny)), by1 = std::min(H - 1, int(maxy) + 1);
        for (int y = by0; y <= by1; ++y)
            for (int x = bx0; x <= bx1; ++x) {
                float qx = (x - Dx) / s, qy = (y - Dy) / s;  // inverse rotate
                float u = ax + qx * c + qy * sn, v = ay - qx * sn + qy * c;
                if (u < x0 || v < y0 || u >= x1 || v >= y1) continue;
                bool edge = u < x0 + bpx || v < y0 + bpx || u >= x1 - bpx || v >= y1 - bpx;
                canvas[size_t(y) * W + x] = edge ? 25 : (unsigned char)src.sample(u, v);
            }
    }

    if (!stbi_write_png(out.c_str(), W, H, 1, canvas.data(), W)) {
        std::fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d, %zu tiles)\n", out.c_str(), W, H, L.tiles.size());
}
