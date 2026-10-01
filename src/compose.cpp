// compose: splice X-ray / MRI / ultrasound tiles into a human silhouette.
// Tiles are pasted as hard-edged rectangles (blocky borders kept on purpose)
// over a soft, speckled "bone-scan" body outline.
//
// usage: compose [layout.txt] [out.png] [width] [height]
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
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Img {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
    unsigned char& at(int x, int y) { return px[size_t(y) * w + x]; }
};

static Img load(const std::string& path) {
    Img im;
    int c;
    unsigned char* d = stbi_load(path.c_str(), &im.w, &im.h, &c, 1);
    if (!d) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return im; }
    im.px.assign(d, d + size_t(im.w) * im.h);
    stbi_image_free(d);
    return im;
}

static Img rot90(const Img& s, int turns) {
    Img r = s;
    for (int t = 0; t < (turns & 3); ++t) {
        Img n; n.w = r.h; n.h = r.w; n.px.resize(r.px.size());
        for (int y = 0; y < r.h; ++y)
            for (int x = 0; x < r.w; ++x)
                n.px[size_t(x) * n.w + (r.h - 1 - y)] = r.px[size_t(y) * r.w + x];
        r = std::move(n);
    }
    return r;
}

// Crop the fractional source rect, then center-crop to the target aspect.
static Img fit(const Img& s, float fx, float fy, float fw, float fh, int tw, int th) {
    int x0 = int(fx * s.w), y0 = int(fy * s.h);
    int cw = std::max(1, int(fw * s.w)), ch = std::max(1, int(fh * s.h));
    float ta = float(tw) / th;
    if (float(cw) / ch > ta) { int nw = int(ch * ta); x0 += (cw - nw) / 2; cw = nw; }
    else                     { int nh = int(cw / ta); y0 += (ch - nh) / 2; ch = nh; }
    x0 = std::clamp(x0, 0, s.w - 1); y0 = std::clamp(y0, 0, s.h - 1);
    cw = std::min(cw, s.w - x0); ch = std::min(ch, s.h - y0);
    Img out; out.w = tw; out.h = th; out.px.resize(size_t(tw) * th);
    stbir_resize_uint8_linear(s.px.data() + size_t(y0) * s.w + x0, cw, ch, s.w,
                              out.px.data(), tw, th, tw, STBIR_1CHANNEL);
    return out;
}

// --- silhouette: union of capsules, rendered as a soft speckled glow --------
struct Cap { float ax, ay, bx, by, r; };

static float capDist(const Cap& c, float x, float y) {
    float vx = c.bx - c.ax, vy = c.by - c.ay, wx = x - c.ax, wy = y - c.ay;
    float t = std::clamp((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-6f), 0.f, 1.f);
    float dx = wx - t * vx, dy = wy - t * vy;
    return std::sqrt(dx * dx + dy * dy) - c.r;
}

static std::vector<Cap> body(float W, float H) {
    auto P = [&](float x, float y, float x2, float y2, float r) {
        return Cap{x * W, y * H, x2 * W, y2 * H, r * W};
    };
    return {
        P(.50f, .085f, .50f, .095f, .085f),   // head
        P(.50f, .15f, .50f, .19f, .045f),     // neck
        P(.40f, .23f, .60f, .23f, .07f),      // shoulders
        P(.50f, .25f, .50f, .46f, .15f),      // torso
        P(.50f, .50f, .50f, .56f, .16f),      // pelvis
        P(.29f, .24f, .21f, .40f, .045f),     // L upper arm
        P(.21f, .40f, .17f, .56f, .038f),     // L forearm
        P(.17f, .58f, .16f, .62f, .04f),      // L hand
        P(.71f, .24f, .79f, .40f, .045f),     // R upper arm
        P(.79f, .40f, .83f, .56f, .038f),     // R forearm
        P(.83f, .58f, .84f, .62f, .04f),      // R hand
        P(.42f, .58f, .41f, .78f, .07f),      // L thigh
        P(.41f, .78f, .41f, .95f, .05f),      // L shin
        P(.58f, .58f, .59f, .78f, .07f),      // R thigh
        P(.59f, .78f, .59f, .95f, .05f),      // R shin
        P(.40f, .965f, .36f, .975f, .03f),    // L foot
        P(.60f, .965f, .64f, .975f, .03f),    // R foot
    };
}

static void drawSilhouette(Img& c, unsigned seed) {
    auto caps = body(float(c.w), float(c.h));
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x) {
            float d = 1e9f;
            for (auto& k : caps) d = std::min(d, capDist(k, float(x), float(y)));
            // inside: mid-grey with brighter edge falloff; outside: fade to paper
            float a = 1.f / (1.f + std::exp(d / 6.f));
            float v = 245.f - a * (150.f + 30.f * n(rng)) - 6.f * std::abs(n(rng));
            c.at(x, y) = (unsigned char)std::clamp(v, 0.f, 255.f);
        }
}

// --- layout ----------------------------------------------------------------
// Each non-comment line:
//   file  x y w h  [sx sy sw sh]  [rot90]  [border]
// x,y,w,h are fractions of the canvas; sx..sh fractions of the source image.
struct Tile {
    std::string file;
    float x, y, w, h, sx = 0, sy = 0, sw = 1, sh = 1;
    int rot = 0, border = 2;
};

static std::vector<Tile> readLayout(const std::string& path) {
    std::vector<Tile> v;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Tile t;
        if (!(ss >> t.file >> t.x >> t.y >> t.w >> t.h)) continue;
        if (ss >> t.sx) ss >> t.sy >> t.sw >> t.sh;
        if (ss >> t.rot) ss >> t.border;
        v.push_back(t);
    }
    return v;
}

int main(int argc, char** argv) {
    std::string layoutPath = argc > 1 ? argv[1] : "layout.txt";
    std::string out = argc > 2 ? argv[2] : "out/composite.png";
    int W = argc > 3 ? std::atoi(argv[3]) : 900;
    int H = argc > 4 ? std::atoi(argv[4]) : 1800;

    auto tiles = readLayout(layoutPath);
    if (tiles.empty()) { std::fprintf(stderr, "no tiles in %s\n", layoutPath.c_str()); return 1; }
    std::string dir = layoutPath.substr(0, layoutPath.find_last_of('/') + 1);

    Img canvas; canvas.w = W; canvas.h = H; canvas.px.resize(size_t(W) * H);
    drawSilhouette(canvas, 7);

    std::map<std::string, Img> cache;
    for (auto& t : tiles) {
        auto it = cache.find(t.file);
        if (it == cache.end()) it = cache.emplace(t.file, load(dir + t.file)).first;
        if (it->second.px.empty()) continue;
        Img src = rot90(it->second, t.rot);
        int tx = int(t.x * W), ty = int(t.y * H), tw = int(t.w * W), th = int(t.h * H);
        Img tile = fit(src, t.sx, t.sy, t.sw, t.sh, tw, th);
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
                int cx = tx + x, cy = ty + y;
                if (cx < 0 || cy < 0 || cx >= W || cy >= H) continue;
                bool edge = x < t.border || y < t.border || x >= tw - t.border || y >= th - t.border;
                canvas.at(cx, cy) = edge ? 20 : tile.at(x, y);
            }
    }

    if (!stbi_write_png(out.c_str(), W, H, 1, canvas.px.data(), W)) {
        std::fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d, %zu tiles)\n", out.c_str(), W, H, tiles.size());
}
