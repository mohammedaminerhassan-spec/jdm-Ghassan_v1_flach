#include "projector/image_video_loader.h"
#include "core/common.h"
#include <cmath>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if defined(_WIN32) || defined(_MSC_VER)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "projector/stb_image.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace gai {

namespace {





static bool run_capture(const std::vector<std::string>& args, std::vector<uint8_t>& out) {
    if (args.empty()) return false;
#if defined(_WIN32) || defined(_MSC_VER)
    
    
    std::string cmdline;
    for (const auto& a : args) {
        cmdline += '"';
        for (char c : a) {
            if (c == '"' || c == '\\') cmdline += '\\';
            cmdline += c;
        }
        cmdline += "\" ";
    }
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.hStdOutput = wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;
    PROCESS_INFORMATION pi{};
    std::wstring wcmd(cmdline.begin(), cmdline.end());
    const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE,
                                   0, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); return false; }
    uint8_t buf[65536];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof(buf), &got, nullptr) && got > 0)
        out.insert(out.end(), buf, buf + got);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return code == 0;
#else
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return false; }
    if (pid == 0) {
        close(fds[0]);
        if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(127);
        close(fds[1]);
        std::vector<char*> av;
        av.reserve(args.size() + 1);
        for (const auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        execvp(args[0].c_str(), av.data());
        _exit(127);
    }
    close(fds[1]);
    uint8_t buf[65536];
    ssize_t got = 0;
    bool read_error = false;
    while ((got = read(fds[0], buf, sizeof(buf))) > 0)
        out.insert(out.end(), buf, buf + got);
    if (got < 0) read_error = true;
    close(fds[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return !read_error && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

} 

bool load_ppm_raw(const std::string& path, float* out, int C, int H, int W) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    auto skip_ws = [&]() {
        while (true) {
            const int c = f.peek();
            if (c == '#') {
                std::string junk;
                std::getline(f, junk);
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { f.get(); continue; }
            break;
        }
    };
    auto read_int = [&](int& v) {
        skip_ws();
        std::string s;
        while (true) {
            const int c = f.peek();
            if (c < '0' || c > '9') break;
            s.push_back(static_cast<char>(f.get()));
        }
        if (s.empty()) return false;
        v = std::atoi(s.c_str());
        return true;
    };

    char magic[3] = {0, 0, 0};
    f.read(magic, 2);
    if (magic[0] != 'P') return false;
    const bool color = (magic[1] == '6');
    if (!color && magic[1] != '5') return false;

    int w = 0, h = 0, maxv = 0;
    if (!read_int(w) || !read_int(h) || !read_int(maxv)) return false;
    if (w != W || h != H || maxv <= 0) return false;

    {
        const int c = f.get();
        if (c == '\r' && f.peek() == '\n') f.get();
        else if (c < 0) return false;
    }

    const int src_channels = color ? 3 : 1;
    const float scale = 1.0f / static_cast<float>(maxv);

    auto store = [&](int x, int y, int c, float v) {
        const size_t base = static_cast<size_t>(y) * W + x;
        if (src_channels == 3) {
            out[static_cast<size_t>(c) * H * W + base] = v;
        } else {
            for (int cc = 0; cc < C; ++cc) out[static_cast<size_t>(cc) * H * W + base] = v;
        }
    };

    if (magic[1] == '6') {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const int v = f.get();
                    if (v < 0) return false;
                    store(x, y, c, static_cast<float>(v) * scale);
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int v = 0;
                if (!read_int(v)) return false;
                store(x, y, 0, static_cast<float>(v) * scale);
            }
        }
    }
    return true;
}

bool resize_bilinear_rgb(const uint8_t* src, int src_w, int src_h,
                         float* dst_chw, int dst_w, int dst_h,
                         bool keep_aspect_ratio) {
    if (!src || !dst_chw || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }

    const float inv255 = 1.0f / 255.0f;
    const size_t plane_stride = static_cast<size_t>(dst_w) * dst_h;
    float* r_plane = dst_chw;
    float* g_plane = r_plane + plane_stride;
    float* b_plane = g_plane + plane_stride;

    if (src_w == dst_w && src_h == dst_h) {
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < dst_w * dst_h; ++i) {
            const size_t src_idx = static_cast<size_t>(i) * 3;
            r_plane[i] = static_cast<float>(src[src_idx + 0]) * inv255;
            g_plane[i] = static_cast<float>(src[src_idx + 1]) * inv255;
            b_plane[i] = static_cast<float>(src[src_idx + 2]) * inv255;
        }
        return true;
    }

    if (keep_aspect_ratio) {
        const float scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                                     static_cast<float>(dst_h) / static_cast<float>(src_h));
        const int scaled_w = std::max(1, std::min(dst_w, static_cast<int>(std::round(src_w * scale))));
        const int scaled_h = std::max(1, std::min(dst_h, static_cast<int>(std::round(src_h * scale))));
        const int offset_x = (dst_w - scaled_w) / 2;
        const int offset_y = (dst_h - scaled_h) / 2;

        const float bg_color = 0.5f;
#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < dst_w * dst_h; ++i) {
            r_plane[i] = bg_color;
            g_plane[i] = bg_color;
            b_plane[i] = bg_color;
        }

        const float src_to_scaled_x = static_cast<float>(src_w) / static_cast<float>(scaled_w);
        const float src_to_scaled_y = static_cast<float>(src_h) / static_cast<float>(scaled_h);

#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int sy = 0; sy < scaled_h; ++sy) {
            const int dst_y = offset_y + sy;
            const float src_yf = (static_cast<float>(sy) + 0.5f) * src_to_scaled_y - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(src_yf)), 0, src_h - 1);
            const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
            const float wy1 = std::clamp(src_yf - static_cast<float>(y0), 0.0f, 1.0f);
            const float wy0 = 1.0f - wy1;

            const size_t row_dst_offset = static_cast<size_t>(dst_y) * dst_w;
            const size_t row_src0_offset = static_cast<size_t>(y0) * src_w * 3;
            const size_t row_src1_offset = static_cast<size_t>(y1) * src_w * 3;

            for (int sx = 0; sx < scaled_w; ++sx) {
                const int dst_x = offset_x + sx;
                const float src_xf = (static_cast<float>(sx) + 0.5f) * src_to_scaled_x - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(src_xf)), 0, src_w - 1);
                const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
                const float wx1 = std::clamp(src_xf - static_cast<float>(x0), 0.0f, 1.0f);
                const float wx0 = 1.0f - wx1;

                const float w00 = wx0 * wy0;
                const float w10 = wx1 * wy0;
                const float w01 = wx0 * wy1;
                const float w11 = wx1 * wy1;

                const uint8_t* p00 = src + row_src0_offset + static_cast<size_t>(x0) * 3;
                const uint8_t* p10 = src + row_src0_offset + static_cast<size_t>(x1) * 3;
                const uint8_t* p01 = src + row_src1_offset + static_cast<size_t>(x0) * 3;
                const uint8_t* p11 = src + row_src1_offset + static_cast<size_t>(x1) * 3;

                const size_t d_idx = row_dst_offset + dst_x;
                r_plane[d_idx] = (w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0]) * inv255;
                g_plane[d_idx] = (w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1]) * inv255;
                b_plane[d_idx] = (w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2]) * inv255;
            }
        }
        return true;
    }

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int dst_y = 0; dst_y < dst_h; ++dst_y) {
        const float src_yf = (static_cast<float>(dst_y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_yf)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy1 = std::clamp(src_yf - static_cast<float>(y0), 0.0f, 1.0f);
        const float wy0 = 1.0f - wy1;

        const size_t row_dst_offset = static_cast<size_t>(dst_y) * dst_w;
        const size_t row_src0_offset = static_cast<size_t>(y0) * src_w * 3;
        const size_t row_src1_offset = static_cast<size_t>(y1) * src_w * 3;

        for (int dst_x = 0; dst_x < dst_w; ++dst_x) {
            const float src_xf = (static_cast<float>(dst_x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_xf)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx1 = std::clamp(src_xf - static_cast<float>(x0), 0.0f, 1.0f);
            const float wx0 = 1.0f - wx1;

            const float w00 = wx0 * wy0;
            const float w10 = wx1 * wy0;
            const float w01 = wx0 * wy1;
            const float w11 = wx1 * wy1;

            const uint8_t* p00 = src + row_src0_offset + static_cast<size_t>(x0) * 3;
            const uint8_t* p10 = src + row_src0_offset + static_cast<size_t>(x1) * 3;
            const uint8_t* p01 = src + row_src1_offset + static_cast<size_t>(x0) * 3;
            const uint8_t* p11 = src + row_src1_offset + static_cast<size_t>(x1) * 3;

            const size_t d_idx = row_dst_offset + dst_x;
            r_plane[d_idx] = (w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0]) * inv255;
            g_plane[d_idx] = (w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1]) * inv255;
            b_plane[d_idx] = (w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2]) * inv255;
        }
    }
    return true;
}

ImageBuffer load_image_from_memory(const uint8_t* data, size_t len, int target_w, int target_h,
                                   bool keep_aspect_ratio) {
    ImageBuffer buf;
    buf.width = target_w;
    buf.height = target_h;
    buf.channels = 3;
    if (!data || len == 0 || target_w <= 0 || target_h <= 0) return buf;

    int w = 0, h = 0, comp = 0;
    stbi_uc* raw_rgb = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &comp, 3);
    if (!raw_rgb) {
        return buf;
    }

    buf.chw_data.assign(static_cast<size_t>(3) * target_w * target_h, 0.0f);
    resize_bilinear_rgb(raw_rgb, w, h, buf.chw_data.data(), target_w, target_h, keep_aspect_ratio);
    stbi_image_free(raw_rgb);
    buf.valid = true;
    return buf;
}

ImageBuffer load_image_any(const std::string& path, int target_w, int target_h,
                           bool keep_aspect_ratio) {
    ImageBuffer buf;
    buf.width = target_w;
    buf.height = target_h;
    buf.channels = 3;
    buf.chw_data.assign(static_cast<size_t>(3) * target_w * target_h, 0.0f);

    int w = 0, h = 0, comp = 0;
    stbi_uc* raw_rgb = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (raw_rgb) {
        resize_bilinear_rgb(raw_rgb, w, h, buf.chw_data.data(), target_w, target_h, keep_aspect_ratio);
        stbi_image_free(raw_rgb);
        buf.valid = true;
        return buf;
    }

    if (path.size() >= 4 && path.substr(path.size() - 4) == ".ppm") {
        if (load_ppm_raw(path, buf.chw_data.data(), 3, target_h, target_w)) {
            buf.valid = true;
            return buf;
        }
    }

    const std::string scale = strfmt("scale=%d:%d", target_w, target_h);
    std::vector<uint8_t> ffmpeg_rgb;
    const bool ffmpeg_ok = run_capture({"ffmpeg", "-v", "error", "-i", path,
                                        "-vf", scale, "-f", "rawvideo",
                                        "-pix_fmt", "rgb24", "-"}, ffmpeg_rgb);

    const size_t raw_bytes = static_cast<size_t>(target_w) * target_h * 3;
    if (!ffmpeg_ok || ffmpeg_rgb.size() < raw_bytes) {
        log_warn(strfmt("[vision] ffmpeg could not decode %s (%zu of %zu bytes)",
                        path.c_str(), ffmpeg_rgb.size(), raw_bytes));
        return buf;
    }

    const float inv255 = 1.0f / 255.0f;
    const size_t plane_stride = static_cast<size_t>(target_w) * target_h;
    float* r_plane = buf.chw_data.data();
    float* g_plane = r_plane + plane_stride;
    float* b_plane = g_plane + plane_stride;

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < target_w * target_h; ++i) {
        const size_t src_idx = static_cast<size_t>(i) * 3;
        r_plane[i] = static_cast<float>(ffmpeg_rgb[src_idx + 0]) * inv255;
        g_plane[i] = static_cast<float>(ffmpeg_rgb[src_idx + 1]) * inv255;
        b_plane[i] = static_cast<float>(ffmpeg_rgb[src_idx + 2]) * inv255;
    }

    buf.valid = true;
    return buf;
}

VideoBuffer load_video_any(const std::string& path, int max_frames, int target_w, int target_h,
                           bool keep_aspect_ratio) {
    VideoBuffer vbuf;
    vbuf.width = target_w;
    vbuf.height = target_h;
    vbuf.channels = 3;

    if (max_frames <= 0) return vbuf;

    std::string vf_filter = keep_aspect_ratio
        ? strfmt("fps=2,scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=gray",
                 target_w, target_h, target_w, target_h)
        : strfmt("fps=2,scale=%d:%d", target_w, target_h);

    std::vector<uint8_t> raw_all;
    const bool ffmpeg_ok = run_capture({"ffmpeg", "-v", "error", "-i", path,
                                        "-vf", vf_filter, "-f", "rawvideo",
                                        "-pix_fmt", "rgb24", "-"}, raw_all);

    const size_t frame_bytes = static_cast<size_t>(target_w) * target_h * 3;
    const float inv255 = 1.0f / 255.0f;
    const size_t plane_stride = static_cast<size_t>(target_w) * target_h;

    std::vector<std::vector<float>> all_extracted_frames;
    const uint8_t* cursor = raw_all.data();
    size_t remaining = raw_all.size();

    while (ffmpeg_ok && remaining >= frame_bytes) {
        std::vector<float> chw(static_cast<size_t>(3) * target_w * target_h);
        float* r_plane = chw.data();
        float* g_plane = r_plane + plane_stride;
        float* b_plane = g_plane + plane_stride;

#ifdef GAI_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < target_w * target_h; ++i) {
            const size_t src_idx = static_cast<size_t>(i) * 3;
            r_plane[i] = static_cast<float>(cursor[src_idx + 0]) * inv255;
            g_plane[i] = static_cast<float>(cursor[src_idx + 1]) * inv255;
            b_plane[i] = static_cast<float>(cursor[src_idx + 2]) * inv255;
        }
        all_extracted_frames.push_back(std::move(chw));
        cursor += frame_bytes;
        remaining -= frame_bytes;
    }

    if (all_extracted_frames.empty()) {
        log_warn("[vision] no video frames could be extracted from: " + path);
        return vbuf;
    }

    if (static_cast<int>(all_extracted_frames.size()) <= max_frames) {
        vbuf.frames = std::move(all_extracted_frames);
    } else if (max_frames == 1) {
        vbuf.frames.push_back(std::move(all_extracted_frames[all_extracted_frames.size() / 2]));
    } else {
        vbuf.frames.reserve(static_cast<size_t>(max_frames));
        const double step = static_cast<double>(all_extracted_frames.size() - 1) /
                            static_cast<double>(max_frames - 1);
        for (int i = 0; i < max_frames; ++i) {
            size_t idx = static_cast<size_t>(std::round(i * step));
            if (idx >= all_extracted_frames.size()) idx = all_extracted_frames.size() - 1;
            vbuf.frames.push_back(std::move(all_extracted_frames[idx]));
        }
    }

    vbuf.num_frames = static_cast<int>(vbuf.frames.size());
    vbuf.valid = true;
    return vbuf;
}

}
