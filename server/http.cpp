#include "server/http.h"
#include "core/json.h"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if defined(__MINGW32__)
    #include <sys/types.h>
  #endif
  #include <winsock2.h>
  // ws2tcpip.h is not needed: sockaddr_in, htons/ntohs, inet_addr and
  // INADDR_ANY all live in winsock2.h, and pulling ws2tcpip.h in breaks the
  // MinGW build (it wants the BSD socket types from sys/types.h).
  #ifdef _MSC_VER
    #pragma comment(lib, "ws2_32.lib")
  #endif
  #define GAI_INVALID_SOCKET INVALID_SOCKET
  #define closesocket_fn closesocket
  // Windows takes an int* for the address length; POSIX takes socklen_t*.
  using HttpSockLen = int;
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #define GAI_INVALID_SOCKET (-1)
  #define closesocket_fn close
  using HttpSockLen = socklen_t;
#endif

namespace gai {

bool http_socket_init(std::string* error) {
#ifdef _WIN32
    WSADATA wsa;
    const int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (rc != 0) {
        if (error) *error = "WSAStartup failed: " + std::to_string(rc);
        return false;
    }
#else
    (void)error;
#endif
    return true;
}

void http_socket_shutdown() {
#ifdef _WIN32
    WSACleanup();
#endif
}

namespace {

std::string lower(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

bool read_line(HttpSocket fd, std::string& buf, std::string& line, size_t limit = 16384) {
    while (true) {
        const size_t nl = buf.find("\r\n");
        if (nl != std::string::npos) {
            line = buf.substr(0, nl);
            buf.erase(0, nl + 2);
            return true;
        }
        if (buf.size() > limit) return false;
        char tmp[4096];
        const int n = static_cast<int>(recv(fd, tmp, sizeof(tmp), 0));
        if (n <= 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
    }
}

bool read_exact(HttpSocket fd, std::string& buf, size_t n, std::string& out) {
    while (buf.size() < n) {
        char tmp[8192];
        const int r = static_cast<int>(recv(fd, tmp, sizeof(tmp), 0));
        if (r <= 0) return false;
        buf.append(tmp, static_cast<size_t>(r));
    }
    out = buf.substr(0, n);
    buf.erase(0, n);
    return true;
}

const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "OK";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// HttpConn
// ---------------------------------------------------------------------------

bool HttpConn::write_all(const char* data, size_t n) {
    size_t sent = 0;
    while (sent < n) {
#ifdef _WIN32
        const int r = send(fd_, data + sent, static_cast<int>(n - sent), 0);
#else
        const ssize_t r = ::send(fd_, data + sent, n - sent, MSG_NOSIGNAL);
#endif
        if (r <= 0) return false;
        sent += static_cast<size_t>(r);
    }
    return true;
}

void HttpConn::build_header_block(std::string& out, bool chunked, size_t content_length) {
    out = "HTTP/1.1 " + std::to_string(status_) + " " + status_text(status_) + "\r\n";
    out += "Server: ghassan-ai\r\n";
    out += "Connection: close\r\n";
    for (const auto& [k, v] : headers_) out += k + ": " + v + "\r\n";
    if (chunked) {
        out += "Transfer-Encoding: chunked\r\n\r\n";
    } else {
        out += "Content-Length: " + std::to_string(content_length) + "\r\n\r\n";
    }
}

bool HttpConn::send_response(const std::string& content_type, const std::string& body) {
    if (headers_sent_) return false;
    headers_sent_ = true;
    std::string head;
    if (headers_.find("Content-Type") == headers_.end()) {
        headers_["Content-Type"] = content_type;
    }
    build_header_block(head, false, body.size());
    if (!write_all(head.data(), head.size())) return false;
    if (!body.empty()) return write_all(body.data(), body.size());
    return true;
}

bool HttpConn::begin_stream(const std::string& content_type) {
    if (headers_sent_) return false;
    headers_sent_ = true;
    streaming_ = true;
    headers_["Content-Type"] = content_type;
    headers_["Cache-Control"] = "no-cache";
    headers_["X-Accel-Buffering"] = "no";
    std::string head;
    build_header_block(head, true, 0);
    return write_all(head.data(), head.size());
}

bool HttpConn::send_chunk(const std::string& data) {
    if (!streaming_ || !headers_sent_) return false;
    char sz[32];
    std::snprintf(sz, sizeof(sz), "%zx\r\n", data.size());
    if (!write_all(sz, strlen(sz))) return false;
    if (!data.empty() && !write_all(data.data(), data.size())) return false;
    return write_all("\r\n", 2);
}

bool HttpConn::end_stream() {
    if (!streaming_) return false;
    streaming_ = false;
    return write_all("0\r\n\r\n", 5);
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start(const std::string& host, int port, HttpHandler handler, std::string* error) {
    handler_ = std::move(handler);

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ == GAI_INVALID_SOCKET) {
        if (error) *error = "socket() failed";
        return false;
    }
    int yes = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(port));
    if (host.empty() || host == "0.0.0.0" || host == "*") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = inet_addr(host.c_str());
        if (addr.sin_addr.s_addr == INADDR_NONE) {
            if (error) *error = "bad host: " + host;
            closesocket_fn(fd_);
            fd_ = GAI_INVALID_SOCKET;
            return false;
        }
    }
    if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (error) *error = "bind() failed (port in use?)";
        closesocket_fn(fd_);
        fd_ = GAI_INVALID_SOCKET;
        return false;
    }
    if (listen(fd_, 64) != 0) {
        if (error) *error = "listen() failed";
        closesocket_fn(fd_);
        fd_ = GAI_INVALID_SOCKET;
        return false;
    }
    sockaddr_in bound{};
    HttpSockLen blen = sizeof(bound);
    if (getsockname(fd_, reinterpret_cast<sockaddr*>(&bound), &blen) == 0) {
        port_ = ntohs(bound.sin_port);
    } else {
        port_ = port;
    }
    running_ = true;
    return true;
}

void HttpServer::stop() {
    if (!running_) return;
    running_ = false;
    if (fd_ != GAI_INVALID_SOCKET) {
        closesocket_fn(fd_);
        fd_ = GAI_INVALID_SOCKET;
    }
}

void HttpServer::serve_forever() {
    while (running_) {
        sockaddr_in peer{};
        HttpSockLen plen = sizeof(peer);
        const HttpSocket cs = accept(fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (cs == GAI_INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }
        std::thread([this, cs]() {
            HttpConn conn(cs);
            std::string buf;
            HttpRequest req;

            // request line
            std::string line;
            if (read_line(cs, buf, line)) {
                const size_t sp1 = line.find(' ');
                const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos
                                                              : line.find(' ', sp1 + 1);
                if (sp1 != std::string::npos) {
                    req.method = line.substr(0, sp1);
                    const std::string target = (sp2 == std::string::npos)
                        ? line.substr(sp1 + 1) : line.substr(sp1 + 1, sp2 - sp1 - 1);
                    const size_t q = target.find('?');
                    if (q == std::string::npos) {
                        req.path = target;
                    } else {
                        req.path = target.substr(0, q);
                        req.query = target.substr(q + 1);
                    }
                }
            }
            // headers
            while (read_line(cs, buf, line)) {
                if (line.empty()) break;
                const size_t colon = line.find(':');
                if (colon == std::string::npos) continue;
                std::string key = lower(line.substr(0, colon));
                std::string val = line.substr(colon + 1);
                while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(0, 1);
                req.headers[key] = val;
            }
            // body
            const std::string clen = req.header("content-length");
            if (!clen.empty()) {
                const size_t n = static_cast<size_t>(std::strtoull(clen.c_str(), nullptr, 10));
                if (n > 64u * 1024u * 1024u) {
                    conn.send_response("application/json",
                                       "{\"error\":\"payload too large\"}");
                } else if (n > 0) {
                    std::string body;
                    if (!read_exact(cs, buf, n, body)) {
                        closesocket_fn(cs);
                        return;
                    }
                    req.body = std::move(body);
                }
            }

            try {
                handler_(conn, req);
            } catch (const std::exception& e) {
                Json err = Json::object();
                err.set("error", Json::string(e.what()));
                conn.send_response("application/json", err.dump());
            }

            if (conn.headers_sent()) conn.end_stream();
            shutdown(cs, 0);
            closesocket_fn(cs);
        }).detach();
    }
}

}  // namespace gai
