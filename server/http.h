#pragma once

#include "core/common.h"

#include <functional>
#include <map>
#include <string>

namespace gai {

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if defined(__MINGW32__)
    // MinGW's ws2ipdef.h needs the BSD socket types (u_char, u_short, ...).
    #include <sys/types.h>
  #endif
  #include <winsock2.h>
  // SOCKET is an unsigned pointer sized handle; keep the platform type so
  // the compiler does not complain about signedness or truncation.
  using HttpSocket = SOCKET;
#else
  using HttpSocket = int;
#endif

// Minimal HTTP/1.1 server for the serving path.
//
// It is deliberately small: request line, headers, a Content-Length body,
// chunked Server-Sent Events for streaming answers, and connection close
// after each response. No keep-alive, no TLS, no dependencies.

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::map<std::string, std::string> headers;
    std::string body;

    std::string header(const std::string& key) const {
        auto it = headers.find(key);
        return it == headers.end() ? std::string() : it->second;
    }
};

class HttpConn {
public:
    explicit HttpConn(HttpSocket fd) : fd_(fd) {}

    HttpSocket fd() const { return fd_; }

    void set_status(int code) { status_ = code; }
    void set_header(const std::string& key, const std::string& value) {
        headers_[key] = value;
    }

    // Sends a complete response.
    bool send_response(const std::string& content_type, const std::string& body);

    // Starts a streamed response (chunked transfer encoding).
    bool begin_stream(const std::string& content_type);

    // Writes one streamed chunk.
    bool send_chunk(const std::string& data);

    // Ends a streamed response.
    bool end_stream();

    bool headers_sent() const { return headers_sent_; }

private:
    bool write_all(const char* data, size_t n);
    void build_header_block(std::string& out, bool chunked, size_t content_length);

    HttpSocket   fd_ = static_cast<HttpSocket>(-1);
    int         status_ = 200;
    bool        headers_sent_ = false;
    bool        streaming_ = false;
    std::map<std::string, std::string> headers_;
};

using HttpHandler = std::function<void(HttpConn&, const HttpRequest&)>;

class HttpServer {
public:
    ~HttpServer();

    // Binds and listens. Returns false with a message on failure.
    bool start(const std::string& host, int port, HttpHandler handler, std::string* error);

    // Accepts connections until stop() is called.
    void serve_forever();

    void stop();

    int port() const { return port_; }

private:
    HttpSocket   fd_ = static_cast<HttpSocket>(-1);
    int         port_ = 0;
    HttpHandler handler_;
    bool        running_ = false;
};

// Initializes the platform socket layer (no-op outside Windows).
bool http_socket_init(std::string* error);
void http_socket_shutdown();

}  // namespace gai
