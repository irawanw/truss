// The kept cold copy's client side: see cold_keep.h for the protocol.
#include "runtime/cold_keep.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

namespace truss::runtime::cold_keep {

namespace {

int connect_to(const std::string & sock)
{
    sockaddr_un a {};
    if (sock.size() >= sizeof a.sun_path) return -1;
    const int s = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    timeval tv { 10, 0 };   // a hung keeper costs at most this, then the load reads the file as before
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    a.sun_family = AF_UNIX;
    std::memcpy(a.sun_path, sock.c_str(), sock.size() + 1);
    if (::connect(s, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0) {
        ::close(s);
        return -1;
    }
    return s;
}

// one line out, with an fd attached when fd >= 0
bool send_line(int s, const std::string & line, int fd)
{
    iovec io { const_cast<char *>(line.data()), line.size() };
    msghdr m {};
    m.msg_iov = &io, m.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))] = {};
    if (fd >= 0) {
        m.msg_control = ctl, m.msg_controllen = sizeof ctl;
        cmsghdr * c = CMSG_FIRSTHDR(&m);
        c->cmsg_level = SOL_SOCKET, c->cmsg_type = SCM_RIGHTS, c->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(c), &fd, sizeof(int));
    }
    return ::sendmsg(s, &m, MSG_NOSIGNAL) == (ssize_t) line.size();
}

// the reply line (up to '\n') and the fd that came with it (-1 if none)
std::string recv_line(int s, int * fd)
{
    char buf[64] = {};
    iovec io { buf, sizeof buf - 1 };
    msghdr m {};
    m.msg_iov = &io, m.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))] = {};
    m.msg_control = ctl, m.msg_controllen = sizeof ctl;
    *fd = -1;
    const ssize_t n = ::recvmsg(s, &m, MSG_CMSG_CLOEXEC);
    if (n <= 0) return "";
    for (cmsghdr * c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) std::memcpy(fd, CMSG_DATA(c), sizeof(int));
    std::string r(buf, (size_t) n);
    if (const size_t nl = r.find('\n'); nl != std::string::npos) r.resize(nl);
    return r;
}

std::string request(const char * verb, uint64_t slot, uint64_t key, size_t bytes)
{
    char line[96];
    std::snprintf(line, sizeof line, "%s %016llx %016llx %zu\n", verb, (unsigned long long) slot,
                  (unsigned long long) key, bytes);
    return line;
}

}  // namespace

std::string socket_path()
{
    const char * e = std::getenv("TRUSS_COLD_KEEP");
    return e && *e && std::strcmp(e, "0") != 0 ? std::string(e) : std::string();
}

int get(const std::string & sock, uint64_t slot, uint64_t key, size_t bytes)
{
    const int s = connect_to(sock);
    if (s < 0) return -1;
    int fd = -1;
    std::string r;
    if (send_line(s, request("GET", slot, key, bytes), -1)) r = recv_line(s, &fd);
    ::close(s);
    if (r != "HIT" && fd >= 0) ::close(fd), fd = -1;
    return fd;
}

bool put(const std::string & sock, uint64_t slot, uint64_t key, size_t bytes, int fd)
{
    const int s = connect_to(sock);
    if (s < 0) return false;
    int extra = -1;
    std::string r;
    if (send_line(s, request("PUT", slot, key, bytes), fd)) r = recv_line(s, &extra);
    ::close(s);
    if (extra >= 0) ::close(extra);
    return r == "OK";
}

}  // namespace truss::runtime::cold_keep
