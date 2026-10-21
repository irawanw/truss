// Internal to the GGUF reader: the mapping registry behind read_mapped() / read_direct() (formats/gguf_io.cc).
#pragma once

#include <cstddef>
#include <string>

namespace truss::gguf {

// a shard File::add_shard() mapped (fd stays open for the reads) / is unmapping
void register_mapping(void * addr, size_t size, int fd, const std::string & path);
void unregister_mapping(void * addr);

}  // namespace truss::gguf
