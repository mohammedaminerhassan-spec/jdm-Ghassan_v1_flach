#include "format/parquet.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>

namespace gai {

namespace snappy {

bool decompress(const uint8_t* src, size_t src_len, std::vector<uint8_t>& dst) {
    if (src == nullptr || src_len == 0) return false;

    size_t src_pos = 0;

    uint64_t len64 = 0;
    int shift = 0;
    bool terminated = false;
    while (src_pos < src_len && shift <= 28) {
        uint8_t b = src[src_pos++];
        len64 |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) { terminated = true; break; }
        shift += 7;
    }
    if (!terminated) return false;

    if (len64 == 0) {
        dst.clear();
        return true;
    }
    if (len64 > (1ULL << 30)) return false;
    const uint64_t max_expansion = static_cast<uint64_t>(src_len) * 64ULL + 1024ULL;
    if (len64 > max_expansion) return false;
    const uint32_t uncompressed_len = static_cast<uint32_t>(len64);

    dst.resize(uncompressed_len);
    size_t dst_pos = 0;

    while (src_pos < src_len && dst_pos < uncompressed_len) {
        uint8_t tag = src[src_pos++];
        uint8_t tag_type = tag & 0x03;

        if (tag_type == 0x00) {
            size_t len = (tag >> 2) + 1;
            if (len > 60) {
                size_t extra_bytes = len - 60;
                if (src_pos + extra_bytes > src_len) return false;
                len = 0;
                for (size_t i = 0; i < extra_bytes; ++i) {
                    len |= static_cast<size_t>(src[src_pos++]) << (i * 8);
                }
                len += 1;
            }
            if (src_pos + len > src_len || dst_pos + len > uncompressed_len) return false;
            std::memcpy(dst.data() + dst_pos, src + src_pos, len);
            src_pos += len;
            dst_pos += len;
        } else if (tag_type == 0x01) {
            size_t len = ((tag >> 2) & 0x07) + 4;
            if (src_pos >= src_len) return false;
            size_t offset = ((static_cast<size_t>(tag & 0xE0) << 3)) | src[src_pos++];
            if (offset == 0 || offset > dst_pos || dst_pos + len > uncompressed_len) return false;
            for (size_t i = 0; i < len; ++i) {
                dst[dst_pos + i] = dst[dst_pos - offset + i];
            }
            dst_pos += len;
        } else if (tag_type == 0x02) {
            size_t len = (tag >> 2) + 1;
            if (src_pos + 2 > src_len) return false;
            size_t offset = src[src_pos] | (static_cast<size_t>(src[src_pos + 1]) << 8);
            src_pos += 2;
            if (offset == 0 || offset > dst_pos || dst_pos + len > uncompressed_len) return false;
            for (size_t i = 0; i < len; ++i) {
                dst[dst_pos + i] = dst[dst_pos - offset + i];
            }
            dst_pos += len;
        } else if (tag_type == 0x03) {
            size_t len = (tag >> 2) + 1;
            if (src_pos + 4 > src_len) return false;
            size_t offset = src[src_pos] |
                            (static_cast<size_t>(src[src_pos + 1]) << 8) |
                            (static_cast<size_t>(src[src_pos + 2]) << 16) |
                            (static_cast<size_t>(src[src_pos + 3]) << 24);
            src_pos += 4;
            if (offset == 0 || offset > dst_pos || dst_pos + len > uncompressed_len) return false;
            for (size_t i = 0; i < len; ++i) {
                dst[dst_pos + i] = dst[dst_pos - offset + i];
            }
            dst_pos += len;
        }
    }

    return (dst_pos == uncompressed_len);
}

}

namespace rle {

bool decode_hybrid(const uint8_t* data, size_t data_len, int bit_width, size_t count, std::vector<uint32_t>& out) {
    if (count == 0) return true;
    if (bit_width < 0 || bit_width > 32) return false;
    constexpr size_t kMaxValues = 1u << 20;
    if (count > kMaxValues) return false;
    out.clear();
    out.reserve(count);

    if (bit_width == 0) {
        out.assign(count, 0);
        return true;
    }
    if (bit_width < 0 || bit_width > 32) return false;

    size_t pos = 0;
    const uint32_t mask = (bit_width == 32) ? 0xFFFFFFFFU : ((1U << bit_width) - 1U);

    while (pos < data_len && out.size() < count) {
        uint32_t header = 0;
        int shift = 0;
        while (pos < data_len) {
            uint8_t b = data[pos++];
            header |= static_cast<uint32_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
            if (shift > 28) return false;
        }

        if ((header & 1) == 0) {
            size_t run_count = static_cast<size_t>(header >> 1);
            size_t bytes_per_val = static_cast<size_t>((bit_width + 7) / 8);
            if (pos + bytes_per_val > data_len) return false;

            uint32_t val = 0;
            for (size_t i = 0; i < bytes_per_val; ++i) {
                val |= (static_cast<uint32_t>(data[pos++]) << (i * 8));
            }
            val &= mask;

            size_t to_add = std::min(run_count, count - out.size());
            out.insert(out.end(), to_add, val);
        } else {
            size_t num_groups = static_cast<size_t>(header >> 1);
            size_t num_values = num_groups * 8;
            size_t bytes_needed = (num_values * static_cast<size_t>(bit_width)) / 8;
            if (pos + bytes_needed > data_len) return false;

            uint64_t bit_buf = 0;
            int bits_in_buf = 0;
            size_t group_vals = 0;

            while (group_vals < num_values && out.size() < count) {
                while (bits_in_buf < bit_width && pos < data_len) {
                    bit_buf |= (static_cast<uint64_t>(data[pos++]) << bits_in_buf);
                    bits_in_buf += 8;
                }
                if (bits_in_buf < bit_width) break;

                uint32_t val = static_cast<uint32_t>(bit_buf & mask);
                bit_buf >>= bit_width;
                bits_in_buf -= bit_width;
                out.push_back(val);
                group_vals++;
            }
        }
    }

    return (out.size() >= count);
}

}

namespace thrift {

enum Type : uint8_t {
    STOP = 0x00,
    BOOLEAN_TRUE = 0x01,
    BOOLEAN_FALSE = 0x02,
    BYTE = 0x03,
    I16 = 0x04,
    I32 = 0x05,
    I64 = 0x06,
    DOUBLE = 0x07,
    BINARY = 0x08,
    LIST = 0x09,
    SET = 0x0A,
    MAP = 0x0B,
    STRUCT = 0x0C
};

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {}

    bool has_more() const { return pos_ < size_; }
    size_t pos() const { return pos_; }

    uint8_t read_u8() {
        if (pos_ >= size_) throw std::runtime_error("thrift: unexpected EOF");
        return data_[pos_++];
    }

    uint64_t read_varint() {
        uint64_t res = 0;
        int shift = 0;
        while (pos_ < size_) {
            uint8_t b = data_[pos_++];
            res |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return res;
            shift += 7;
            if (shift >= 64) throw std::runtime_error("thrift: varint overflow");
        }
        throw std::runtime_error("thrift: unexpected EOF in varint");
    }

    int64_t read_zigzag() {
        uint64_t u = read_varint();
        return static_cast<int64_t>((u >> 1) ^ (-(static_cast<int64_t>(u & 1))));
    }

    std::string read_binary() {
        uint64_t len = read_varint();
        if (len > (1ULL << 30)) throw std::runtime_error("thrift: binary length excessive");
        if (pos_ + len > size_) throw std::runtime_error("thrift: binary length out of bounds");
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += static_cast<size_t>(len);
        return s;
    }

    bool read_field_header(int16_t& last_id, uint8_t& type, int16_t& out_id) {
        if (pos_ >= size_) return false;
        uint8_t b = read_u8();
        type = b & 0x0F;
        if (type == STOP) return false;

        uint8_t delta = (b >> 4) & 0x0F;
        if (delta == 0) {
            out_id = static_cast<int16_t>(read_zigzag());
        } else {
            out_id = last_id + static_cast<int16_t>(delta);
        }
        last_id = out_id;
        return true;
    }

    void skip(uint8_t type, int depth = 0) {
        if (depth > 64) throw std::runtime_error("thrift: nesting depth exceeded");
        switch (type) {
            case BOOLEAN_TRUE:
            case BOOLEAN_FALSE:
                break;
            case BYTE:
                read_u8();
                break;
            case I16:
            case I32:
            case I64:
                read_zigzag();
                break;
            case DOUBLE:
                if (pos_ + 8 > size_) throw std::runtime_error("thrift: double EOF");
                pos_ += 8;
                break;
            case BINARY: {
                uint64_t len = read_varint();
                if (len > (1ULL << 30) || len > size_ - pos_)
                    throw std::runtime_error("thrift: binary skip EOF");
                pos_ += static_cast<size_t>(len);
                break;
            }
            case LIST:
            case SET: {
                uint8_t b = read_u8();
                uint8_t elem_type = b & 0x0F;
                uint32_t count = (b >> 4) & 0x0F;
                if (count == 0x0F) count = static_cast<uint32_t>(read_varint());
                if (count > 1000000 ||
                    (count > 1024 && (elem_type == BOOLEAN_TRUE || elem_type == BOOLEAN_FALSE)))
                    throw std::runtime_error("thrift: excessive container element count");
                for (uint32_t i = 0; i < count; ++i) skip(elem_type, depth + 1);
                break;
            }
            case MAP: {
                uint32_t count = static_cast<uint32_t>(read_varint());
                if (count > 1000000)
                    throw std::runtime_error("thrift: excessive map element count");
                if (count > 0) {
                    uint8_t b = read_u8();
                    uint8_t k_type = (b >> 4) & 0x0F;
                    uint8_t v_type = b & 0x0F;
                    for (uint32_t i = 0; i < count; ++i) {
                        skip(k_type, depth + 1);
                        skip(v_type, depth + 1);
                    }
                }
                break;
            }
            case STRUCT: {
                int16_t last_id = 0;
                uint8_t f_type = 0;
                int16_t f_id = 0;
                while (read_field_header(last_id, f_type, f_id)) {
                    skip(f_type, depth + 1);
                }
                break;
            }
            default:
                throw std::runtime_error("thrift: unknown type to skip: " + std::to_string(type));
        }
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

}

struct ColumnChunkMeta {
    std::string column_name;
    ParquetPhysicalType type = ParquetPhysicalType::UNKNOWN;
    int codec = 0;
    int64_t num_values = 0;
    int64_t total_compressed_size = 0;
    int64_t total_uncompressed_size = 0;
    int64_t data_page_offset = 0;
    int64_t dictionary_page_offset = 0;
    int type_length = 0;
};

struct RowGroupMeta {
    int64_t num_rows = 0;
    std::vector<ColumnChunkMeta> columns;
};

struct FileMeta {
    int64_t num_rows = 0;
    std::vector<ParquetColumnInfo> schema;
    std::vector<RowGroupMeta> row_groups;
};

static void parse_schema_element(thrift::Reader& r, ParquetColumnInfo& elem) {
    int16_t last_id = 0, f_id = 0;
    uint8_t f_type = 0;
    while (r.read_field_header(last_id, f_type, f_id)) {
        if (f_id == 1 && f_type == thrift::I32) {
            elem.physical_type = static_cast<ParquetPhysicalType>(r.read_zigzag());
        } else if (f_id == 2 && f_type == thrift::I32) {
            elem.type_length = static_cast<int>(r.read_zigzag());
        } else if (f_id == 3 && f_type == thrift::I32) {
            elem.repetition = static_cast<ParquetRepetition>(r.read_zigzag());
        } else if (f_id == 4 && f_type == thrift::BINARY) {
            elem.name = r.read_binary();
        } else if (f_id == 6 && f_type == thrift::I32) {
            elem.converted_type = static_cast<int>(r.read_zigzag());
        } else {
            r.skip(f_type);
        }
    }
}

static void parse_column_metadata(thrift::Reader& r, ColumnChunkMeta& col) {
    int16_t last_id = 0, f_id = 0;
    uint8_t f_type = 0;
    while (r.read_field_header(last_id, f_type, f_id)) {
        if (f_id == 1 && f_type == thrift::I32) {
            col.type = static_cast<ParquetPhysicalType>(r.read_zigzag());
        } else if (f_id == 3 && (f_type == thrift::LIST || f_type == thrift::SET)) {
            uint8_t b = r.read_u8();
            uint32_t count = (b >> 4) & 0x0F;
            if (count == 0x0F) count = static_cast<uint32_t>(r.read_varint());
            for (uint32_t i = 0; i < count; ++i) {
                std::string part = r.read_binary();
                if (col.column_name.empty()) col.column_name = part;
                else col.column_name += "." + part;
            }
        } else if (f_id == 4 && f_type == thrift::I32) {
            col.codec = static_cast<int>(r.read_zigzag());
        } else if (f_id == 5 && f_type == thrift::I64) {
            col.num_values = r.read_zigzag();
        } else if (f_id == 6 && f_type == thrift::I64) {
            col.total_uncompressed_size = r.read_zigzag();
        } else if (f_id == 7 && f_type == thrift::I64) {
            col.total_compressed_size = r.read_zigzag();
        } else if (f_id == 9 && f_type == thrift::I64) {
            col.data_page_offset = r.read_zigzag();
        } else if (f_id == 11 && f_type == thrift::I64) {
            col.dictionary_page_offset = r.read_zigzag();
        } else {
            r.skip(f_type);
        }
    }
}

static void parse_column_chunk(thrift::Reader& r, ColumnChunkMeta& col) {
    int16_t last_id = 0, f_id = 0;
    uint8_t f_type = 0;
    while (r.read_field_header(last_id, f_type, f_id)) {
        if (f_id == 3 && f_type == thrift::STRUCT) {
            parse_column_metadata(r, col);
        } else {
            r.skip(f_type);
        }
    }
}

static void parse_row_group(thrift::Reader& r, RowGroupMeta& rg) {
    int16_t last_id = 0, f_id = 0;
    uint8_t f_type = 0;
    while (r.read_field_header(last_id, f_type, f_id)) {
        if (f_id == 1 && (f_type == thrift::LIST || f_type == thrift::SET)) {
            uint8_t b = r.read_u8();
            uint32_t count = (b >> 4) & 0x0F;
            if (count == 0x0F) count = static_cast<uint32_t>(r.read_varint());
            if (count > 100000)
                throw std::runtime_error("too many columns in row group");
            rg.columns.resize(count);
            for (uint32_t i = 0; i < count; ++i) {
                parse_column_chunk(r, rg.columns[i]);
            }
        } else if (f_id == 3 && f_type == thrift::I64) {
            rg.num_rows = r.read_zigzag();
        } else {
            r.skip(f_type);
        }
    }
}

static bool parse_file_metadata(const uint8_t* data, size_t size, FileMeta& meta, std::string* err) {
    try {
        thrift::Reader r(data, size);
        int16_t last_id = 0, f_id = 0;
        uint8_t f_type = 0;

        while (r.read_field_header(last_id, f_type, f_id)) {
            if (f_id == 2 && (f_type == thrift::LIST || f_type == thrift::SET)) {
                uint8_t b = r.read_u8();
                uint32_t count = (b >> 4) & 0x0F;
                if (count == 0x0F) count = static_cast<uint32_t>(r.read_varint());
                if (count > 100000) throw std::runtime_error("too many schema elements");
                for (uint32_t i = 0; i < count; ++i) {
                    ParquetColumnInfo elem;
                    parse_schema_element(r, elem);
                    if (i > 0) {
                        meta.schema.push_back(elem);
                    }
                }
            } else if (f_id == 3 && f_type == thrift::I64) {
                meta.num_rows = r.read_zigzag();
            } else if (f_id == 4 && (f_type == thrift::LIST || f_type == thrift::SET)) {
                uint8_t b = r.read_u8();
                uint32_t count = (b >> 4) & 0x0F;
                if (count == 0x0F) count = static_cast<uint32_t>(r.read_varint());
                if (count > 100000) throw std::runtime_error("too many row groups");
                meta.row_groups.resize(count);
                for (uint32_t i = 0; i < count; ++i) {
                    parse_row_group(r, meta.row_groups[i]);
                }
            } else {
                r.skip(f_type);
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = "Parquet metadata parse error: " + std::string(e.what());
        return false;
    }
}

struct PageHeaderInfo {
    int type = 0;
    int32_t uncompressed_page_size = 0;
    int32_t compressed_page_size = 0;

    int32_t num_values = 0;
    int32_t num_nulls = 0;
    int32_t num_rows = 0;
    int encoding = 0;
    int def_level_encoding = 0;
    int32_t def_levels_byte_len = 0;
    int32_t rep_levels_byte_len = 0;
    bool is_compressed = true;
};

static bool parse_page_header(thrift::Reader& r, PageHeaderInfo& hdr, std::string* err) {
    try {
        int16_t last_id = 0, f_id = 0;
        uint8_t f_type = 0;

        while (r.read_field_header(last_id, f_type, f_id)) {
            if (f_id == 1 && f_type == thrift::I32) {
                hdr.type = static_cast<int>(r.read_zigzag());
            } else if (f_id == 2 && f_type == thrift::I32) {
                hdr.uncompressed_page_size = static_cast<int32_t>(r.read_zigzag());
            } else if (f_id == 3 && f_type == thrift::I32) {
                hdr.compressed_page_size = static_cast<int32_t>(r.read_zigzag());
            } else if (f_id == 5 && f_type == thrift::STRUCT) {
                int16_t s_last = 0, s_id = 0;
                uint8_t s_type = 0;
                while (r.read_field_header(s_last, s_type, s_id)) {
                    if (s_id == 1 && s_type == thrift::I32) {
                        hdr.num_values = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 2 && s_type == thrift::I32) {
                        hdr.encoding = static_cast<int>(r.read_zigzag());
                    } else if (s_id == 3 && s_type == thrift::I32) {
                        hdr.def_level_encoding = static_cast<int>(r.read_zigzag());
                    } else {
                        r.skip(s_type);
                    }
                }
            } else if (f_id == 7 && f_type == thrift::STRUCT) {
                int16_t s_last = 0, s_id = 0;
                uint8_t s_type = 0;
                while (r.read_field_header(s_last, s_type, s_id)) {
                    if (s_id == 1 && s_type == thrift::I32) {
                        hdr.num_values = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 2 && s_type == thrift::I32) {
                        hdr.encoding = static_cast<int>(r.read_zigzag());
                    } else {
                        r.skip(s_type);
                    }
                }
            } else if (f_id == 8 && f_type == thrift::STRUCT) {
                int16_t s_last = 0, s_id = 0;
                uint8_t s_type = 0;
                while (r.read_field_header(s_last, s_type, s_id)) {
                    if (s_id == 1 && s_type == thrift::I32) {
                        hdr.num_values = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 2 && s_type == thrift::I32) {
                        hdr.num_nulls = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 3 && s_type == thrift::I32) {
                        hdr.num_rows = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 4 && s_type == thrift::I32) {
                        hdr.encoding = static_cast<int>(r.read_zigzag());
                    } else if (s_id == 5 && s_type == thrift::I32) {
                        hdr.def_levels_byte_len = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 6 && s_type == thrift::I32) {
                        hdr.rep_levels_byte_len = static_cast<int32_t>(r.read_zigzag());
                    } else if (s_id == 7) {
                        hdr.is_compressed = (s_type == thrift::BOOLEAN_TRUE);
                    } else {
                        r.skip(s_type);
                    }
                }
            } else {
                r.skip(f_type);
            }
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = "Page header parse error: " + std::string(e.what());
        return false;
    }
}

static bool decode_plain_cell(const uint8_t* data, size_t size, size_t& pos,
                              ParquetPhysicalType type, int type_len, ParquetCell& cell) {
    cell.is_null = false;
    switch (type) {
        case ParquetPhysicalType::BOOLEAN: {
            const size_t byte = pos / 8;
            if (byte >= size) return false;
            cell.bool_val = ((data[byte] >> (pos % 8)) & 1u) != 0;
            ++pos;
            cell.string_val = cell.bool_val ? "true" : "false";
            return true;
        }
        case ParquetPhysicalType::INT32: {
            if (pos + 4 > size) return false;
            int32_t v;
            std::memcpy(&v, data + pos, 4);
            pos += 4;
            cell.int_val = v;
            cell.string_val = std::to_string(v);
            return true;
        }
        case ParquetPhysicalType::INT64: {
            if (pos + 8 > size) return false;
            int64_t v;
            std::memcpy(&v, data + pos, 8);
            pos += 8;
            cell.int_val = v;
            cell.string_val = std::to_string(v);
            return true;
        }
        case ParquetPhysicalType::FLOAT: {
            if (pos + 4 > size) return false;
            float v;
            std::memcpy(&v, data + pos, 4);
            pos += 4;
            cell.double_val = v;
            char b[32];
            std::snprintf(b, sizeof(b), "%.9g", v);
            cell.string_val = b;
            return true;
        }
        case ParquetPhysicalType::DOUBLE: {
            if (pos + 8 > size) return false;
            double v;
            std::memcpy(&v, data + pos, 8);
            pos += 8;
            cell.double_val = v;
            char b[32];
            std::snprintf(b, sizeof(b), "%.17g", v);
            cell.string_val = b;
            return true;
        }
        case ParquetPhysicalType::BYTE_ARRAY: {
            if (pos + 4 > size) return false;
            uint32_t len;
            std::memcpy(&len, data + pos, 4);
            pos += 4;
            if (pos + len > size) return false;
            cell.string_val.assign(reinterpret_cast<const char*>(data + pos), len);
            pos += len;
            return true;
        }
        case ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY: {
            if (type_len <= 0 || pos + static_cast<size_t>(type_len) > size) return false;
            cell.string_val.assign(reinterpret_cast<const char*>(data + pos), type_len);
            pos += static_cast<size_t>(type_len);
            return true;
        }
        default:
            return false;
    }
}

bool ParquetReader::is_parquet(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4];
    if (!f.read(magic, 4) || std::memcmp(magic, "PAR1", 4) != 0) return false;

    f.seekg(-4, std::ios::end);
    if (!f.read(magic, 4) || std::memcmp(magic, "PAR1", 4) != 0) return false;

    return true;
}

bool ParquetReader::inspect(const std::string& path, ParquetSchema& schema, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        if (err) *err = "Cannot open file: " + path;
        return false;
    }

    size_t file_size = static_cast<size_t>(f.tellg());
    if (file_size < 12) {
        if (err) *err = "File too small to be Parquet: " + path;
        return false;
    }

    f.seekg(static_cast<std::streamoff>(file_size - 8));
    uint32_t footer_len = 0;
    f.read(reinterpret_cast<char*>(&footer_len), 4);

    if (footer_len == 0 || static_cast<uint64_t>(footer_len) + 8 > file_size) {
        if (err) *err = "Corrupt Parquet footer length in " + path;
        return false;
    }

    std::vector<uint8_t> footer(footer_len);
    f.seekg(static_cast<std::streamoff>(file_size - 8 - footer_len));
    if (!f.read(reinterpret_cast<char*>(footer.data()), static_cast<std::streamsize>(footer_len))) {
        if (err) *err = "Truncated Parquet footer in " + path;
        return false;
    }

    FileMeta meta;
    if (!parse_file_metadata(footer.data(), footer.size(), meta, err)) {
        return false;
    }

    schema.columns = meta.schema;
    schema.num_rows = meta.num_rows;
    schema.num_row_groups = static_cast<int>(meta.row_groups.size());
    return true;
}

bool ParquetReader::read_rows(const std::string& path,
                             const std::vector<std::string>& selected_columns,
                             ParquetRowCallback callback,
                             size_t max_rows,
                             std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        if (err) *err = "Cannot open file: " + path;
        return false;
    }

    size_t file_size = static_cast<size_t>(f.tellg());
    if (file_size < 12) {
        if (err) *err = "File too small: " + path;
        return false;
    }

    f.seekg(static_cast<std::streamoff>(file_size - 8));
    uint32_t footer_len = 0;
    f.read(reinterpret_cast<char*>(&footer_len), 4);

    if (footer_len == 0 || static_cast<uint64_t>(footer_len) + 8 > file_size) {
        if (err) *err = "Corrupt footer length in " + path;
        return false;
    }

    std::vector<uint8_t> footer(footer_len);
    f.seekg(static_cast<std::streamoff>(file_size - 8 - footer_len));
    if (!f.read(reinterpret_cast<char*>(footer.data()), static_cast<std::streamsize>(footer_len))) {
        if (err) *err = "Truncated footer in " + path;
        return false;
    }

    FileMeta meta;
    if (!parse_file_metadata(footer.data(), footer.size(), meta, err)) return false;

    std::unordered_map<std::string, ParquetColumnInfo> schema_map;
    for (const auto& col : meta.schema) {
        schema_map[col.name] = col;
    }

    size_t rows_delivered = 0;

    for (const auto& rg : meta.row_groups) {
        if (max_rows > 0 && rows_delivered >= max_rows) break;

        std::unordered_map<std::string, std::vector<ParquetCell>> group_data;

        for (const auto& col : rg.columns) {
            bool requested = selected_columns.empty();
            if (!requested) {
                for (const auto& req : selected_columns) {
                    if (req == col.column_name) { requested = true; break; }
                }
            }
            if (!requested) continue;

            if (col.codec != 0 && col.codec != 1) {
                const char* codec_name = (col.codec == 2) ? "GZIP" :
                                         (col.codec == 3) ? "LZO" :
                                         (col.codec == 4) ? "BROTLI" :
                                         (col.codec == 5) ? "LZ4" :
                                         (col.codec == 6) ? "ZSTD" : "UNKNOWN";
                if (err) {
                    *err = "Unsupported Parquet compression codec: " + std::string(codec_name) +
                           " (codec id " + std::to_string(col.codec) + ") in column '" +
                           col.column_name + "'. Currently supported: UNCOMPRESSED, SNAPPY.";
                }
                return false;
            }

            ParquetColumnInfo col_info;
            if (schema_map.count(col.column_name)) col_info = schema_map[col.column_name];

            int64_t offset = (col.dictionary_page_offset > 0) ? col.dictionary_page_offset : col.data_page_offset;
            if (offset < 4 || offset >= static_cast<int64_t>(file_size)) {
                if (err) *err = "Corrupt column page offset in " + col.column_name;
                return false;
            }
            f.seekg(static_cast<std::streampos>(offset));

            std::vector<ParquetCell> dictionary;
            std::vector<ParquetCell> column_cells;
            if (rg.num_rows < 0 || static_cast<uint64_t>(rg.num_rows) > file_size) {
                if (err) *err = "Invalid row count in row group of " + col.column_name;
                return false;
            }
            if (rg.num_rows > 1000000 || static_cast<uint64_t>(rg.num_rows) > static_cast<uint64_t>(file_size)) return false;
            column_cells.reserve(static_cast<size_t>(rg.num_rows));
            int64_t values_read = 0;

            while (values_read < col.num_values && f.tellg() < static_cast<std::streampos>(offset + col.total_compressed_size)) {
                std::vector<uint8_t> hdr_buf(1024);
                auto hdr_start = f.tellg();
                f.read(reinterpret_cast<char*>(hdr_buf.data()), hdr_buf.size());
                std::streamsize actual_read = f.gcount();
                if (actual_read <= 0) break;

                thrift::Reader tr(hdr_buf.data(), static_cast<size_t>(actual_read));
                PageHeaderInfo phdr;
                if (!parse_page_header(tr, phdr, err)) return false;

                f.seekg(hdr_start + static_cast<std::streampos>(tr.pos()));

                if (phdr.compressed_page_size <= 0 || phdr.compressed_page_size > (1 << 30)) {
                    if (err) *err = "Invalid page size in column: " + col.column_name;
                    return false;
                }
                if (phdr.num_values < 0 || static_cast<uint64_t>(phdr.num_values) > file_size) {
                    if (err) *err = "Invalid value count in column: " + col.column_name;
                    return false;
                }

                if (col.data_page_offset + static_cast<uint64_t>(phdr.compressed_page_size) > file_size) {
                    if (err) *err = "Page payload exceeds file in column: " + col.column_name;
                    return false;
                }
                std::vector<uint8_t> page_raw(static_cast<size_t>(phdr.compressed_page_size));
                if (!f.read(reinterpret_cast<char*>(page_raw.data()), phdr.compressed_page_size)) {
                    if (err) *err = "Truncated page payload in column: " + col.column_name;
                    return false;
                }

                if (phdr.type == 2) {
                    std::vector<uint8_t> decomp;
                    const uint8_t* p_data = page_raw.data();
                    size_t p_size = page_raw.size();
                    if (col.codec == 1) {
                        if (!snappy::decompress(p_data, p_size, decomp)) {
                            if (err) *err = "Snappy decompression failed for dictionary page in " + col.column_name;
                            return false;
                        }
                        p_data = decomp.data();
                        p_size = decomp.size();
                    }

                    if (phdr.num_values < 0 || static_cast<uint64_t>(phdr.num_values) > p_size) {
                        if (err) *err = "Invalid dictionary value count in " + col.column_name;
                        return false;
                    }
                    size_t p_pos = 0;
                    if (phdr.num_values > 1000000 || phdr.num_values > static_cast<i32>(p_size) * 8)
                        return false;
                    if (phdr.num_values > 1000000 || phdr.num_values > static_cast<i32>(p_size) * 8) {
                        if (err) *err = "Implausible dictionary size in " + col.column_name;
                        return false;
                    }
                    dictionary.resize(static_cast<size_t>(phdr.num_values));
                    for (int i = 0; i < phdr.num_values; ++i) {
                        if (!decode_plain_cell(p_data, p_size, p_pos, col.type, col_info.type_length, dictionary[i])) {
                            if (err) *err = "Failed decoding dictionary entry in " + col.column_name;
                            return false;
                        }
                    }
                } else if (phdr.type == 0 || phdr.type == 3) {
                    std::vector<uint8_t> def_levels_raw;
                    std::vector<uint8_t> values_raw;

                    if (phdr.type == 3) {
                        if (phdr.rep_levels_byte_len < 0 || phdr.def_levels_byte_len < 0) {
                            if (err) *err = "V2 negative levels length in " + col.column_name;
                            return false;
                        }
                        size_t rep_len = static_cast<size_t>(phdr.rep_levels_byte_len);
                        size_t def_len = static_cast<size_t>(phdr.def_levels_byte_len);
                        if (rep_len > page_raw.size() || def_len > page_raw.size() - rep_len) {
                            if (err) *err = "V2 levels length exceeds page size in " + col.column_name;
                            return false;
                        }

                        if (def_len > 0) {
                            def_levels_raw.assign(page_raw.begin() + rep_len, page_raw.begin() + rep_len + def_len);
                        }

                        const uint8_t* v_src = page_raw.data() + rep_len + def_len;
                        size_t v_src_len = page_raw.size() - rep_len - def_len;

                        if (phdr.is_compressed && col.codec == 1) {
                            if (!snappy::decompress(v_src, v_src_len, values_raw)) {
                                if (err) *err = "Snappy decompression failed for V2 data in " + col.column_name;
                                return false;
                            }
                        } else {
                            values_raw.assign(v_src, v_src + v_src_len);
                        }
                    } else {
                        std::vector<uint8_t> decomp;
                        const uint8_t* p_data = page_raw.data();
                        size_t p_size = page_raw.size();
                        if (col.codec == 1) {
                            if (!snappy::decompress(p_data, p_size, decomp)) {
                                if (err) *err = "Snappy decompression failed for V1 page in " + col.column_name;
                                return false;
                            }
                            p_data = decomp.data();
                            p_size = decomp.size();
                        }

                        size_t p_pos = 0;
                        if (col_info.repetition == ParquetRepetition::OPTIONAL) {
                            if (p_pos + 4 > p_size) {
                                if (err) *err = "V1 def level length truncated in " + col.column_name;
                                return false;
                            }
                            uint32_t def_len = 0;
                            std::memcpy(&def_len, p_data + p_pos, 4);
                            p_pos += 4;
                            if (p_pos + def_len > p_size) {
                                if (err) *err = "V1 def levels buffer out of bounds in " + col.column_name;
                                return false;
                            }
                            def_levels_raw.assign(p_data + p_pos, p_data + p_pos + def_len);
                            p_pos += def_len;
                        }
                        values_raw.assign(p_data + p_pos, p_data + p_size);
                    }

                    std::vector<uint32_t> def_levels;
                    if (col_info.repetition == ParquetRepetition::OPTIONAL && !def_levels_raw.empty()) {
                        if (!rle::decode_hybrid(def_levels_raw.data(), def_levels_raw.size(), 1,
                                                static_cast<size_t>(phdr.num_values), def_levels)) {
                            if (err) *err = "Failed decoding RLE definition levels in " + col.column_name;
                            return false;
                        }
                    }

                    size_t v_pos = 0;
                    if (phdr.encoding == 7 || phdr.encoding == 1) {
                        if (values_raw.empty() || dictionary.empty()) {
                            if (err) *err = "Missing dictionary or empty values for dictionary encoding in " + col.column_name;
                            return false;
                        }
                        uint8_t bit_width = values_raw[0];
                        std::vector<uint32_t> indices;
                        size_t non_null_count = 0;
                        if (!def_levels.empty()) {
                            for (uint32_t dl : def_levels) if (dl == 1) non_null_count++;
                        } else {
                            non_null_count = static_cast<size_t>(phdr.num_values);
                        }

                        if (!rle::decode_hybrid(values_raw.data() + 1, values_raw.size() - 1, bit_width, non_null_count, indices)) {
                            if (err) *err = "Failed decoding hybrid RLE dictionary indices in " + col.column_name;
                            return false;
                        }

                        size_t idx_cur = 0;
                        for (int i = 0; i < phdr.num_values; ++i) {
                            if (!def_levels.empty() && def_levels[static_cast<size_t>(i)] == 0) {
                                ParquetCell null_cell;
                                null_cell.is_null = true;
                                column_cells.push_back(null_cell);
                            } else {
                                if (idx_cur >= indices.size()) {
                                    if (err) *err = "Dictionary index stream underflow in " + col.column_name;
                                    return false;
                                }
                                uint32_t d_idx = indices[idx_cur++];
                                if (d_idx >= dictionary.size()) {
                                    if (err) *err = "Invalid dictionary index " + std::to_string(d_idx) + " (dict size " + std::to_string(dictionary.size()) + ") in " + col.column_name;
                                    return false;
                                }
                                column_cells.push_back(dictionary[d_idx]);
                            }
                        }
                    } else if (phdr.encoding == 0) {
                        for (int i = 0; i < phdr.num_values; ++i) {
                            if (!def_levels.empty() && def_levels[static_cast<size_t>(i)] == 0) {
                                ParquetCell null_cell;
                                null_cell.is_null = true;
                                column_cells.push_back(null_cell);
                            } else {
                                ParquetCell cell;
                                if (!decode_plain_cell(values_raw.data(), values_raw.size(), v_pos, col.type, col_info.type_length, cell)) {
                                    if (err) *err = "Failed decoding PLAIN cell in " + col.column_name;
                                    return false;
                                }
                                column_cells.push_back(cell);
                            }
                        }
                    } else {
                        if (err) log_warn("parquet: skipping unsupported encoding " +
                                          std::to_string(phdr.encoding));
                        for (int i = 0; i < phdr.num_values; ++i) {
                            ParquetCell nc;
                            nc.is_null = true;
                            column_cells.push_back(nc);
                        }
                    }

                    values_read += phdr.num_values;
                }
            }

            group_data[col.column_name] = std::move(column_cells);
        }

        size_t rg_rows = static_cast<size_t>(rg.num_rows);
        for (size_t r = 0; r < rg_rows; ++r) {
            if (max_rows > 0 && rows_delivered >= max_rows) break;

            ParquetRowMap row;
            for (auto& kv : group_data) {
                if (r < kv.second.size()) {
                    row[kv.first] = std::move(kv.second[r]);
                } else {
                    ParquetCell null_cell;
                    null_cell.is_null = true;
                    row[kv.first] = null_cell;
                }
            }

            if (!callback(row)) return true;
            rows_delivered++;
        }
    }

    return true;
}

}
