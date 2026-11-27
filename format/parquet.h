#pragma once

#include "core/common.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cstdint>

namespace gai {

enum class ParquetPhysicalType : int {
    BOOLEAN = 0,
    INT32 = 1,
    INT64 = 2,
    INT96 = 3,
    FLOAT = 4,
    DOUBLE = 5,
    BYTE_ARRAY = 6,
    FIXED_LEN_BYTE_ARRAY = 7,
    UNKNOWN = -1
};

enum class ParquetRepetition : int {
    REQUIRED = 0,
    OPTIONAL = 1,
    REPEATED = 2
};

struct ParquetColumnInfo {
    std::string name;
    ParquetPhysicalType physical_type = ParquetPhysicalType::UNKNOWN;
    ParquetRepetition repetition = ParquetRepetition::REQUIRED;
    int type_length = 0;
    int converted_type = -1;
};

struct ParquetSchema {
    std::vector<ParquetColumnInfo> columns;
    int64_t num_rows = 0;
    int num_row_groups = 0;
};

struct ParquetCell {
    bool is_null = false;
    std::string string_val;
    int64_t int_val = 0;
    double double_val = 0.0;
    bool bool_val = false;

    const std::string& as_string() const { return string_val; }
};

using ParquetRowMap = std::unordered_map<std::string, ParquetCell>;

using ParquetRowCallback = std::function<bool(const ParquetRowMap& row)>;

class ParquetReader {
public:
    ParquetReader() = default;
    ~ParquetReader() = default;

    static bool is_parquet(const std::string& path);

    static bool inspect(const std::string& path, ParquetSchema& schema, std::string* err = nullptr);

    static bool read_rows(const std::string& path,
                          const std::vector<std::string>& selected_columns,
                          ParquetRowCallback callback,
                          size_t max_rows = 0,
                          std::string* err = nullptr);
};

namespace snappy {
    bool decompress(const uint8_t* src, size_t src_len, std::vector<uint8_t>& dst);
}

namespace rle {
    bool decode_hybrid(const uint8_t* data, size_t data_len, int bit_width, size_t count, std::vector<uint32_t>& out);
}

}
