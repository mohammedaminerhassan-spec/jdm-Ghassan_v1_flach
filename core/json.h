#pragma once

#include "core/common.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gai {

// Minimal JSON reader/writer.
//
// The engine needs JSON in three places: the HTTP API, the training
// recipes and the safetensors header. A dependency free parser keeps the
// build self contained; this is deliberately small (no comments, no
// trailing commas, no NaN) and strict about what it accepts.

class Json {
public:
    enum class Type : u8 { Null, Bool, Number, String, Array, Object };

    Json() = default;
    static Json null();
    static Json boolean(bool v);
    static Json number(double v);
    static Json integer(i64 v);
    static Json string(std::string v);
    static Json array();
    static Json object();

    Type type() const { return type_; }
    bool is_null()   const { return type_ == Type::Null; }
    bool is_bool()   const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    // True when the value was parsed as (or created as) an exact int64.
    bool is_integer() const { return type_ == Type::Number && is_int_; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array()  const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool        as_bool(bool def = false) const;
    double      as_number(double def = 0.0) const;
    i64         as_int(i64 def = 0) const;
    std::string as_string(const std::string& def = "") const;

    // Object access
    bool        has(const std::string& key) const;
    const Json& at(const std::string& key) const;   // returns a null Json when absent
    void        set(const std::string& key, const Json& v);
    const std::map<std::string, Json>& items() const { return object_; }
    std::map<std::string, Json>& items() { return object_; }

    // Array access
    void                push(const Json& v);
    size_t              size() const { return array_.size(); }
    const Json&         operator[](size_t i) const;
    const std::vector<Json>& elements() const { return array_; }

    std::string dump() const;      // compact
    std::string dump_pretty() const;

    // Parses `text`. Returns false and fills `error` on failure.
    static bool parse(const std::string& text, Json& out, std::string* error = nullptr);

    // Escapes a string for a JSON string literal.
    static std::string escape(const std::string& s);

private:
    Type                            type_ = Type::Null;
    bool                            bool_ = false;
    double                          num_ = 0.0;
    bool                            is_int_ = false;
    i64                             int_ = 0;
    std::string                     str_;
    std::vector<Json>               array_;
    std::map<std::string, Json>     object_;
};

}  // namespace gai
