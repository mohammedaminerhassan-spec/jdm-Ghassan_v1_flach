#pragma once

#include "core/common.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gai {

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
    bool is_integer() const { return type_ == Type::Number && is_int_; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array()  const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool        as_bool(bool def = false) const;
    double      as_number(double def = 0.0) const;
    i64         as_int(i64 def = 0) const;
    std::string as_string(const std::string& def = "") const;

    bool        has(const std::string& key) const;
    const Json& at(const std::string& key) const;
    void        set(const std::string& key, const Json& v);
    const std::map<std::string, Json>& items() const { return object_; }
    std::map<std::string, Json>& items() { return object_; }

    void                push(const Json& v);
    size_t              size() const { return array_.size(); }
    const Json&         operator[](size_t i) const;
    const std::vector<Json>& elements() const { return array_; }

    std::string dump() const;
    std::string dump_pretty() const;

    static bool parse(const std::string& text, Json& out, std::string* error = nullptr);

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

}
