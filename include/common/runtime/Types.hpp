#ifndef H_Types
#define H_Types
//---------------------------------------------------------------------------
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>    // Required for std::string wrappers
#include <algorithm> // Required for std::min
//---------------------------------------------------------------------------
// HyPer
// (c) Thomas Neumann 2010 - modernized 2026
//---------------------------------------------------------------------------
namespace types {

typedef uint64_t Tid;

template <unsigned size> struct LengthSwitch {};
template <> struct LengthSwitch<1> { typedef uint8_t type; };
template <> struct LengthSwitch<2> { typedef uint16_t type; };
template <> struct LengthSwitch<4> { typedef uint32_t type; };

template <unsigned maxLen> struct LengthIndicator {
   typedef typename LengthSwitch<
       (maxLen < 256) ? 1 : (maxLen < 65536) ? 2 : 4>::type type;
};

// --- Atomic Helpers (Inlined)
template <class T> inline void atomicMax(T& x, const T& y) {
   while ((y.value > x.value) && !__sync_bool_compare_and_swap(&x.value, x.value, y.value));
}

template <class T> inline void atomicMin(T& x, const T& y) {
   while ((y.value < x.value) && !__sync_bool_compare_and_swap(&x.value, x.value, y.value));
}

template <class T> inline void atomicAdd(T& x, const T& y) {
   __sync_fetch_and_add(&x.value, y.value);
}

// --- Integer Class
class Integer {
 public:
   int32_t value;

   inline Integer() : value(0) {}
   inline Integer(int32_t value) : value(value) {}

   inline uint64_t hash() const __attribute__((always_inline));
   
   inline bool operator==(const Integer& n) const { return value == n.value; }
   inline bool operator!=(const Integer& n) const { return value != n.value; }
   inline bool operator<(const Integer& n) const { return value < n.value; }
   inline bool operator<=(const Integer& n) const { return value <= n.value; }
   inline bool operator>(const Integer& n) const { return value > n.value; }
   inline bool operator>=(const Integer& n) const { return value >= n.value; }

   inline Integer operator+(const Integer& n) const { return Integer(value + n.value); }
   inline Integer& operator+=(const Integer& n) { value += n.value; return *this; }
   inline Integer operator-(const Integer& n) const { return Integer(value - n.value); }
   inline Integer operator*(const Integer& n) const { return Integer(value * n.value); }

   static Integer castString(const char* str, uint32_t strLen);
   inline static Integer castString(std::string s) { return castString(s.data(), s.size()); }
};

inline Integer modulo(Integer x, int32_t y) { return Integer(x.value % y); }

// --- Varchar Template
template <unsigned maxLen> class Varchar {
 public:
   typename LengthIndicator<maxLen>::type len;
   char value[maxLen];

   inline unsigned length() const { return len; }
   inline uint64_t hash() const __attribute__((always_inline));
   
   inline char* begin() { return value; }
   inline char* end() { return value + length(); }
   inline const char* begin() const { return value; }
   inline const char* end() const { return value + length(); }

   inline bool operator==(const char* other) const { return strncmp(value, other, len) == 0; }
   inline bool operator==(const Varchar& other) const { return (len == other.len) && (memcmp(value, other.value, len) == 0); }
   inline bool operator<(const Varchar& other) const __attribute__((always_inline));

   static Varchar build(const char* value) {
      Varchar result;
      strncpy(result.value, value, maxLen);
      result.len = strnlen(value, maxLen);
      return result;
   }
   static Varchar<maxLen> castString(const char* str, uint32_t strLen) {
      assert(strLen <= maxLen);
      Varchar<maxLen> result;
      result.len = strLen;
      memcpy(result.value, str, strLen);
      return result;
   };
   inline static Varchar<maxLen> castString(std::string s) { return castString(s.data(), s.size()); }
};

// --- Char Template
template <unsigned maxLen> class Char {
 public:
   typename LengthIndicator<maxLen>::type len;
   char value[maxLen];

   inline unsigned length() const { return len; }
   inline uint64_t hash() const __attribute__((always_inline));
   
   inline char* begin() { return value; }
   inline char* end() { return value + length(); }
   inline const char* begin() const { return value; }
   inline const char* end() const { return value + length(); }

   inline bool operator==(const char* other) const {
      return (other[0] == value[0]) && (len == strlen(other)) && (strncmp(value, other, len) == 0);
   }
   inline bool operator!=(const char* other) const {
      return (len != strlen(other)) || (strncmp(value, other, len) != 0);
   }
   inline bool operator==(const Char& other) const {
      return (len == other.len) && (memcmp(value, other.value, len) == 0);
   }
   
   inline bool operator<(const Char& other) const __attribute__((always_inline));
   inline bool operator>(const Char& other) const __attribute__((always_inline));
   inline bool operator<=(const Char& other) const __attribute__((always_inline));
   inline bool operator>=(const Char& other) const __attribute__((always_inline));

   static Char build(const char* value) {
      Char result;
      memcpy(result.value, value, maxLen);
      result.len = strnlen(result.value, maxLen);
      return result;
   }
   static Char<maxLen> castString(const char* str, uint32_t strLen) {
      while ((*str) == ' ') { str++; strLen--; }
      assert(strLen <= maxLen);
      Char<maxLen> result;
      result.len = strLen;
      memcpy(result.value, str, strLen);
      return result;
   }
   inline static Char<maxLen> castString(std::string s) { return castString(s.data(), s.size()); }
};

// Specialization for Char<1>
template <> class Char<1> {
 public:
   char value;
   inline uint64_t hash() const __attribute__((always_inline));
   inline unsigned length() const { return value != ' '; }
   inline char* begin() { return &value; }
   inline char* end() { return &value + length(); }
   inline const char* begin() const { return &value; }
   inline const char* end() const { return &value + length(); }

   inline bool operator==(const char* other) const { return (value == other[0]) && (strlen(other) == 1); }
   inline bool operator==(const Char& other) const { return value == other.value; }
   inline bool operator!=(const Char& other) const { return value != other.value; }
   inline bool operator<(const Char& other) const { return value < other.value; }
   inline bool operator<=(const Char& other) const { return value <= other.value; }
   inline bool operator>(const Char& other) const { return value > other.value; }
   inline bool operator>=(const Char& other) const { return value >= other.value; }

   static Char build(const char* value) { Char result; result.value = *value; return result; }
   static Char<1> castString(const char* str, uint32_t /*strLen*/) { Char<1> x; x.value = str[0]; return x; }
   inline static Char<1> castString(std::string s) { return castString(s.data(), s.size()); }
};

// --- Numeric Template
static constexpr uint64_t numericShifts[19] = {
    1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull, 100000000ull, 
    1000000000ull, 10000000000ull, 100000000000ull, 1000000000000ull, 10000000000000ull, 
    100000000000000ull, 1000000000000000ull, 10000000000000000ull, 100000000000000000ull, 1000000000000000000ull
};

template <unsigned len, unsigned precision> class Numeric {
 public:
   int64_t value;

   inline Numeric() : value(0) {}
   inline Numeric(Integer x) __attribute__((always_inline)) : value(static_cast<int64_t>(x.value) * numericShifts[precision]) {}
   inline Numeric(int64_t x) : value(x) {}

   inline void assignRaw(long v) { value = v; }
   inline long getRaw() const { return value; }

   inline uint64_t hash() const __attribute__((always_inline));
   
   inline bool operator==(const Numeric& n) const { return value == n.value; }
   inline bool operator!=(const Numeric& n) const { return value != n.value; }
   inline bool operator<(const Numeric& n) const { return value < n.value; }
   inline bool operator<=(const Numeric& n) const { return value <= n.value; }
   inline bool operator>(const Numeric& n) const { return value > n.value; }
   inline bool operator>=(const Numeric& n) const { return value >= n.value; }

   inline Numeric operator+(const Numeric& n) const { return Numeric(value + n.value); }
   inline Numeric& operator+=(const Numeric& n) { value += n.value; return *this; }
   inline Numeric operator-(const Numeric& n) const { return Numeric(value - n.value); }
   
   inline Numeric operator/(const Integer& n) const { return Numeric(value / n.value); }
   template <unsigned l> inline Numeric operator/(const Numeric<l, 0>& n) const { return Numeric(value / n.value); }
   template <unsigned l> inline Numeric operator/(const Numeric<l, 1>& n) const { return Numeric(value * 10 / n.value); }
   template <unsigned l> inline Numeric operator/(const Numeric<l, 2>& n) const { return Numeric(value * 100 / n.value); }
   template <unsigned l> inline Numeric operator/(const Numeric<l, 4>& n) const { return Numeric(value * 10000 / n.value); }

   template <unsigned pO> inline Numeric<len, precision + pO> operator*(const Numeric<len, pO>& n) const {
      return Numeric<len, precision + pO>(value * n.value);
   }

   inline Numeric operator-() { return Numeric(-value); }

   static Numeric<len, precision> castString(const char* str, uint32_t strLen);
   inline static Numeric<len, precision> castString(std::string s) { return castString(s.data(), s.size()); }
   static Numeric<len, precision> buildRaw(long v) { Numeric r; r.value = v; return r; }
};

// --- Date Class
class Date {
 public:
   int32_t value;

   inline Date() : value(0) {}
   inline Date(int32_t value) : value(value) {}

   inline uint64_t hash() const __attribute__((always_inline));
   inline bool operator==(const Date& n) const { return value == n.value; }
   inline bool operator!=(const Date& n) const { return value != n.value; }
   inline bool operator<(const Date& n) const { return value < n.value; }
   inline bool operator<=(const Date& n) const { return value <= n.value; }
   inline bool operator>(const Date& n) const { return value > n.value; }
   inline bool operator>=(const Date& n) const { return value >= n.value; }

   static Date castString(const char* str, uint32_t strLen);
   inline static Date castString(std::string s) { return castString(s.data(), s.size()); }
};

Integer extractYear(const Date& d);

// --- Timestamp Class
class Timestamp {
 public:
   uint64_t value;

   inline Timestamp() : value(0) {}
   inline Timestamp(uint64_t value) : value(value) {}

   static Timestamp null();
   inline uint64_t getRaw() const { return value; }

   inline uint64_t hash() const __attribute__((always_inline));
   inline bool operator==(const Timestamp& t) const { return value == t.value; }
   inline bool operator!=(const Timestamp& t) const { return value != t.value; }
   inline bool operator<(const Timestamp& t) const { return value < t.value; }
   inline bool operator>(const Timestamp& t) const { return value > t.value; }

   static Timestamp castString(const char* str, uint32_t strLen);
   inline static Timestamp castString(std::string s) { return castString(s.data(), s.size()); }
};

// ---------------------------------------------------------------------------
// Implementation section (now inlined)
// ---------------------------------------------------------------------------

template <unsigned maxLen>
inline bool Varchar<maxLen>::operator<(const Varchar& other) const {
   int c = memcmp(value, other.value, std::min(len, other.len));
   return (c < 0) || ((c == 0) && (len < other.len));
}

template <unsigned maxLen>
inline bool Char<maxLen>::operator<(const Char& other) const {
   int c = memcmp(value, other.value, std::min(len, other.len));
   return (c < 0) || ((c == 0) && (len < other.len));
}

template <unsigned maxLen> inline bool Char<maxLen>::operator<=(const Char& other) const {
   int c = memcmp(value, other.value, std::min(len, other.len));
   return (c < 0) || ((c == 0) && (len <= other.len));
}

template <unsigned maxLen> inline bool Char<maxLen>::operator>(const Char& other) const {
   int c = memcmp(value, other.value, std::min(len, other.len));
   return (c > 0) || ((c == 0) && (len > other.len));
}

template <unsigned maxLen> inline bool Char<maxLen>::operator>=(const Char& other) const {
   int c = memcmp(value, other.value, std::min(len, other.len));
   return (c > 0) || ((c == 0) && (len >= other.len));
}

inline uint64_t Integer::hash() const {
   uint64_t r = 88172645463325252ull ^ value;
   r ^= (r << 13); r ^= (r >> 7); return (r ^= (r << 17));
}

inline uint64_t Date::hash() const {
   uint64_t r = 88172645463325252ull ^ value;
   r ^= (r << 13); r ^= (r >> 7); return (r ^= (r << 17));
}

inline uint64_t Timestamp::hash() const {
   uint64_t r = 88172645463325252ull ^ value;
   r ^= (r << 13); r ^= (r >> 7); return (r ^= (r << 17));
}

template <unsigned len, unsigned precision>
inline uint64_t Numeric<len, precision>::hash() const {
   uint64_t r = 88172645463325252ull ^ value;
   r ^= (r << 13); r ^= (r >> 7); return (r ^= (r << 17));
}

// --- Parsing Logic 

template <unsigned len, unsigned precision>
inline Numeric<len, precision> Numeric<len, precision>::castString(const char* str, uint32_t strLen) {
    auto iter = str, limit = str + strLen;
    while ((iter != limit) && ((*iter) == ' ')) ++iter;
    while ((iter != limit) && ((*(limit - 1)) == ' ')) --limit;

    bool neg = false;
    if (iter != limit) {
        if ((*iter) == '-') { neg = true; ++iter; } 
        else if ((*iter) == '+') ++iter;
    }

    int64_t result = 0;
    bool fraction = false;
    unsigned digitsSeen = 0, digitsSeenFraction = 0;
    for (; iter != limit; ++iter) {
        char c = *iter;
        if ((c >= '0') && (c <= '9')) {
            if (fraction) { result = (result * 10) + (c - '0'); ++digitsSeenFraction; }
            else { ++digitsSeen; result = (result * 10) + (c - '0'); }
        } else if (c == '.') {
            if (fraction) throw "invalid number format";
            fraction = true;
        }
    }
    result *= numericShifts[precision - digitsSeenFraction];
    return buildRaw(neg ? -result : result);
}

inline Integer Integer::castString(const char* str, uint32_t strLen) {
    auto iter = str, limit = str + strLen;
    while ((iter != limit) && ((*iter) == ' ')) ++iter;
    while ((iter != limit) && ((*(limit - 1)) == ' ')) --limit;

    bool neg = false;
    if (iter != limit) {
        if ((*iter) == '-') { neg = true; ++iter; } 
        else if ((*iter) == '+') ++iter;
    }

    int64_t result = 0;
    for (; iter != limit; ++iter) {
        char c = *iter;
        if ((c >= '0') && (c <= '9')) {
            result = (result * 10) + (c - '0');
        } else if (c == '.') break;
    }
    return Integer(neg ? -result : result);
}

} // namespace types
#endif
