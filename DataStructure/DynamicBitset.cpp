#if __has_include(<bit>)
  #include <bit>     // C++20 popcount/countr_zero
#endif

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

class DynamicBitset {
public:
    using block_type = std::uint64_t;
    using size_type  = std::size_t;

    static constexpr size_type npos = static_cast<size_type>(-1);
    static constexpr size_type bits_per_block = 64;

    // ----- proxy reference (like std::bitset::reference) -----
    class reference {
        friend class DynamicBitset;
        block_type* blk_ = nullptr;
        block_type  mask_ = 0;

        reference(block_type& blk, size_type bit_in_block) noexcept
            : blk_(&blk), mask_(block_type(1) << bit_in_block) {}

    public:
        reference() = delete;

        reference& operator=(bool v) noexcept {
            if (v) *blk_ |= mask_;
            else   *blk_ &= ~mask_;
            return *this;
        }

        reference& operator=(const reference& r) noexcept {
            return (*this = static_cast<bool>(r));
        }

        operator bool() const noexcept {
            return (*blk_ & mask_) != 0;
        }

        reference& flip() noexcept {
            *blk_ ^= mask_;
            return *this;
        }
    };

    // ----- ctors -----
    DynamicBitset() = default;

    explicit DynamicBitset(size_type nbits, bool value = false)
        : nbits_(nbits), data_(blocks_for(nbits), value ? ~block_type(0) : block_type(0)) {
        trim_();
    }

    // 从 0/1 字符串构造（与 std::bitset::to_string 方向一致：s[0] 是最高位）
    explicit DynamicBitset(const std::string& s, char zero = '0', char one = '1')
        : nbits_(s.size()), data_(blocks_for(s.size()), 0) {
        for (size_type i = 0; i < s.size(); ++i) {
            const char c = s[i];
            if (c == one) {
                const size_type pos = (s.size() - 1 - i); // s[0] -> highest bit
                set_unchecked_(pos, true);
            } else if (c != zero) {
                throw std::invalid_argument("DynamicBitset: invalid character in string");
            }
        }
        trim_();
    }

    // ----- capacity -----
    size_type size() const noexcept { return nbits_; }
    bool empty() const noexcept { return nbits_ == 0; }

    void reserve(size_type nbits) {
        data_.reserve(blocks_for(nbits));
    }

    void resize(size_type nbits, bool value = false) {
        const size_type old_bits = nbits_;
        const size_type old_blocks = data_.size();

        nbits_ = nbits;
        const size_type new_blocks = blocks_for(nbits_);

        if (new_blocks != old_blocks) {
            data_.resize(new_blocks, value ? ~block_type(0) : block_type(0));
        }

        if (value && nbits_ > old_bits) {
            // 补齐 old_bits..nbits_-1 为 1
            if (old_bits > 0) {
                const size_type b = old_bits / bits_per_block;
                const size_type o = old_bits % bits_per_block;

                if (b < data_.size()) {
                    if (o != 0) {
                        // 把原最后一个 block 中 o..63 先置 1（再 trim 到新大小）
                        data_[b] |= (~block_type(0) << o);
                    }
                    // 新增 blocks 已经用 ~0 初始化了，无需额外处理
                }
            } else {
                // old_bits==0，若 new_blocks>0 且 value==true，vector resize 已经填 ~0
            }
        }

        trim_();
    }

    // ----- element access -----
    // 与 std::bitset 一致：operator[] 不做越界检查（越界是 UB / 由 assert 捕获）
    reference operator[](size_type pos) noexcept {
        assert(pos < nbits_);
        return reference(data_[pos / bits_per_block], pos % bits_per_block);
    }

    bool operator[](size_type pos) const noexcept {
        assert(pos < nbits_);
        return test_unchecked_(pos);
    }

    // 与 std::bitset::test 一致：越界抛 out_of_range
    bool test(size_type pos) const {
        if (pos >= nbits_) throw std::out_of_range("DynamicBitset::test out of range");
        return test_unchecked_(pos);
    }

    // ----- modifiers -----
    DynamicBitset& set() noexcept {
        std::fill(data_.begin(), data_.end(), ~block_type(0));
        trim_();
        return *this;
    }

    DynamicBitset& reset() noexcept {
        std::fill(data_.begin(), data_.end(), block_type(0));
        return *this;
    }

    DynamicBitset& flip() noexcept {
        for (auto& x : data_) x = ~x;
        trim_();
        return *this;
    }

    DynamicBitset& set(size_type pos, bool value = true) {
        if (pos >= nbits_) throw std::out_of_range("DynamicBitset::set(pos) out of range");
        set_unchecked_(pos, value);
        return *this;
    }

    DynamicBitset& reset(size_type pos) {
        return set(pos, false);
    }

    DynamicBitset& flip(size_type pos) {
        if (pos >= nbits_) throw std::out_of_range("DynamicBitset::flip(pos) out of range");
        data_[pos / bits_per_block] ^= (block_type(1) << (pos % bits_per_block));
        return *this;
    }

    // ----- queries -----
    size_type count() const noexcept {
        size_type ans = 0;
        for (block_type x : data_) ans += popcount64_(x);
        return ans;
    }

    bool any() const noexcept {
        for (block_type x : data_) if (x) return true;
        return false;
    }

    bool none() const noexcept { return !any(); }

    bool all() const noexcept {
        if (nbits_ == 0) return true; // vacuously true
        const size_type nb = data_.size();
        if (nb == 0) return true;

        for (size_type i = 0; i + 1 < nb; ++i) {
            if (data_[i] != ~block_type(0)) return false;
        }
        return (data_.back() & last_mask_()) == last_mask_();
    }

    // ----- find set bits (fast iteration) -----
    // 返回第一个置 1 bit 的位置，若不存在返回 npos
    size_type find_first() const noexcept {
        for (size_type i = 0; i < data_.size(); ++i) {
            const block_type x = data_[i];
            if (x) return i * bits_per_block + ctz64_(x);
        }
        return npos;
    }

    // 返回严格大于 pos 的下一个置 1 bit 的位置，若不存在返回 npos
    size_type find_next(size_type pos) const noexcept {
        if (pos >= nbits_) return npos;
        pos += 1;
        if (pos >= nbits_) return npos;

        size_type i = pos / bits_per_block;
        const size_type off = pos % bits_per_block;

        block_type x = data_[i] & (~block_type(0) << off);
        if (x) return i * bits_per_block + ctz64_(x);

        for (++i; i < data_.size(); ++i) {
            x = data_[i];
            if (x) return i * bits_per_block + ctz64_(x);
        }
        return npos;
    }

    // ----- conversion -----
    std::string to_string(char zero = '0', char one = '1') const {
        std::string s;
        s.resize(nbits_);
        for (size_type i = 0; i < nbits_; ++i) {
            const bool bit = test_unchecked_(nbits_ - 1 - i);
            s[i] = bit ? one : zero;
        }
        return s;
    }

    unsigned long to_ulong() const {
        return to_uint_checked_<unsigned long>();
    }

    unsigned long long to_ullong() const {
        return to_uint_checked_<unsigned long long>();
    }

    // ----- bitwise ops (sizes must match, like std::bitset) -----
    DynamicBitset& operator&=(const DynamicBitset& rhs) noexcept {
        assert(nbits_ == rhs.nbits_);
        for (size_type i = 0; i < data_.size(); ++i) data_[i] &= rhs.data_[i];
        // 不必 trim（双方都保持 trimmed），但保守起见可 trim
        trim_();
        return *this;
    }

    DynamicBitset& operator|=(const DynamicBitset& rhs) noexcept {
        assert(nbits_ == rhs.nbits_);
        for (size_type i = 0; i < data_.size(); ++i) data_[i] |= rhs.data_[i];
        trim_();
        return *this;
    }

    DynamicBitset& operator^=(const DynamicBitset& rhs) noexcept {
        assert(nbits_ == rhs.nbits_);
        for (size_type i = 0; i < data_.size(); ++i) data_[i] ^= rhs.data_[i];
        trim_();
        return *this;
    }

    // ----- shifts -----
    DynamicBitset& operator<<=(size_type k) noexcept {
        if (k == 0 || nbits_ == 0) return *this;
        if (k >= nbits_) return reset();

        const size_type nb = data_.size();
        const size_type bs = k / bits_per_block;
        const size_type os = k % bits_per_block;

        if (bs) {
            for (size_type i = nb; i-- > bs; ) data_[i] = data_[i - bs];
            std::fill(data_.begin(), data_.begin() + bs, block_type(0));
        }

        if (os) {
            for (size_type i = nb; i-- > 1; ) {
                data_[i] = (data_[i] << os) | (data_[i - 1] >> (bits_per_block - os));
            }
            data_[0] <<= os;
        }

        trim_();
        return *this;
    }

    DynamicBitset& operator>>=(size_type k) noexcept {
        if (k == 0 || nbits_ == 0) return *this;
        if (k >= nbits_) return reset();

        const size_type nb = data_.size();
        const size_type bs = k / bits_per_block;
        const size_type os = k % bits_per_block;

        if (bs) {
            for (size_type i = 0; i + bs < nb; ++i) data_[i] = data_[i + bs];
            std::fill(data_.end() - bs, data_.end(), block_type(0));
        }

        if (os) {
            for (size_type i = 0; i + 1 < nb; ++i) {
                data_[i] = (data_[i] >> os) | (data_[i + 1] << (bits_per_block - os));
            }
            data_[nb - 1] >>= os;
        }

        trim_();
        return *this;
    }

    // ----- friends: operators -----
    friend DynamicBitset operator~(DynamicBitset x) noexcept { x.flip(); return x; }

    friend DynamicBitset operator&(DynamicBitset a, const DynamicBitset& b) noexcept { a &= b; return a; }
    friend DynamicBitset operator|(DynamicBitset a, const DynamicBitset& b) noexcept { a |= b; return a; }
    friend DynamicBitset operator^(DynamicBitset a, const DynamicBitset& b) noexcept { a ^= b; return a; }

    friend DynamicBitset operator<<(DynamicBitset a, size_type k) noexcept { a <<= k; return a; }
    friend DynamicBitset operator>>(DynamicBitset a, size_type k) noexcept { a >>= k; return a; }

    friend bool operator==(const DynamicBitset& a, const DynamicBitset& b) noexcept {
        return a.nbits_ == b.nbits_ && a.data_ == b.data_;
    }
    friend bool operator!=(const DynamicBitset& a, const DynamicBitset& b) noexcept {
        return !(a == b);
    }

    friend std::ostream& operator<<(std::ostream& os, const DynamicBitset& b) {
        return os << b.to_string();
    }

    void swap(DynamicBitset& other) noexcept {
        std::swap(nbits_, other.nbits_);
        data_.swap(other.data_);
    }

private:
    size_type nbits_ = 0;
    std::vector<block_type> data_;

    static size_type blocks_for(size_type nbits) noexcept {
        return (nbits + bits_per_block - 1) / bits_per_block;
    }

    block_type last_mask_() const noexcept {
        const size_type r = nbits_ % bits_per_block;
        if (r == 0) return ~block_type(0);
        return (block_type(1) << r) - 1;
    }

    void trim_() noexcept {
        if (!data_.empty()) data_.back() &= last_mask_();
    }

    bool test_unchecked_(size_type pos) const noexcept {
        return (data_[pos / bits_per_block] >> (pos % bits_per_block)) & 1;
    }

    void set_unchecked_(size_type pos, bool value) noexcept {
        block_type& blk = data_[pos / bits_per_block];
        const block_type m = block_type(1) << (pos % bits_per_block);
        if (value) blk |= m;
        else       blk &= ~m;
    }

    // ----- fast bit ops (popcount / ctz) -----
    static inline std::uint32_t popcount64_(std::uint64_t x) noexcept {
    #if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
        return static_cast<std::uint32_t>(std::popcount(x));
    #elif defined(_MSC_VER)
      #if defined(_M_X64) || defined(_M_ARM64)
        return static_cast<std::uint32_t>(__popcnt64(x));
      #else
        return static_cast<std::uint32_t>(__popcnt(static_cast<unsigned>(x)) +
                                          __popcnt(static_cast<unsigned>(x >> 32)));
      #endif
    #else
        return static_cast<std::uint32_t>(__builtin_popcountll(static_cast<unsigned long long>(x)));
    #endif
    }

    // x 必须非 0
    static inline std::uint32_t ctz64_(std::uint64_t x) noexcept {
    #if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
        return static_cast<std::uint32_t>(std::countr_zero(x));
    #elif defined(_MSC_VER)
        unsigned long idx = 0;
      #if defined(_M_X64) || defined(_M_ARM64)
        _BitScanForward64(&idx, x);
        return static_cast<std::uint32_t>(idx);
      #else
        // 32-bit: split scan
        const unsigned lo = static_cast<unsigned>(x);
        if (lo) {
            _BitScanForward(&idx, lo);
            return static_cast<std::uint32_t>(idx);
        }
        _BitScanForward(&idx, static_cast<unsigned>(x >> 32));
        return static_cast<std::uint32_t>(idx + 32);
      #endif
    #else
        return static_cast<std::uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(x)));
    #endif
    }

    template <class U>
    U to_uint_checked_() const {
        static_assert(std::is_unsigned<U>::value, "U must be an unsigned integer type");

        constexpr size_type W = std::numeric_limits<U>::digits;
        // 检查溢出：若有任何位 >= W 为 1，则 overflow
        if (nbits_ > W) {
            const size_type idx = W / bits_per_block;
            const size_type off = W % bits_per_block;

            if (idx < data_.size()) {
                if (off != 0) {
                    const block_type m = (~block_type(0) << off);
                    if (data_[idx] & m) throw std::overflow_error("DynamicBitset: to_uint overflow");
                } else {
                    if (data_[idx]) throw std::overflow_error("DynamicBitset: to_uint overflow");
                }
                for (size_type j = idx + 1; j < data_.size(); ++j) {
                    if (data_[j]) throw std::overflow_error("DynamicBitset: to_uint overflow");
                }
            }
        }

        // 取低 W 位
        U res = 0;
        size_type remaining = std::min(nbits_, W);

        size_type bi = 0;
        size_type shift = 0;
        while (remaining > 0) {
            const size_type take = std::min(remaining, bits_per_block);
            const block_type mask = (take == bits_per_block) ? ~block_type(0)
                                                            : ((block_type(1) << take) - 1);
            const block_type part = data_[bi] & mask;
            res |= (static_cast<U>(part) << shift);

            remaining -= take;
            shift += take;
            ++bi;
        }
        return res;
    }
};