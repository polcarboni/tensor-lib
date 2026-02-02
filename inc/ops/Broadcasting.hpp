#pragma once
#include "core/TensorImpl.hpp"
#include <vector>

namespace tensor
{
    struct BroadcastInfo
    {
        std::vector<size_t> shape_;
        std::vector<size_t> strides_lhs_;    /* Virtual strides for lhs */
        std::vector<size_t> strides_rhs_;    /* Virtual strides for rhs */
    };
    
    inline std::vector<size_t> broadcast_shapes(const std::vector<size_t>& s1, const std::vector<size_t>& s2);
    // TODO: this has to be integrated in the dispatcher function interface
    inline BroadcastInfo get_broadcast_info(const TensorImpl& lhs, const TensorImpl& rhs);

} // namespace tensor