// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/compute_capability.h>

CUB_NAMESPACE_BEGIN

//! The tuning policy for a single pass of deterministic (gpu-to-gpu) reduction.
struct ReduceDeterministicPassPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  BlockReduceAlgorithm block_algorithm; //!< The @ref BlockReduceAlgorithm used for block-wide reduction

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const ReduceDeterministicPassPolicy& lhs, const ReduceDeterministicPassPolicy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.block_algorithm == rhs.block_algorithm;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const ReduceDeterministicPassPolicy& lhs, const ReduceDeterministicPassPolicy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const ReduceDeterministicPassPolicy& p)
  {
    return os << "ReduceDeterministicPassPolicy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .block_algorithm = " << p.block_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for deterministic (gpu-to-gpu) reduction algorithms in @ref DeviceReduce.
struct ReduceDeterministicPolicy
{
  ReduceDeterministicPassPolicy multi_tile; //!< Policy for the multi-tile reduce pass
  ReduceDeterministicPassPolicy single_tile; //!< Policy for the single-tile reduce pass

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const ReduceDeterministicPolicy& lhs, const ReduceDeterministicPolicy& rhs)
  {
    return lhs.multi_tile == rhs.multi_tile && lhs.single_tile == rhs.single_tile;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const ReduceDeterministicPolicy& lhs, const ReduceDeterministicPolicy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const ReduceDeterministicPolicy& p)
  {
    return os
        << "ReduceDeterministicPolicy { .multi_tile = " << p.multi_tile << ", .single_tile = " << p.single_tile << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::rfa
{
struct policy_selector
{
  type_t accum_t;
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> ReduceDeterministicPolicy
  {
    if (cc >= ::cuda::compute_capability{9, 0})
    {
      // only tuned for float, fall through for other types
      if (accum_t == type_t::float32)
      {
        // ipt_13.tpb_224  1.107188  1.009709  1.097114  1.316820
        const auto scaled = scale_mem_bound(224, 13, accum_size);
        return {{scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
    }

    if (cc >= ::cuda::compute_capability{8, 6})
    {
      // only tuned for float and double, fall through for other types
      if (accum_t == type_t::float32)
      {
        // ipt_6.tpb_224  1.034383  1.000000  1.032097  1.090909
        const auto scaled = scale_mem_bound(224, 6, accum_size);
        return {{scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
      if (accum_t == type_t::float64)
      {
        // ipt_11.tpb_128 ()  1.232089  1.002124  1.245336  1.582279
        const auto scaled = scale_mem_bound(128, 11, accum_size);
        return {{scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const auto scaled = scale_mem_bound(256, 16, accum_size);
      return {{scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
              {scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
    }

    const auto scaled = scale_mem_bound(256, 20, accum_size);
    return {{scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
            {scaled.threads_per_block, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
  }
};

// stateless version which can be passed to kernels
template <typename AccumT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> ReduceDeterministicPolicy
  {
    return policy_selector{classify_type<AccumT>, int{sizeof(AccumT)}}(cc);
  }
};
} // namespace detail::rfa

CUB_NAMESPACE_END
