// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>

#include <cuda/__memory_resource/resource.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail::__reduce
{
template <class _Buffer, class _Env>
struct __partial_redop
{
  using __buffer_type = _Buffer;
  using __env_type    = _Env;

  _Buffer __buffer;
  _Env __env;
  ::cuda::stream_ref __stream;
};

template <class _Buffer, class _Comm, class _Env, class _InputRange, class _Tp, class _BinaryOp>
[[nodiscard]] _CCCL_HOST_API __partial_redop<_Buffer, _Env> __local_reduction(
  const ::cuda::std::int32_t __ROOT_RANK,
  _Comm&& __comm,
  _Env __env,
  _InputRange&& __inputs,
  const _Tp& __init,
  _BinaryOp __op)
{
  const auto& __logical_device = __comm.device();
  auto __stream                = __stream_from_env(__env);
  auto __resource              = __resource_from_env(__env, __logical_device);

  static_assert(::cuda::mr::resource_with<decltype(__resource), ::cuda::mr::device_accessible>,
                "Provided memory resource must be device accessible");

  const auto __num_items = ::cuda::std::ranges::size(__inputs);
  // Allocate enough storage so that we can use the buffer directly in an in-place comm all
  // gather/all reduce call. Those calls require that the receive buffer is of size nranks *
  // sendcount.
  auto __buff = __make_safe_uninitialized_buffer<_Tp>(__stream, __resource, __comm.size(), __env);
  static_assert(::cuda::std::same_as<decltype(__buff), _Buffer>);

  if (const auto __rank = __comm.rank(); __rank == __ROOT_RANK)
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __logical_device,
      __num_items,
      CUB_NS_QUALIFIER::DeviceReduce::Reduce,
      (::cuda::std::ranges::begin(__inputs),
       // Similarly to above, prepare for the comm calls later. In order for those to be
       // in-place, the sendbuff = recvbuff + rank, so we need to place our partial result
       // there
       __buff.begin() + __rank,
       __num_items_fixed,
       ::cuda::std::move(__op),
       __init,
       __env));
  }
  else
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __logical_device,
      __num_items,
      CUB_NS_QUALIFIER::DeviceReduce::Reduce,
      (::cuda::std::ranges::begin(__inputs),
       // Similarly to above, prepare for the com calls later. In order for those to be
       // in-place, the sendbuff = recvbuff + rank, so we need to place our partial result
       // there
       __buff.begin() + __rank,
       __num_items_fixed,
       ::cuda::std::move(__op),
       CUB_NS_QUALIFIER::detail::reduce::no_init,
       __env));
  }
  return {::cuda::std::move(__buff), ::cuda::std::move(__env), __stream};
}
} // namespace __detail::__reduce

template <class _RangeOfRanges>
_CCCL_CONCEPT __range_of_sized_random_access_ranges = _CCCL_REQUIRES_EXPR((_RangeOfRanges), )(
  requires(::cuda::std::ranges::forward_range<_RangeOfRanges>),
  requires(::cuda::std::ranges::sized_range<_RangeOfRanges>),
  requires(::cuda::std::ranges::random_access_range<::cuda::std::ranges::range_reference_t<_RangeOfRanges>>));

template <class _RangeOfIters, class _Tp>
_CCCL_CONCEPT __range_of_output_iters = _CCCL_REQUIRES_EXPR((_RangeOfIters, _Tp), )(
  requires(::cuda::std::ranges::forward_range<_RangeOfIters>),
  requires(::cuda::std::output_iterator<::cuda::std::ranges::range_reference_t<_RangeOfIters>, _Tp>));

//! @brief Reduce each input range over its communicator and write one result per output
//! iterator.
//!
//! Performs one reduction per communicator in parallel across devices. The communicators,
//! environments, input ranges and output iterators are iterated in lockstep, so for the
//! i-th element of each range the i-th input range is reduced with `__op` seeded by
//! `__init` on the i-th communicator's devices, the partial results are combined across
//! all ranks of that communicator, and the final value is written through the i-th output
//! iterator.
//!
//! All five ranges must have the same length.
//!
//! After this call returns, all local output iterators will hold the same value. In that sense
//! this routine is similar to an "all reduce".
//!
//! @tparam _CommRange The range of communicators. Each element must model the communicator
//!         concept.
//! @tparam _EnvRange The range of execution environments. Each environment supplies the
//!         stream and memory resource used for its communicator.
//! @tparam _InputRangeOfRanges The range whose elements are the per-communicator input
//!         ranges. Each element must be a sized random-access range.
//! @tparam _RangeOfOutputIt The range of output iterators, one per communicator.
//! @tparam _Tp The reduction and result value type. Deduced by default from the output
//!         element type.
//! @tparam _BinaryOp The binary reduction operator type. Defaults to `::cuda::std::plus<>`.
//!
//! @param[in] __comms The range of communicators.
//! @param[in] __envs The range of execution environments.
//! @param[in] __range_of_inputs The range of per-communicator input ranges to reduce.
//! @param[out] __outputs The range of output iterators receiving the per-communicator results.
//! @param[in] __init The initial value seeding each reduction.
//! @param[in] __op The binary reduction operator.
_CCCL_TEMPLATE(class _CommRange,
               class _EnvRange,
               class _InputRangeOfRanges,
               class _RangeOfOutputIt,
               class _Tp       = ::cuda::std::iter_value_t<::cuda::std::ranges::range_reference_t<_RangeOfOutputIt>>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(
  __range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND
    __range_of_sized_random_access_ranges<_InputRangeOfRanges> _CCCL_AND __range_of_output_iters<_RangeOfOutputIt, _Tp>)
_CCCL_HOST_API void reduce(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRangeOfRanges&& __range_of_inputs,
  _RangeOfOutputIt&& __outputs,
  _Tp __init     = {},
  _BinaryOp __op = {})
{
  static_assert(::cuda::std::ranges::sized_range<_CommRange>);

  using __properties =
    ::cuda::experimental::__detail::__in_range_out_it_properties<_InputRangeOfRanges, _RangeOfOutputIt, _EnvRange>;
  using __partial_type = ::cuda::experimental::__detail::__reduce::__partial_redop<typename __properties::__buffer_type,
                                                                                   typename __properties::__env_type>;

  constexpr auto __ROOT_RANK = 0;
  const auto __num_local     = ::cuda::std::ranges::size(__comms);
  auto __partials            = ::std::vector<__partial_type>{};

  __partials.reserve(__num_local);
  // TODO(jfaibussowit): can just be ranges::zip | ranges::transform | ranges::to() (and then
  // we don't need to do the env, and buffer type deduction upfront)
  for (auto&& [__comm, __env, __inputs] : ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_inputs))
  {
    __partials.emplace_back(
      ::cuda::experimental::__detail::__reduce::__local_reduction<typename __partial_type::__buffer_type>(
        __ROOT_RANK, __comm, __env, __inputs, __init, __op));
  }

  if (__num_local)
  {
    auto&& __token = ::cuda::std::ranges::begin(__comms)->group_token();

    for (const auto& [__comm, __local] : ::cuda::std::ranges::views::zip(__comms, __partials))
    {
      auto* const __ptr = __local.__buffer.data();

      __comm.all_gather(__token, __ptr + __comm.rank(), __ptr, /*__count=*/1, __local.__stream);
    }
  }

  // TODO(jfaibussowit): Implement specialized reduction path where we call ncclAllReduce()
  // directly on the partials (or directly on the inputs). Calling on the partials requires:
  //
  // 1. The op maps directly to a nccl op.
  // 2. The value type maps directly to a nccl value type.
  //
  // Calling directly on the inputs further requires:
  //
  // 1. All input ranges are contiguous ranges.
  // 2. The initializer is exactly the identity value for the chosen op. So 0 for sum or prod
  //    and MIN/MAX for max/min respectively.
  for (auto&& [__comm, __part, __out] : ::cuda::std::ranges::views::zip(__comms, __partials, __outputs))
  {
    auto&& [__buffer, __env, _] = __part;
    const auto __num_items      = __buffer.size();

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.device(),
      __num_items,
      CUB_NS_QUALIFIER::DeviceReduce::Reduce,
      (__buffer.begin(), __out, __num_items_fixed, __op, CUB_NS_QUALIFIER::detail::reduce::no_init, __env));
  }
}

//! @brief Reduce a single input range over a single communicator using the given execution
//! environment.
//!
//! Convenience wrapper that forwards a single `(communicator, environment, input range,
//! output iterator)` to the range-based overload. The input range is reduced with `__op`
//! seeded by `__init` across the communicator's ranks and the final value is written
//! through `__output`.
//!
//! @tparam _Comm The communicator type. Must model the communicator concept.
//! @tparam _Env The execution environment type. Supplies the stream and memory resource.
//! @tparam _InputRange The input range type. Must be a random-access range.
//! @tparam _OutputIt The output iterator type.
//! @tparam _Tp The reduction and result value type. Deduced by default from the output
//! value type.
//! @tparam _BinaryOp The binary reduction operator type. Defaults to `::cuda::std::plus<>`.
//!
//! @param[in] __comm The communicator.
//! @param[in] __env The execution environment.
//! @param[in] __inputs The input range to reduce.
//! @param[out] __output The output iterator receiving the result.
//! @param[in] __init The initial value seeding the reduction.
//! @param[in] __op The binary reduction operator.
_CCCL_TEMPLATE(class _Comm,
               class _Env,
               class _InputRange,
               class _OutputIt,
               class _Tp       = ::cuda::std::iter_value_t<_OutputIt>,
               class _BinaryOp = ::cuda::std::plus<>)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::ranges::random_access_range<_InputRange>
                 _CCCL_AND ::cuda::std::output_iterator<_OutputIt, _Tp>)
_CCCL_HOST_API void
reduce(_Comm&& __comm, _Env&& __env, _InputRange&& __inputs, _OutputIt __output, _Tp __init = {}, _BinaryOp __op = {})
{
  reduce(::cuda::std::span<::cuda::std::remove_reference_t<_Comm>, 1>{::cuda::std::addressof(__comm), 1},
         ::cuda::std::span<::cuda::std::remove_reference_t<_Env>, 1>{::cuda::std::addressof(__env), 1},
         ::cuda::std::span<::cuda::std::remove_reference_t<_InputRange>, 1>{::cuda::std::addressof(__inputs), 1},
         ::cuda::std::span<::cuda::std::remove_reference_t<_OutputIt>, 1>{::cuda::std::addressof(__output), 1},
         ::cuda::std::move(__init),
         ::cuda::std::move(__op));
}

//! @brief Reduce a single input range over a single communicator using
//! `::cuda::std::plus<>` and a default-constructed initial value.
//!
//! Convenience wrapper that supplies a default execution environment and a default
//! `::cuda::std::plus<>` reduction, forwarding to the environment-taking overload. The
//! result value type is the input range's value type.
//!
//! @tparam _Comm The communicator type. Must model the communicator concept.
//! @tparam _InputRange The input range type. Must be a random-access range.
//! @tparam _OutputIt The output iterator type.
//!
//! @param[in] __comm The communicator.
//! @param[in] __inputs The input range to reduce.
//! @param[out] __output The output iterator receiving the result.
_CCCL_TEMPLATE(class _Comm, class _InputRange, class _OutputIt)
_CCCL_REQUIRES(__communicator<_Comm> _CCCL_AND ::cuda::std::ranges::random_access_range<_InputRange>
                 _CCCL_AND ::cuda::std::output_iterator<_OutputIt, ::cuda::std::ranges::range_value_t<_InputRange>>)
_CCCL_HOST_API void reduce(_Comm&& __comm, _InputRange&& __inputs, _OutputIt __output)
{
  reduce(::cuda::std::forward<_Comm>(__comm),
         ::cuda::std::execution::env<>{},
         ::cuda::std::forward<_InputRange>(__inputs),
         ::cuda::std::forward<_OutputIt>(__output));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_REDUCE_H
