// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_reduce.cuh>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto [items, threads] =
      cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, int{sizeof(AccumT)});
    const auto policy = cub::ReducePassPolicy{
      threads, items, 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using init_t = T;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // FIXME(bgruber): the previous implementation did target cub::DispatchReduce, and provided T as accumulator type.
  // This is not realistic, since a user cannot override the accumulator type the same way at the public API. For
  // example, reducing I8 over cuda::std::plus deduces accumulator type I32 at the public API, but the benchmark forces
  // it to I8. This skews the MemBoundScaling, leading to 20% regression for the same tuning when the public API is
  // called (with accum_t I32) over the benchmark (forced accum_t of I8). See also:
  // https://github.com/NVIDIA/cccl/issues/6576
  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<T>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceReduce::Reduce, "Reduce failed", d_in, d_out, static_cast<OffsetT>(elements), op_t{}, init_t{}, env);
  });
}

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
