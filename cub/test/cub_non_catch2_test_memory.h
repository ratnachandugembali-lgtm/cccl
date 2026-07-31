// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

// Non-Catch2 runtime tests must declare exactly one memory class. CMake reads
// this declaration when assigning the test's GPU resource claim.

#define CUB_TEST_MEMORY_CLASS(MEMORY_CLASS) CUB_TEST_MEMORY_CLASS_##MEMORY_CLASS

#define CUB_TEST_MEMORY_CLASS_CUB_SMALL static_assert(true)
#define CUB_TEST_MEMORY_CLASS_CUB_LARGE static_assert(true)
