/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_C_PJRT_C_API_GPU_AUTOTUNE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_GPU_AUTOTUNE_EXTENSION_H_

#include <stddef.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes the in-memory GPU autotuning cache.
struct PJRT_Gpu_Autotune_Serialize_Args {
  size_t struct_size;
  bool as_textproto;     // in
  const char* out_data;  // out, valid until next serialize call
  size_t out_size;       // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Autotune_Serialize_Args, out_size);

typedef PJRT_Error* PJRT_Gpu_Autotune_Serialize(
    PJRT_Gpu_Autotune_Serialize_Args* args);

// Loads autotuning results into the in-memory GPU autotuning cache.
struct PJRT_Gpu_Autotune_Load_Args {
  size_t struct_size;
  const char* data;     // in
  size_t data_size;     // in
  bool as_textproto;    // in
  bool allow_override;  // in
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Autotune_Load_Args, allow_override);

typedef PJRT_Error* PJRT_Gpu_Autotune_Load(
    PJRT_Gpu_Autotune_Load_Args* args);

// Clears the in-memory GPU autotuning cache.
struct PJRT_Gpu_Autotune_Clear_Args {
  size_t struct_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Autotune_Clear_Args, struct_size);

typedef PJRT_Error* PJRT_Gpu_Autotune_Clear(
    PJRT_Gpu_Autotune_Clear_Args* args);

typedef struct PJRT_Gpu_Autotune {
  PJRT_Extension_Base base;
  PJRT_Gpu_Autotune_Serialize* serialize;
  PJRT_Gpu_Autotune_Load* load;
  PJRT_Gpu_Autotune_Clear* clear;
} PJRT_Gpu_Autotune;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Autotune, clear);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_GPU_AUTOTUNE_EXTENSION_H_
