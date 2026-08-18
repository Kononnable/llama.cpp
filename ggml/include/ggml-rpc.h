#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    5
#define RPC_PROTO_MINOR_VERSION    3
#define RPC_PROTO_PATCH_VERSION    0

// Minor-version protocol features (client gates on negotiated server minor):
//   5.1  rpc_tensor data may carry a buffer-relative offset (RPC_TENSOR_FLAG2_DATA_IS_OFFSET)
//   5.2  RPC_CMD_SUPPORTS_OP / RPC_CMD_GET_DEV_PROPS (honest remote capability reporting)
//   5.3  RPC_CMD_GET_DEV_PROPS response includes the remote device type

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);
GGML_BACKEND_API bool ggml_backend_rpc_dev_is_rpc(ggml_backend_dev_t dev);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, uint32_t poll, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
