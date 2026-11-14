# SPDX-License-Identifier: Apache-2.0
# Device kernels for selected-KV prefill. These are built only on the A3
# target and stay separate from the ordinary operator library.
set(SELECTED_KV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
ascendc_library(sparse_prefill_union_pipeline SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_union_pipeline.cpp")
ascendc_library(sparse_prefill_fused_fixed SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_fused_fixed.cpp")
ascendc_library(dsa_packed_inplace SHARED "${SELECTED_KV_SOURCE_DIR}/packed_inplace.cpp")
ascendc_library(dsa_selected_li_pages SHARED "${SELECTED_KV_SOURCE_DIR}/li_page_gather.cpp")

# Address/status checks are required for correctness, including Release builds.
foreach(selected_kv_target IN ITEMS sparse_prefill_union_pipeline sparse_prefill_fused_fixed dsa_packed_inplace)
    ascendc_compile_definitions(${selected_kv_target} PRIVATE ASCENDC_DUMP=1)
    ascendc_compile_options(${selected_kv_target} PRIVATE
        -UNDEBUG
        -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)
endforeach()

# Keep the tested main-kernel build settings unchanged; LI is a separate target.
ascendc_compile_definitions(dsa_selected_li_pages PRIVATE ASCENDC_DUMP=1)
ascendc_compile_options(dsa_selected_li_pages PRIVATE
    -UNDEBUG
    -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)

set(SELECTED_KV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
if(NOT DEFINED VLLM_ASCEND_PACKAGE_PATH)
    set(VLLM_ASCEND_PACKAGE_PATH "${VLLM_ASCEND_INSTALL_PATH}/vllm_ascend")
endif()

set(SELECTED_KV_TARGETS
    sparse_prefill_union_pipeline
    sparse_prefill_fused_fixed
    dsa_packed_inplace
    dsa_selected_li_pages)
set_target_properties(${SELECTED_KV_TARGETS} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SELECTED_KV_LIBRARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE INSTALL_RPATH "$ORIGIN")
add_dependencies(vllm_ascend_C ${SELECTED_KV_TARGETS})
install(TARGETS ${SELECTED_KV_TARGETS} LIBRARY DESTINATION "${VLLM_ASCEND_PACKAGE_PATH}/lib")

# The fixed kernel and the host launch/queue wrapper are not registered yet:
#   sparse_prefill_fused_fixed_launch
#   sparse_prefill_batch_queued
# They depend on the target's selected runtime signatures and will be added with
# the Python/C++ execution wiring.
