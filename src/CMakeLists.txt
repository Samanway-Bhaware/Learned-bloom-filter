add_library(lbf_model STATIC
  model.cpp
  features/ngram_hasher.cpp
  models/logistic_regression.cpp
  models/onnx_model.cpp        # empty TU when LBF_ENABLE_ONNX=OFF
  learned_bloom_filter.cpp)
target_link_libraries(lbf_model PUBLIC lbf)

if(LBF_ENABLE_ONNX)
  find_package(onnxruntime CONFIG REQUIRED)
  target_link_libraries(lbf_model PUBLIC onnxruntime::onnxruntime)
  target_compile_definitions(lbf_model PUBLIC LBF_ENABLE_ONNX)
endif()

