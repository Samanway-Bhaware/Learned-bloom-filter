# Training CLI — Phase 3 Step 3.
# Trains a logistic regression Learned Bloom Filter from a labelled TSV file.

add_executable(lbf_train train_main.cpp)
target_link_libraries(lbf_train PRIVATE lbf_model)
target_include_directories(lbf_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
# cxxopts is header-only; include as SYSTEM to suppress third-party warnings.
target_include_directories(lbf_train SYSTEM PRIVATE ${cxxopts_SOURCE_DIR}/include)

if(MSVC)
  # std::getline and friends trigger C4996 ("may be unsafe") under /W4.
  target_compile_definitions(lbf_train PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
