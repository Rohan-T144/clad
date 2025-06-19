# Paths
INFER_SRC      = inference.cpp
INCLUDE_DIR    = ../../include/
OUT_DEFAULT    = inference
OUT_OMP        = inference_par
PROFILING_DIR  = profiling

# Compiler and flags
CXX            = c++
CXXFLAGS       = -std=c++20 -O3 -I$(INCLUDE_DIR)
OMP_FLAGS      = -DOMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -Xpreprocessor -fopenmp

# Default target: build both
all: $(OUT_DEFAULT) $(OUT_OMP)

# Build without OpenMP
$(OUT_DEFAULT): $(INFER_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Build with OpenMP
$(OUT_OMP): $(INFER_SRC)
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) $< -o $@

# Clean build artifacts and profiling output
clean:
	rm -f $(OUT_DEFAULT) $(OUT_OMP)
	rm -rf $(PROFILING_DIR)

# Run profiler on selected binary (default: inference)
profile:
	mkdir -p $(PROFILING_DIR)
	xctrace record --output $(PROFILING_DIR)/ --template "Time Profiler" --time-limit 30s --launch ./$(BIN)

.PHONY: all clean profile
