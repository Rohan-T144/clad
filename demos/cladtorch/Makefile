# Robust Makefile for building multiple C++ targets with clad plugin support
# Usage: make [target] [BUILD_TYPE=debug|release] [VERBOSE=1] [other variables...]

# =============================================================================
# Configuration Variables
# =============================================================================

# Build configuration
BUILD_TYPE ?= release
VERBOSE ?= 0
DEBUG ?= 0
STD_VERSION ?= c++20
STDLIB ?= libc++

# Build type specific paths
ifeq ($(BUILD_TYPE),debug)
    LLVM_PATH ?= ../../../llvm-project/build_debug
    CLAD_PATH ?= ../../../clad
    CLAD_BUILD ?= build_debug
    BUILD_FLAGS ?= -g -O0 -DDEBUG
else ifeq ($(BUILD_TYPE),release)
    LLVM_PATH ?= $(shell brew --prefix llvm@18 2>/dev/null || echo "/usr/local/opt/llvm@18")
    CLAD_PATH ?= ../../../clad
    CLAD_BUILD ?= build_RelWithDebInfo
    BUILD_FLAGS ?= -O3 -DNDEBUG
else
    $(error Invalid BUILD_TYPE: $(BUILD_TYPE). Must be 'debug' or 'release')
endif

# Compiler configuration
CXX = $(LLVM_PATH)/bin/clang++
CLAD_PLUGIN = $(CLAD_PATH)/$(CLAD_BUILD)/lib/clad.dylib
CLAD_INCLUDE = $(CLAD_PATH)/include/

# =============================================================================
# Target Definitions
# =============================================================================

# Define all available targets and their source files
TARGETS = train_llm inference infgrad mvp nn nn_static nntest test train_pool

# Source files for each target (defaults to <target>.cpp)
train_llm_SRC = train_llm.cpp
inference_SRC = inference.cpp
infgrad_SRC = infgrad.cpp
mvp_SRC = mvp.cpp
nn_SRC = nn.cpp
nn_static_SRC = nn_static.cpp
nntest_SRC = nntest.cpp
test_SRC = test.cpp
train_pool_SRC = train_pool.cpp

# Per-target feature flags (override defaults)
# Example: train_llm_USE_OMP = 1
# Example: inference_USE_ACCELERATE = 0

# =============================================================================
# Feature Flags (Global Defaults)
# =============================================================================

# OpenMP support
USE_OMP ?= 0
ifeq ($(USE_OMP),1)
    OMP_FLAGS = -DOMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -Xpreprocessor -fopenmp
else
    OMP_FLAGS =
endif

# Accelerate framework support
USE_ACCELERATE ?= 1
ifeq ($(USE_ACCELERATE),1)
    ACCELERATE_FLAGS = -framework Accelerate -DACCELERATE_NEW_LAPACK
else
    ACCELERATE_FLAGS =
endif

# Fast math
USE_FAST_MATH ?= 1
ifeq ($(USE_FAST_MATH),1)
    FAST_MATH_FLAGS = -ffast-math
else
    FAST_MATH_FLAGS =
endif

# Clad plugin arguments
DUMP_DERIVED_FN ?= 0
ifeq ($(DUMP_DERIVED_FN),1)
    CLAD_PLUGIN_ARGS = -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn
else
    CLAD_PLUGIN_ARGS =
endif

# =============================================================================
# Compiler Flags
# =============================================================================

# Base flags
BASE_FLAGS = -std=$(STD_VERSION) -I$(CLAD_INCLUDE)

# Clad plugin flags
CLAD_FLAGS = -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang $(CLAD_PLUGIN) $(CLAD_PLUGIN_ARGS)

# macOS specific flags
ifeq ($(shell uname -s),Darwin)
    PLATFORM_FLAGS = -isysroot $(shell xcrun --show-sdk-path) -stdlib=$(STDLIB)
else
    PLATFORM_FLAGS =
endif

# Combined flags
CXXFLAGS = $(BASE_FLAGS) $(CLAD_FLAGS) $(BUILD_FLAGS) $(PLATFORM_FLAGS) $(OMP_FLAGS) $(ACCELERATE_FLAGS) $(FAST_MATH_FLAGS)

# Extra flags from command line
EXTRA_FLAGS ?=
ifneq ($(EXTRA_FLAGS),)
    CXXFLAGS += $(EXTRA_FLAGS)
endif

# =============================================================================
# Colors and Logging
# =============================================================================

# Colors
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m

# Logging functions
define log_info
	@echo "$(BLUE)[INFO]$(NC) $(1)"
endef

define log_warn
	@echo "$(YELLOW)[WARN]$(NC) $(1)"
endef

define log_error
	@echo "$(RED)[ERROR]$(NC) $(1)"
endef

define log_success
	@echo "$(GREEN)[SUCCESS]$(NC) $(1)"
endef

# =============================================================================
# Helper Functions
# =============================================================================

# Function to get source file for a target
get_source = $(if $($(1)_SRC),$($(1)_SRC),$(1).cpp)

# Function to get target-specific flags
define get_target_flags
$(if $($(1)_USE_OMP),$(if $(filter 1,$($(1)_USE_OMP)),-DOMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -Xpreprocessor -fopenmp,),$(OMP_FLAGS)) \
$(if $($(1)_USE_ACCELERATE),$(if $(filter 1,$($(1)_USE_ACCELERATE)),-framework Accelerate -DACCELERATE_NEW_LAPACK,$(if $(filter 0,$($(1)_USE_ACCELERATE)),,$(ACCELERATE_FLAGS))),$(ACCELERATE_FLAGS)) \
$(if $($(1)_USE_FAST_MATH),$(if $(filter 1,$($(1)_USE_FAST_MATH)),-ffast-math,$(if $(filter 0,$($(1)_USE_FAST_MATH)),,$(FAST_MATH_FLAGS))),$(FAST_MATH_FLAGS)) \
$(if $($(1)_DUMP_DERIVED_FN),$(if $(filter 1,$($(1)_DUMP_DERIVED_FN)),-Xclang -plugin-arg-clad -Xclang -fdump-derived-fn,),$(CLAD_PLUGIN_ARGS)) \
$($(1)_EXTRA_FLAGS)
endef

# =============================================================================
# Targets
# =============================================================================

.PHONY: all clean help validate debug release info list-targets $(TARGETS)
.PHONY: $(addsuffix -debug,$(TARGETS)) $(addsuffix -fast,$(TARGETS)) $(addsuffix -omp,$(TARGETS)) $(addsuffix -clad,$(TARGETS))
.DEFAULT_GOAL := help

# Main targets
all: validate $(TARGETS)

# Individual target rules
$(TARGETS): validate
	$(call log_info,Building target: $@)
	$(eval SOURCE_FILE := $(call get_source,$@))
	$(eval TARGET_FLAGS := $(call get_target_flags,$@))
	$(eval FINAL_FLAGS := $(BASE_FLAGS) $(CLAD_FLAGS) $(BUILD_FLAGS) $(PLATFORM_FLAGS) $(TARGET_FLAGS) $(EXTRA_FLAGS))
	@if [ ! -f "$(SOURCE_FILE)" ]; then \
		$(call log_error,Source file not found: $(SOURCE_FILE)); \
		exit 1; \
	fi
	@if [ "$(VERBOSE)" = "1" ]; then \
		printf "$(BLUE)[INFO]$(NC) Source: %s\n" "$(SOURCE_FILE)"; \
		printf "$(BLUE)[INFO]$(NC) Target: %s\n" "$@"; \
		printf "$(BLUE)[INFO]$(NC) Command: %s\n" "$(CXX) $(FINAL_FLAGS) $(SOURCE_FILE) -o $@"; \
		echo; \
	fi
	@$(CXX) $(FINAL_FLAGS) $(SOURCE_FILE) -o $@
	$(call log_success,Built $@ successfully!)
	@if command -v file >/dev/null 2>&1; then \
		printf "$(BLUE)[INFO]$(NC) File info: %s\n" "$$(file $@)"; \
	fi; \
	if [ -f "$@" ]; then \
		printf "$(BLUE)[INFO]$(NC) File size: %s\n" "$$(du -h $@ | cut -f1)"; \
	fi; \
	echo

# Debug variants
$(addsuffix -debug,$(TARGETS)): %-debug: validate
	@$(MAKE) BUILD_TYPE=debug $* --no-print-directory

# Fast variants (all optimizations)
$(addsuffix -fast,$(TARGETS)): %-fast: validate
	@$(MAKE) BUILD_TYPE=release USE_OMP=1 USE_ACCELERATE=1 USE_FAST_MATH=1 $* --no-print-directory

# OpenMP variants
$(addsuffix -omp,$(TARGETS)): %-omp: validate
	@$(MAKE) USE_OMP=1 $* --no-print-directory

# Clad debug variants
$(addsuffix -clad,$(TARGETS)): %-clad: validate
	@$(MAKE) DUMP_DERIVED_FN=1 VERBOSE=1 $* --no-print-directory

# Build shortcuts
debug:
	@$(MAKE) BUILD_TYPE=debug $(TARGETS) --no-print-directory

release:
	@$(MAKE) BUILD_TYPE=release $(TARGETS) --no-print-directory

fast:
	@$(MAKE) BUILD_TYPE=release USE_OMP=1 USE_ACCELERATE=1 USE_FAST_MATH=1 $(TARGETS) --no-print-directory

# Validate environment
validate:
	@if [ ! -x "$(CXX)" ]; then \
		printf "$(RED)[ERROR]$(NC) Compiler not found or not executable: %s\n" "$(CXX)"; \
		exit 1; \
	fi
	@if [ ! -f "$(CLAD_PLUGIN)" ]; then \
		printf "$(RED)[ERROR]$(NC) Clad plugin not found: %s\n" "$(CLAD_PLUGIN)"; \
		printf "$(YELLOW)[WARN]$(NC) Make sure clad is built in: %s\n" "$(CLAD_PATH)/$(CLAD_BUILD)/lib/"; \
		exit 1; \
	fi
	@if [ ! -d "$(CLAD_INCLUDE)" ]; then \
		printf "$(RED)[ERROR]$(NC) Clad include directory not found: %s\n" "$(CLAD_INCLUDE)"; \
		exit 1; \
	fi
	@if [ "$$(uname -s)" = "Darwin" ] && ! command -v xcrun >/dev/null 2>&1; then \
		printf "$(RED)[ERROR]$(NC) xcrun not found (required on macOS)\n"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@rm -f $(TARGETS)
	@rm -f $(addsuffix -debug,$(TARGETS))
	@rm -f $(addsuffix -fast,$(TARGETS))
	@rm -f $(addsuffix -omp,$(TARGETS))
	@rm -f $(addsuffix -clad,$(TARGETS))
	$(call log_info,Cleaned all build artifacts)

# List available targets
list-targets:
	@echo "Available targets:"
	@for target in $(TARGETS); do \
		echo "  $$target"; \
	done

# Display configuration information
info:
	@echo "=== Build Configuration ==="
	@echo "BUILD_TYPE:        $(BUILD_TYPE)"
	@echo "CXX:               $(CXX)"
	@echo "STD_VERSION:       $(STD_VERSION)"
	@echo "STDLIB:            $(STDLIB)"
	@echo ""
	@echo "=== Paths ==="
	@echo "LLVM_PATH:         $(LLVM_PATH)"
	@echo "CLAD_PATH:         $(CLAD_PATH)"
	@echo "CLAD_BUILD:        $(CLAD_BUILD)"
	@echo "CLAD_PLUGIN:       $(CLAD_PLUGIN)"
	@echo "CLAD_INCLUDE:      $(CLAD_INCLUDE)"
	@echo ""
	@echo "=== Global Features ==="
	@echo "USE_OMP:           $(USE_OMP)"
	@echo "USE_ACCELERATE:    $(USE_ACCELERATE)"
	@echo "USE_FAST_MATH:     $(USE_FAST_MATH)"
	@echo "DUMP_DERIVED_FN:   $(DUMP_DERIVED_FN)"
	@echo ""
	@echo "=== Available Targets ==="
	@for target in $(TARGETS); do \
		echo "  $$target (source: $(call get_source,$$target))"; \
	done
	@echo ""
	@echo "=== Flags ==="
	@echo "CXXFLAGS:          $(CXXFLAGS)"
	@echo "EXTRA_FLAGS:       $(EXTRA_FLAGS)"

# Help target
help:
	@echo "Usage: make [target] [variables...]"
	@echo ""
	@echo "=== Available Targets ==="
	@for target in $(TARGETS); do \
		echo "  $$target           Build $$target ($(call get_source,$$target))"; \
	done
	@echo ""
	@echo "=== Batch Targets ==="
	@echo "  all              Build all targets"
	@echo "  debug            Build all targets in debug mode"
	@echo "  release          Build all targets in release mode"
	@echo "  fast             Build all targets with all optimizations"
	@echo ""
	@echo "=== Target Variants ==="
	@echo "  <target>-debug   Build target in debug mode"
	@echo "  <target>-fast    Build target with all optimizations"
	@echo "  <target>-omp     Build target with OpenMP"
	@echo "  <target>-clad    Build target with clad debugging"
	@echo ""
	@echo "=== Utility Targets ==="
	@echo "  clean            Remove all build artifacts"
	@echo "  info             Display configuration"
	@echo "  list-targets     List all available targets"
	@echo "  validate         Validate build environment"
	@echo "  help             Show this help"
	@echo ""
	@echo "=== Variables ==="
	@echo "  BUILD_TYPE       debug|release (default: release)"
	@echo "  USE_OMP          0|1 Enable OpenMP (default: 0)"
	@echo "  USE_ACCELERATE   0|1 Enable Accelerate framework (default: 1)"
	@echo "  USE_FAST_MATH    0|1 Enable fast math (default: 1)"
	@echo "  DUMP_DERIVED_FN  0|1 Dump derived functions (default: 0)"
	@echo "  VERBOSE          0|1 Verbose output (default: 0)"
	@echo "  EXTRA_FLAGS      Additional compiler flags"
	@echo ""
	@echo "=== Per-Target Overrides ==="
	@echo "  <target>_USE_OMP        Override OpenMP for specific target"
	@echo "  <target>_USE_ACCELERATE Override Accelerate for specific target"
	@echo "  <target>_USE_FAST_MATH  Override fast math for specific target"
	@echo "  <target>_DUMP_DERIVED_FN Override clad dumping for specific target"
	@echo "  <target>_EXTRA_FLAGS    Additional flags for specific target"
	@echo ""
	@echo "=== Examples ==="
	@echo "  make train_llm                           # Build train_llm"
	@echo "  make inference                           # Build inference"
	@echo "  make all                                 # Build all targets"
	@echo "  make train_llm-fast                      # Build train_llm with optimizations"
	@echo "  make inference-debug                     # Build inference in debug mode"
	@echo "  make train_llm USE_OMP=1                 # Build train_llm with OpenMP"
	@echo "  make train_llm DUMP_DERIVED_FN=1 VERBOSE=1  # Build with clad debugging"
	@echo "  make train_llm_USE_OMP=1 inference_USE_OMP=0 train_llm inference"
	@echo "  make EXTRA_FLAGS=\"-DCUSTOM_FLAG\" train_llm"
	@echo ""
	@echo "=== Adding New Targets ==="
	@echo "To add a new target 'mytarget':"
	@echo "1. Add 'mytarget' to the TARGETS variable"
	@echo "2. Optionally set mytarget_SRC if source file isn't mytarget.cpp"
	@echo "3. Optionally set per-target flags (mytarget_USE_OMP, etc.)"