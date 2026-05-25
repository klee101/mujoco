#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: mjbatch/scripts/build.sh [options]

Configure and build the mjbatch targets from the repository root.

Options:
  --debug             Configure a Debug build.
  --release           Configure a Release build. This is the default.
  --configure-only    Run CMake configure only.
  --build-only        Run CMake build only.
  --target NAME       Build one target. Can be passed multiple times.
  -h, --help          Show this help.

Environment overrides:
  BUILD_DIR           Build directory. Default: <repo>/build
  BUILD_TYPE          CMake build type. Default: Release
  PYTHON_EXECUTABLE   Python interpreter passed to CMake.
  C_COMPILER          C compiler. Default: gcc-11 when available.
  CXX_COMPILER        C++ compiler. Default: g++-11 when available.
  TARGETS             Space-separated target list. Default: "mjb CompileShaders"
  PARALLEL            Parallel build jobs. Default: CMake default.
  EXTRA_CMAKE_ARGS    Extra CMake configure arguments.

Examples:
  mjbatch/scripts/build.sh
  BUILD_TYPE=Debug mjbatch/scripts/build.sh --target mjb
  EXTRA_CMAKE_ARGS="-DGLFW_BUILD_X11=ON" mjbatch/scripts/build.sh
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${repo_root}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

run_configure=1
run_build=1
cli_targets=()

while (($#)); do
  case "$1" in
    --debug)
      BUILD_TYPE=Debug
      shift
      ;;
    --release)
      BUILD_TYPE=Release
      shift
      ;;
    --configure-only)
      run_build=0
      shift
      ;;
    --build-only)
      run_configure=0
      shift
      ;;
    --target)
      if (($# < 2)); then
        echo "error: --target requires a target name" >&2
        exit 2
      fi
      cli_targets+=("$2")
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXECUTABLE="$(command -v python3)"
  else
    echo "error: python or python3 was not found; set PYTHON_EXECUTABLE" >&2
    exit 1
  fi
fi

if [[ -z "${C_COMPILER:-}" ]] && command -v gcc-11 >/dev/null 2>&1; then
  C_COMPILER="gcc-11"
fi

if [[ -z "${CXX_COMPILER:-}" ]] && command -v g++-11 >/dev/null 2>&1; then
  CXX_COMPILER="g++-11"
fi

if [[ -z "${CMAKE_INTERPROCEDURAL_OPTIMIZATION+x}" ]]; then
  if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
  else
    CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
  fi
fi

if [[ -z "${CMAKE_CXX_FLAGS+x}" && "${BUILD_TYPE}" != "Debug" ]]; then
  CMAKE_CXX_FLAGS="-march=native -O3 -funroll-loops"
fi

cmake_args=(
  -S "${repo_root}"
  -B "${BUILD_DIR}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=${CMAKE_INTERPROCEDURAL_OPTIMIZATION}"
  "-DMUJOCO_ENABLE_AVX=${MUJOCO_ENABLE_AVX:-ON}"
  "-DMUJOCO_ENABLE_AVX_INTRINSICS=${MUJOCO_ENABLE_AVX_INTRINSICS:-ON}"
  "-DGLFW_BUILD_WAYLAND=${GLFW_BUILD_WAYLAND:-OFF}"
  "-DGLFW_BUILD_X11=${GLFW_BUILD_X11:-OFF}"
  "-DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}"
)

if [[ -n "${C_COMPILER:-}" ]]; then
  cmake_args+=("-DCMAKE_C_COMPILER=${C_COMPILER}")
fi

if [[ -n "${CXX_COMPILER:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
fi

if [[ -n "${CMAKE_CXX_FLAGS:-}" ]]; then
  cmake_args+=("-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
fi

if [[ -n "${EXTRA_CMAKE_ARGS:-}" ]]; then
  read -r -a extra_cmake_args <<< "${EXTRA_CMAKE_ARGS}"
  cmake_args+=("${extra_cmake_args[@]}")
fi

if ((${#cli_targets[@]})); then
  build_targets=("${cli_targets[@]}")
else
  read -r -a build_targets <<< "${TARGETS:-mjb CompileShaders}"
fi

if ((run_configure)); then
  cmake "${cmake_args[@]}"
fi

if ((run_build)); then
  build_args=(--build "${BUILD_DIR}" --config "${BUILD_TYPE}")
  if [[ -n "${PARALLEL:-}" ]]; then
    build_args+=(--parallel "${PARALLEL}")
  else
    build_args+=(--parallel)
  fi

  for target in "${build_targets[@]}"; do
    cmake "${build_args[@]}" --target "${target}"
  done
fi
