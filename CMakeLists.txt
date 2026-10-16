cmake_minimum_required(VERSION 3.18)
project(WhisperTranscriber VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFmpeg REQUIRED IMPORTED_TARGET libavformat>=59 libavcodec>=59 libavutil>=57 libswresample>=4)
find_package(CURL 7.85 REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(whisper CONFIG REQUIRED)
find_package(ggml CONFIG REQUIRED)

get_target_property(WHISPER_LIBRARY whisper IMPORTED_LOCATION)
get_filename_component(WHISPER_LIBRARY_DIR "${WHISPER_LIBRARY}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${FFmpeg_LIBRARY_DIRS};${WHISPER_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH "${FFmpeg_LIBRARY_DIRS};${WHISPER_LIBRARY_DIR}")
if(UNIX AND NOT APPLE)
    # DT_RPATH also resolves the private libraries' transitive dependencies.
    add_link_options(-Wl,--disable-new-dtags)
endif()

option(WHISPER_SERVICE_CUDA "Detect CUDA devices at runtime (Whisper must also have a CUDA backend)" ON)
if(WHISPER_SERVICE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()

add_library(transcriber_core STATIC src/audio.cpp src/model.cpp src/response.cpp)
target_include_directories(transcriber_core PUBLIC include)
target_link_libraries(transcriber_core PUBLIC PkgConfig::FFmpeg nlohmann_json::nlohmann_json whisper
    PRIVATE CURL::libcurl)
target_compile_options(transcriber_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(whisper_transcribe src/voice_transcribe.cpp)
target_link_libraries(whisper_transcribe PRIVATE transcriber_core whisper Threads::Threads)
target_compile_options(whisper_transcribe PRIVATE -Wall -Wextra -Wpedantic)
if(WHISPER_SERVICE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(whisper_transcribe PRIVATE CUDA::cudart ggml::ggml ggml::ggml-base)
    target_compile_definitions(whisper_transcribe PRIVATE WHISPER_SERVICE_HAS_CUDA=1)
endif()

# Keep private FFmpeg/Whisper libraries discoverable after installation.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(whisper_transcribe PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS whisper_transcribe RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(UNIX AND NOT APPLE)
    set(SERVICE_USER "whisper" CACHE STRING "Unprivileged systemd service user")
    set(SERVICE_HOME "/var/lib/whisper" CACHE PATH "Service home and model cache")
    set(SERVICE_MODEL "small" CACHE STRING "Default model for the systemd service")
    set(SERVICE_HOST "127.0.0.1" CACHE STRING "HTTP bind address for the systemd service")
    set(SERVICE_DEVICE "auto" CACHE STRING "Inference device for the systemd service")
    configure_file(whisper_transcribe.service.in whisper_transcribe.service @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/whisper_transcribe.service
        DESTINATION ${CMAKE_INSTALL_DATADIR}/whisper-transcriber)
endif()

if(BUILD_TESTING)
    add_executable(transcriber_tests tests/transcriber_tests.cpp)
    target_link_libraries(transcriber_tests PRIVATE transcriber_core Threads::Threads)
    add_test(NAME transcriber_unit COMMAND transcriber_tests)
endif()
