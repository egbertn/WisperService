cmake_minimum_required(VERSION 3.10)
project(WhisperTranscriber)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify compile options
add_compile_options(-O2 -pthread)

# Find threading package
find_package(Threads REQUIRED)

# Use pkg-config to find FFMPEG, and CURL
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFmpeg REQUIRED libavformat libavcodec libavutil libswresample)
pkg_check_modules(CURL REQUIRED libcurl)

# Detect CUDA and set up optional linking
find_package(CUDA QUIET)

if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_LIBS ${CUDA_CUDART_LIBRARY})
else()
    message(STATUS "CUDA not found. Compiling without GPU support.")
    set(CUDA_LIBS "")
endif()

# Include directories from pkg-config
include_directories(${FFmpeg_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

# Link directories from pkg-config
link_directories(${FFmpeg_LIBRARY_DIRS})
link_directories(${CURL_LIBRARY_DIRS})

# Link libraries from pkg-config
set(FFmpeg_LIBS ${FFmpeg_LIBRARIES})
set(CURL_LIBS ${CURL_LIBRARIES})

# Specify the path to the pre-built Whisper library and include files
set(WHISPER_LIB_PATH "/usr/local/lib")  # Update this path if necessary
set(WHISPER_INCLUDE_PATH "/usr/local/include")  # Update this path if necessary

# Include directories
include_directories(${WHISPER_INCLUDE_PATH})

# Include your project's include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add the executable
add_executable(whisper_transcribe src/voice_transcribe.cpp)

# Link libraries (optioneel CUDA)
target_link_libraries(whisper_transcribe
    ${FFmpeg_LIBS}
    ${CURL_LIBS}
    ${CUDA_LIBS}
    Threads::Threads
    whisper    
)

# Install the executable
install(TARGETS whisper_transcribe DESTINATION /usr/local/bin)

# Show CUDA status
if(CUDA_FOUND)
    message(STATUS "CUDA support enabled. Linking against ${CUDA_LIBS}")
else()
    message(STATUS "CUDA support disabled. CPU-only build.")
endif()

# Show linked libraries
message(STATUS "FFmpeg include dirs: ${FFmpeg_INCLUDE_DIRS}")
message(STATUS "FFmpeg library dirs: ${FFmpeg_LIBRARY_DIRS}")
message(STATUS "FFmpeg libraries: ${FFmpeg_LIBRARIES}")

message(STATUS "CURL include dirs: ${CURL_INCLUDE_DIRS}")
message(STATUS "CURL library dirs: ${CURL_LIBRARY_DIRS}")
message(STATUS "CURL libraries: ${CURL_LIBRARIES}")

# Install the service (Linux)
if(UNIX AND NOT APPLE)
    configure_file(${CMAKE_SOURCE_DIR}/whisper_transcribe.service.in
                   ${CMAKE_BINARY_DIR}/whisper_transcribe.service @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/whisper_transcribe.service
            DESTINATION /etc/systemd/system)
    install(CODE "execute_process(COMMAND systemctl enable whisper_transcribe.service)")
    install(CODE "execute_process(COMMAND systemctl start whisper_transcribe.service)")
endif()

# Install the service (Windows)
if(WIN32)
    install(TARGETS whisper_transcribe DESTINATION C:/whisper_service)

    # Add custom command to create the service
    install(CODE "execute_process(COMMAND sc create WhisperService binPath= C:/whisper_service/whisper_transcribe.exe start= auto)")
    install(CODE "execute_process(COMMAND sc start WhisperService)")
endif()
