## Description

Determines blockiness of frames by adding the relevant frame property.

Based on Remco Muijs and Ihor Kirenko: "A no-reference blocking artifact measure for adaptive video processing." 2005 13th European signal processing conference. (http://www.eurasip.org/Proceedings/Eusipco/Eusipco2005/defevent/papers/cr1042.pdf)

This is [a port of the FFmpeg filter blockdetect](https://ffmpeg.org/ffmpeg-filters.html#blockdetect-1).

### Requirements:

- AviSynth+ r3688 (can be downloaded from [here](https://gitlab.com/uvz/AviSynthPlus-Builds) until official release is uploaded) or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
BlockDetect(clip input, int "period_min", int "period_max", int[] "planes", int "opt")
```

### Parameters:

- input\
    A clip to process.\
    It must be in 8..32-bit planar format.

- period_min, period_max\
    Set minimum and maximum values for determining pixel grids (periods).\
    period_min must be between 2..32.\
    period_max must be between 2..64.\
    Default: period_min = 3, period_max = 24.

- planes\
    Sets which planes will be processed.\
    There will be new frame property `blockiness_...` for every processed plane.\
    Default: [0, 1, 2, 3].

- opt\
    Sets which cpu optimizations to use.\
    -1: Auto-detect.\
    0: Use C++ code.\
    1: Use SSE2 code.\
    2: Use AVX2 code.\
    3: Use AVX-512 code.\
    Default: -1.

### Building:

- Windows\
    Use solution files.

- Linux
    ```
    Requirements:
        - Git
        - C++17 compiler
        - CMake >= 3.16
    ```
    ```
    git clone https://github.com/Asd-g/AviSynthPlus-BlockDetect && \
    cd AviSynthPlus-BlockDetect && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install
    ```
