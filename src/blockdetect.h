#pragma once

#include <cmath>
#include <memory>

#include "avisynth_c.h"

struct blockdetect
{
    int period_min;
    int period_max;
    bool process[4];

    const float (*calculate)(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
};

template <typename T, int range_size>
const float calculate_blockiness_sse2(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template <typename T, int range_size>
const float calculate_blockiness_avx2(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template <typename T, int range_size>
const float calculate_blockiness_avx512(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
