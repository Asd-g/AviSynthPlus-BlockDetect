#include <array>
#include <string>

#include "blockdetect.h"
#include "VCL2/instrset.h"

template <typename T, int range_size>
static const float calculate_blockiness(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept
{
    const size_t pitch{ avs_get_pitch_p(frame, plane) / sizeof(T) };
    const size_t width{ avs_get_row_size_p(frame, plane) / sizeof(T) };
    const int height{ avs_get_height_p(frame, plane) };
    const T* srcp{ reinterpret_cast<const T*>(avs_get_read_ptr_p(frame, plane)) };

    std::unique_ptr<float[]> grad{ std::make_unique<float[]>(width * height) };
    float ret{ 0.0f };

    // Calculate BS in horizontal and vertical directions according to (1)(2)(3).
    // Also try to find integer pixel periods (grids) even for scaled images.
    // In case of fractional periods, FFMAX of current and neighbor pixels
    // can help improve the correlation with MQS.
    // Skip linear correction term (4)(5), as it appears only valid for their own test samples.

    // horizontal blockiness (fixed width)
    for (int y{ 1 }; y < height; ++y)
    {
        for (int x{ 3 }; x < width - 4; ++x)
        {
            float temp{ 0.0f };
            grad[y * width + x] = std::abs(srcp[y * pitch + x + 0] - srcp[y * pitch + x + 1]);
            temp += std::abs(srcp[y * pitch + x + 1] - srcp[y * pitch + x + 2]);
            temp += std::abs(srcp[y * pitch + x + 2] - srcp[y * pitch + x + 3]);
            temp += std::abs(srcp[y * pitch + x + 3] - srcp[y * pitch + x + 4]);
            temp += std::abs(srcp[y * pitch + x - 0] - srcp[y * pitch + x - 1]);
            temp += std::abs(srcp[y * pitch + x - 1] - srcp[y * pitch + x - 2]);
            temp += std::abs(srcp[y * pitch + x - 2] - srcp[y * pitch + x - 3]);

            if (temp)
                grad[y * width + x] /= temp;
            else
                grad[y * width + x] /= range_size;

            // use first row to store acculated results
            grad[x] += grad[y * width + x];
        }
    }

    // find horizontal period
    for (int period{ d->period_min }; period < d->period_max + 1; ++period)
    {
        float temp;
        float block{ 0.0f };
        float nonblock{ 0.0f };
        int block_count{ 0 };
        int nonblock_count{ 0 };

        for (int x{ 3 }; x < width - 4; ++x)
        {
            if ((x % period) == (period - 1))
            {
                block += std::max(std::max(grad[x + 0], grad[x + 1]), grad[x - 1]);
                block_count++;
            }
            else
            {
                nonblock += grad[x];
                nonblock_count++;
            }
        }

        if (block_count && nonblock_count && nonblock)
        {
            temp = (block / block_count) / (nonblock / nonblock_count);
            ret = std::max(ret, temp);
        }
    }

    // vertical blockiness (fixed height)
    for (int y{ 3 }; y < height - 4; ++y)
    {
        for (int x{ 1 }; x < width; ++x)
        {
            float temp = 0.0f;
            grad[y * width + x] = std::abs(srcp[(y + 0) * pitch + x] - srcp[(y + 1) * pitch + x]);
            temp += std::abs(srcp[(y + 1) * pitch + x] - srcp[(y + 2) * pitch + x]);
            temp += std::abs(srcp[(y + 2) * pitch + x] - srcp[(y + 3) * pitch + x]);
            temp += std::abs(srcp[(y + 3) * pitch + x] - srcp[(y + 4) * pitch + x]);
            temp += std::abs(srcp[(y - 0) * pitch + x] - srcp[(y - 1) * pitch + x]);
            temp += std::abs(srcp[(y - 1) * pitch + x] - srcp[(y - 2) * pitch + x]);
            temp += std::abs(srcp[(y - 2) * pitch + x] - srcp[(y - 3) * pitch + x]);

            if (temp)
                grad[y * width + x] /= temp;
            else
                grad[y * width + x] /= range_size;

            // use first column to store accumulated results
            grad[y * width] += grad[y * width + x];
        }
    }

    // find vertical period
    for (int period{ d->period_min }; period < d->period_max + 1; ++period)
    {
        float temp;
        float block{ 0.0f };
        float nonblock{ 0.0f };
        int block_count{ 0 };
        int nonblock_count{ 0 };

        for (int y{ 3 }; y < height - 4; y++)
        {
            if ((y % period) == (period - 1))
            {
                block += std::max(std::max(grad[(y + 0) * width], grad[(y + 1) * width]), grad[(y - 1) * width]);
                block_count++;
            }
            else
            {
                nonblock += grad[y * width];
                nonblock_count++;
            }
        }

        if (block_count && nonblock_count && nonblock)
        {
            temp = (block / block_count) / (nonblock / nonblock_count);
            ret = std::max(ret, temp);
        }
    }

    // return highest value of horz||vert
    return ret;
}

static AVS_VideoFrame* AVSC_CC get_frame_blockdetect(AVS_FilterInfo* fi, int n)
{
    blockdetect* d{ reinterpret_cast<blockdetect*>(fi->user_data) };

    AVS_VideoFrame* frame{ avs_get_frame(fi->child, n) };
    if (!frame)
        return nullptr;

    avs_make_property_writable(fi->env, &frame);
    AVS_Map* props{ avs_get_frame_props_rw(fi->env, frame) };

    const std::array<std::string, 4> blockiness_y{ "blockiness_y", "blockiness_u", "blockiness_v", "blockiness_a" };
    const std::array<std::string, 4> blockiness_r{ "blockiness_r", "blockiness_g", "blockiness_b", "blockiness_a" };
    constexpr std::array<int, 4> planes_y{ AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V, AVS_PLANAR_A };
    constexpr std::array<int, 4> planes_r{ AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_A };

    const std::array<std::string, 4>* block;
    const std::array<int, 4>* planes;

    if (avs_is_rgb(&fi->vi))
    {
        block = &blockiness_r;
        planes = &planes_r;

    }
    else
    {
        block = &blockiness_y;
        planes = &planes_y;
    }

    for (int i{ 0 }; i < avs_num_components(&fi->vi); ++i)
    {
        if (d->process[i])
            avs_prop_set_float(fi->env, props, block->at(i).c_str(), d->calculate(frame, d, planes->at(i)), 0);
    }

    return frame;
}

static void AVSC_CC free_blockdetect(AVS_FilterInfo* fi)
{
    blockdetect* d{ reinterpret_cast<blockdetect*>(fi->user_data) };
    delete d;
}

static int AVSC_CC set_cache_hints_blockdetect(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 3 : 0;
}

static AVS_Value AVSC_CC Create_blockdetect(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum { Clip, Period_min, Period_max, Planes, Opt };

    blockdetect* d{ new blockdetect() };

    AVS_FilterInfo* fi;
    AVS_Clip* clip{ avs_new_c_filter(env, &fi, avs_array_elt(args, Clip), 1) };

    const auto set_error{ [&](const char* error)
        {
            avs_release_clip(clip);

            return avs_new_value_error(error);
        }
    };

    if (!avs_check_version(env, 9))
    {
        if (avs_check_version(env, 10))
        {
            if (avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX) < 2)
                return set_error("BlockDetect: AviSynth+ version must be r3688 or later.");
        }
    }
    else
        return set_error("BlockDetect: AviSynth+ version must be r3688 or later.");

    if (!avs_is_planar(&fi->vi))
        return set_error("BlockDetect: clip must be in planar format.");

    d->period_min = avs_defined(avs_array_elt(args, Period_min)) ? avs_as_int(avs_array_elt(args, Period_min)) : 3;
    d->period_max = avs_defined(avs_array_elt(args, Period_max)) ? avs_as_int(avs_array_elt(args, Period_max)) : 24;

    if (d->period_min <= 0)
        return set_error("BlockDetect: period_min must be greater than 0.");
    if (d->period_max <= 0)
        return set_error("BlockDetect: period_max must be greater than 0.");

    const int opt{ avs_defined(avs_array_elt(args, Opt)) ? avs_as_int(avs_array_elt(args, Opt)) : -1 };
    if (opt < -1 || opt > 3)
        return set_error("BlockDetect: opt must be between -1..3.");

    const int iset{ instrset_detect() };

    if (opt == 1 && iset < 2)
        return set_error("BlockDetect: opt=1 requires SSE2.");
    if (opt == 2 && iset < 8)
        return set_error("BlockDetect: opt=2 requires AVX2.");
    if (opt == 3 && iset < 10)
        return set_error("BlockDetect: opt=3 requires AVX512F.");

    const int num_planes{ (avs_defined(avs_array_elt(args, Planes))) ? avs_array_size(avs_array_elt(args, Planes)) : 0 };

    for (int i{ 0 }; i < 4; ++i)
        d->process[i] = (num_planes <= 0);

    for (int i{ 0 }; i < num_planes; ++i)
    {
        const int n{ avs_as_int(*(avs_as_array(avs_array_elt(args, Planes)) + i)) };

        if (n >= avs_num_components(&fi->vi))
            return set_error("BlockDetect: plane index out of range");

        if (d->process[n])
            return set_error("BlockDetect: plane specified twice");

        d->process[n] = true;
    }

    if ((opt == -1 && iset >= 10) || opt == 3)
    {
        switch (avs_component_size(&fi->vi))
        {
            case 1: d->calculate = calculate_blockiness_avx512<uint8_t, 256>; break;
            case 2:
            {
                switch (avs_bits_per_component(&fi->vi))
                {
                    case 10: d->calculate = calculate_blockiness_avx512<uint16_t, 1024>; break;
                    case 12: d->calculate = calculate_blockiness_avx512<uint16_t, 4096>; break;
                    case 14: d->calculate = calculate_blockiness_avx512<uint16_t, 16384>; break;
                    default: d->calculate = calculate_blockiness_avx512<uint16_t, 65536>;
                }
                break;
            }
            default: d->calculate = calculate_blockiness_avx512<float, 1>;
        }
    }
    else if ((opt == -1 && iset >= 8) || opt == 2)
    {
        switch (avs_component_size(&fi->vi))
        {
            case 1: d->calculate = calculate_blockiness_avx2<uint8_t, 256>; break;
            case 2:
            {
                switch (avs_bits_per_component(&fi->vi))
                {
                    case 10: d->calculate = calculate_blockiness_avx2<uint16_t, 1024>; break;
                    case 12: d->calculate = calculate_blockiness_avx2<uint16_t, 4096>; break;
                    case 14: d->calculate = calculate_blockiness_avx2<uint16_t, 16384>; break;
                    default: d->calculate = calculate_blockiness_avx2<uint16_t, 65536>;
                }
                break;
            }
            default: d->calculate = calculate_blockiness_avx2<float, 1>;
        }
    }
    else if ((opt == -1 && iset >= 2) || opt == 1)
    {
        switch (avs_component_size(&fi->vi))
        {
            case 1: d->calculate = calculate_blockiness_sse2<uint8_t, 256>; break;
            case 2:
            {
                switch (avs_bits_per_component(&fi->vi))
                {
                    case 10: d->calculate = calculate_blockiness_sse2<uint16_t, 1024>; break;
                    case 12: d->calculate = calculate_blockiness_sse2<uint16_t, 4096>; break;
                    case 14: d->calculate = calculate_blockiness_sse2<uint16_t, 16384>; break;
                    default: d->calculate = calculate_blockiness_sse2<uint16_t, 65536>;
                }
                break;
            }
            default: d->calculate = calculate_blockiness_sse2<float, 1>;
        }
    }
    else
    {
        switch (avs_component_size(&fi->vi))
        {
            case 1: d->calculate = calculate_blockiness<uint8_t, 256>; break;
            case 2:
            {
                switch (avs_bits_per_component(&fi->vi))
                {
                    case 10: d->calculate = calculate_blockiness<uint16_t, 1024>; break;
                    case 12: d->calculate = calculate_blockiness<uint16_t, 4096>; break;
                    case 14: d->calculate = calculate_blockiness<uint16_t, 16384>; break;
                    default: d->calculate = calculate_blockiness<uint16_t, 65536>;
                }
                break;
            }
            default: d->calculate = calculate_blockiness<float, 1>;
        }
    }

    AVS_Value v{ avs_new_value_clip(clip) };

    fi->user_data = reinterpret_cast<void*>(d);
    fi->get_frame = get_frame_blockdetect;
    fi->set_cache_hints = set_cache_hints_blockdetect;
    fi->free_filter = free_blockdetect;

    avs_release_clip(clip);

    return v;
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "BlockDetect", "c[period_min]i[period_max]i[planes]i[opt]i", Create_blockdetect, 0);
    return "BlockDetect";
}
