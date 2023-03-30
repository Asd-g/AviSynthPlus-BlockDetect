#include "blockdetect.h"
#include "VCL2/vectorclass.h"

template <typename T, int range_size>
const float calculate_blockiness_sse2(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept
{
    const size_t pitch{ avs_get_pitch_p(frame, plane) / sizeof(T) };
    const size_t width{ avs_get_row_size_p(frame, plane) / sizeof(T) };
    const int height{ avs_get_height_p(frame, plane) };
    const T* srcp{ reinterpret_cast<const T*>(avs_get_read_ptr_p(frame, plane)) };

    std::unique_ptr<float[]> grad{ std::make_unique<float[]>(width * height) };
    float ret{ 0 };

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        // Calculate BS in horizontal and vertical directions according to (1)(2)(3).
        // Also try to find integer pixel periods (grids) even for scaled images.
        // In case of fractional periods, FFMAX of current and neighbor pixels
        // can help improve the correlation with MQS.
        // Skip linear correction term (4)(5), as it appears only valid for their own test samples.

        // horizontal blockiness (fixed width)
        for (int y{ 1 }; y < height; ++y)
        {
            for (int x{ 3 }; x < width - 4; x += 16)
            {
                Vec4i temp0{ zero_si128() };
                Vec4i temp1{ zero_si128() };
                Vec4i temp2{ zero_si128() };
                Vec4i temp3{ zero_si128() };

                auto grad_temp0{ to_float(abs(Vec4i().load_4uc(&srcp[y * pitch + x + 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 1]))) };
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 2]));
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 3]));
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 3]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4]));
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x - 0]) - Vec4i().load_4uc(&srcp[y * pitch + x - 1]));
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x - 1]) - Vec4i().load_4uc(&srcp[y * pitch + x - 2]));
                temp0 += abs(Vec4i().load_4uc(&srcp[y * pitch + x - 2]) - Vec4i().load_4uc(&srcp[y * pitch + x - 3]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / to_float(temp0), grad_temp0 / range_size);

                auto grad_temp1{ to_float(abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 1]))) };
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 2]));
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 3]));
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 3]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 1]));
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 2]));
                temp1 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 4 - 3]));
                grad_temp1 = select(temp1 > 0, grad_temp1 / to_float(temp1), grad_temp1 / range_size);

                auto grad_temp2{ to_float(abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 1]))) };
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 2]));
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 3]));
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 3]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 + 4]));
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 1]));
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 2]));
                temp2 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 8 - 3]));
                grad_temp2 = select(temp2 > 0, grad_temp2 / to_float(temp2), grad_temp2 / range_size);

                auto grad_temp3{ to_float(abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 1]))) };
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 2]));
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 3]));
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 3]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 + 4]));
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 0]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 1]));
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 1]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 2]));
                temp3 += abs(Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 2]) - Vec4i().load_4uc(&srcp[y * pitch + x + 12 - 3]));
                grad_temp3 = select(temp3 > 0, grad_temp3 / to_float(temp3), grad_temp3 / range_size);

                // use first row to store acculated results
                (Vec4f().load(&grad[x]) + grad_temp0).store(&grad[x]);
                (Vec4f().load(&grad[x + 4]) + grad_temp1).store(&grad[x + 4]);
                (Vec4f().load(&grad[x + 8]) + grad_temp2).store(&grad[x + 8]);
                (Vec4f().load(&grad[x + 12]) + grad_temp3).store(&grad[x + 12]);
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
            for (int x{ 1 }; x < width; x += 16)
            {
                Vec4i temp0{ zero_si128() };
                Vec4i temp1{ zero_si128() };
                Vec4i temp2{ zero_si128() };
                Vec4i temp3{ zero_si128() };

                auto grad_temp0{ to_float(abs(Vec4i().load_4uc(&srcp[(y + 0) * pitch + x]) - Vec4i().load_4uc(&srcp[(y + 1) * pitch + x]))) };
                temp0 += abs(Vec4i().load_4uc(&srcp[(y + 1) * pitch + x]) - Vec4i().load_4uc(&srcp[(y + 2) * pitch + x]));
                temp0 += abs(Vec4i().load_4uc(&srcp[(y + 2) * pitch + x]) - Vec4i().load_4uc(&srcp[(y + 3) * pitch + x]));
                temp0 += abs(Vec4i().load_4uc(&srcp[(y + 3) * pitch + x]) - Vec4i().load_4uc(&srcp[(y + 4) * pitch + x]));
                temp0 += abs(Vec4i().load_4uc(&srcp[(y - 0) * pitch + x]) - Vec4i().load_4uc(&srcp[(y - 1) * pitch + x]));
                temp0 += abs(Vec4i().load_4uc(&srcp[(y - 1) * pitch + x]) - Vec4i().load_4uc(&srcp[(y - 2) * pitch + x]));
                temp0 += abs(Vec4i().load_4uc(&srcp[(y - 2) * pitch + x]) - Vec4i().load_4uc(&srcp[(y - 3) * pitch + x]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / to_float(temp0), grad_temp0 / range_size);

                auto grad_temp1{ to_float(abs(Vec4i().load_4uc(&srcp[(y + 0) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 4]))) };
                temp1 += abs(Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y + 4) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[(y - 0) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 4]) - Vec4i().load_4uc(&srcp[(y - 3) * pitch + x + 4]));
                grad_temp1 = select(temp1 > 0, grad_temp1 / to_float(temp1), grad_temp1 / range_size);

                auto grad_temp2{ to_float(abs(Vec4i().load_4uc(&srcp[(y + 0) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 8]))) };
                temp2 += abs(Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 8]));
                temp2 += abs(Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 8]));
                temp2 += abs(Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y + 4) * pitch + x + 8]));
                temp2 += abs(Vec4i().load_4uc(&srcp[(y - 0) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 8]));
                temp2 += abs(Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 8]));
                temp2 += abs(Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 8]) - Vec4i().load_4uc(&srcp[(y - 3) * pitch + x + 8]));
                grad_temp2 = select(temp2 > 0, grad_temp2 / to_float(temp2), grad_temp2 / range_size);

                auto grad_temp3{ to_float(abs(Vec4i().load_4uc(&srcp[(y + 0) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 12]))) };
                temp3 += abs(Vec4i().load_4uc(&srcp[(y + 1) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 12]));
                temp3 += abs(Vec4i().load_4uc(&srcp[(y + 2) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 12]));
                temp3 += abs(Vec4i().load_4uc(&srcp[(y + 3) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y + 4) * pitch + x + 12]));
                temp3 += abs(Vec4i().load_4uc(&srcp[(y - 0) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 12]));
                temp3 += abs(Vec4i().load_4uc(&srcp[(y - 1) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 12]));
                temp3 += abs(Vec4i().load_4uc(&srcp[(y - 2) * pitch + x + 12]) - Vec4i().load_4uc(&srcp[(y - 3) * pitch + x + 12]));
                grad_temp3 = select(temp3 > 0, grad_temp3 / to_float(temp3), grad_temp3 / range_size);

                // use first column to store accumulated results
                (Vec4f().load(&grad[y * width]) + grad_temp0).store(&grad[y * width]);
                (Vec4f().load(&grad[y * width]) + grad_temp1).store(&grad[y * width]);
                (Vec4f().load(&grad[y * width]) + grad_temp2).store(&grad[y * width]);
                (Vec4f().load(&grad[y * width]) + grad_temp3).store(&grad[y * width]);
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
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        // Calculate BS in horizontal and vertical directions according to (1)(2)(3).
        // Also try to find integer pixel periods (grids) even for scaled images.
        // In case of fractional periods, FFMAX of current and neighbor pixels
        // can help improve the correlation with MQS.
        // Skip linear correction term (4)(5), as it appears only valid for their own test samples.

        // horizontal blockiness (fixed width)
        for (int y{ 1 }; y < height; ++y)
        {
            for (int x{ 3 }; x < width - 4; x += 8)
            {
                Vec4i temp0{ zero_si128() };
                Vec4i temp1{ zero_si128() };

                auto grad_temp0{ to_float(abs(Vec4i().load_4us(&srcp[y * pitch + x + 0]) - Vec4i().load_4us(&srcp[y * pitch + x + 1]))) };
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 1]) - Vec4i().load_4us(&srcp[y * pitch + x + 2]));
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 2]) - Vec4i().load_4us(&srcp[y * pitch + x + 3]));
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 3]) - Vec4i().load_4us(&srcp[y * pitch + x + 4]));
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x - 0]) - Vec4i().load_4us(&srcp[y * pitch + x - 1]));
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x - 1]) - Vec4i().load_4us(&srcp[y * pitch + x - 2]));
                temp0 += abs(Vec4i().load_4us(&srcp[y * pitch + x - 2]) - Vec4i().load_4us(&srcp[y * pitch + x - 3]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / to_float(temp0), grad_temp0 / range_size);

                auto grad_temp1{ to_float(abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 + 0]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 + 1]))) };
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 + 1]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 + 2]));
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 + 2]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 + 3]));
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 + 3]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 - 0]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 - 1]));
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 - 1]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 - 2]));
                temp1 += abs(Vec4i().load_4us(&srcp[y * pitch + x + 4 - 2]) - Vec4i().load_4us(&srcp[y * pitch + x + 4 - 3]));
                grad_temp1 = select(temp1 > 0, grad_temp1 / to_float(temp1), grad_temp1 / range_size);

                // use first row to store acculated results
                (Vec4f().load(&grad[x]) + grad_temp0).store(&grad[x]);
                (Vec4f().load(&grad[x + 4]) + grad_temp1).store(&grad[x + 4]);
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
            for (int x{ 1 }; x < width; x += 8)
            {
                Vec4i temp0{ zero_si128() };
                Vec4i temp1{ zero_si128() };

                auto grad_temp0{ to_float(abs(Vec4i().load_4us(&srcp[(y + 0) * pitch + x]) - Vec4i().load_4us(&srcp[(y + 1) * pitch + x]))) };
                temp0 += abs(Vec4i().load_4us(&srcp[(y + 1) * pitch + x]) - Vec4i().load_4us(&srcp[(y + 2) * pitch + x]));
                temp0 += abs(Vec4i().load_4us(&srcp[(y + 2) * pitch + x]) - Vec4i().load_4us(&srcp[(y + 3) * pitch + x]));
                temp0 += abs(Vec4i().load_4us(&srcp[(y + 3) * pitch + x]) - Vec4i().load_4us(&srcp[(y + 4) * pitch + x]));
                temp0 += abs(Vec4i().load_4us(&srcp[(y - 0) * pitch + x]) - Vec4i().load_4us(&srcp[(y - 1) * pitch + x]));
                temp0 += abs(Vec4i().load_4us(&srcp[(y - 1) * pitch + x]) - Vec4i().load_4us(&srcp[(y - 2) * pitch + x]));
                temp0 += abs(Vec4i().load_4us(&srcp[(y - 2) * pitch + x]) - Vec4i().load_4us(&srcp[(y - 3) * pitch + x]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / to_float(temp0), grad_temp0 / range_size);

                auto grad_temp1{ to_float(abs(Vec4i().load_4us(&srcp[(y + 0) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y + 1) * pitch + x + 4]))) };
                temp1 += abs(Vec4i().load_4us(&srcp[(y + 1) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y + 2) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[(y + 2) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y + 3) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[(y + 3) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y + 4) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[(y - 0) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y - 1) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[(y - 1) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y - 2) * pitch + x + 4]));
                temp1 += abs(Vec4i().load_4us(&srcp[(y - 2) * pitch + x + 4]) - Vec4i().load_4us(&srcp[(y - 3) * pitch + x + 4]));
                grad_temp1 = select(temp1 > 0, grad_temp1 / to_float(temp1), grad_temp1 / range_size);

                // use first column to store accumulated results
                (Vec4f().load(&grad[y * width]) + grad_temp0).store(&grad[y * width]);
                (Vec4f().load(&grad[y * width]) + grad_temp1).store(&grad[y * width]);
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
    }
    else
    {
        // Calculate BS in horizontal and vertical directions according to (1)(2)(3).
        // Also try to find integer pixel periods (grids) even for scaled images.
        // In case of fractional periods, FFMAX of current and neighbor pixels
        // can help improve the correlation with MQS.
        // Skip linear correction term (4)(5), as it appears only valid for their own test samples.

        // horizontal blockiness (fixed width)
        for (int y{ 1 }; y < height; ++y)
        {
            for (int x{ 3 }; x < width - 4; x += 4)
            {
                Vec4f temp0{ zero_4f() };

                auto grad_temp0{ abs(Vec4f().load(&srcp[y * pitch + x + 0]) - Vec4f().load(&srcp[y * pitch + x + 1])) };
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x + 1]) - Vec4f().load(&srcp[y * pitch + x + 2]));
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x + 2]) - Vec4f().load(&srcp[y * pitch + x + 3]));
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x + 3]) - Vec4f().load(&srcp[y * pitch + x + 4]));
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x - 0]) - Vec4f().load(&srcp[y * pitch + x - 1]));
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x - 1]) - Vec4f().load(&srcp[y * pitch + x - 2]));
                temp0 += abs(Vec4f().load(&srcp[y * pitch + x - 2]) - Vec4f().load(&srcp[y * pitch + x - 3]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / temp0, grad_temp0);

                // use first row to store acculated results
                (Vec4f().load(&grad[x]) + grad_temp0).store(&grad[x]);
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
            for (int x{ 1 }; x < width; x += 4)
            {
                Vec4f temp0{ zero_4f() };

                auto grad_temp0{ abs(Vec4f().load(&srcp[(y + 0) * pitch + x]) - Vec4f().load(&srcp[(y + 1) * pitch + x])) };
                temp0 += abs(Vec4f().load(&srcp[(y + 1) * pitch + x]) - Vec4f().load(&srcp[(y + 2) * pitch + x]));
                temp0 += abs(Vec4f().load(&srcp[(y + 2) * pitch + x]) - Vec4f().load(&srcp[(y + 3) * pitch + x]));
                temp0 += abs(Vec4f().load(&srcp[(y + 3) * pitch + x]) - Vec4f().load(&srcp[(y + 4) * pitch + x]));
                temp0 += abs(Vec4f().load(&srcp[(y - 0) * pitch + x]) - Vec4f().load(&srcp[(y - 1) * pitch + x]));
                temp0 += abs(Vec4f().load(&srcp[(y - 1) * pitch + x]) - Vec4f().load(&srcp[(y - 2) * pitch + x]));
                temp0 += abs(Vec4f().load(&srcp[(y - 2) * pitch + x]) - Vec4f().load(&srcp[(y - 3) * pitch + x]));
                grad_temp0 = select(temp0 > 0, grad_temp0 / temp0, grad_temp0);

                // use first column to store accumulated results
                (Vec4f().load(&grad[y * width]) + grad_temp0).store(&grad[y * width]);
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
    }

    // return highest value of horz||vert
    return ret;
}

template const float calculate_blockiness_sse2<uint8_t, 256>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template const float calculate_blockiness_sse2<uint16_t, 1024>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template const float calculate_blockiness_sse2<uint16_t, 4096>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template const float calculate_blockiness_sse2<uint16_t, 16384>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template const float calculate_blockiness_sse2<uint16_t, 65536>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
template const float calculate_blockiness_sse2<float, 1>(AVS_VideoFrame* frame, const blockdetect* d, const int plane) noexcept;
