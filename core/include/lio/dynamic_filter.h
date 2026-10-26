// Lightweight online dynamic-point filter (M-detector / DUFOMap inspired).
// Builds a spherical range image of the previous body-frame scan; points in the
// current scan that are much closer along the same ray than the previous return
// are treated as dynamic (appeared in front of static structure) and dropped
// before map insertion. Functions stay under 100 lines.
#pragma once
#include <cmath>
#include <vector>
#include "lio/types.h"

namespace lio
{
struct DynamicFilterParams
{
    bool enable = false;
    int azimuth_bins = 360;
    int elev_bins = 60;
    double elev_min = -30.0 * M_PI / 180.0; // rad
    double elev_max = 30.0 * M_PI / 180.0;
    double appear_thresh = 0.8; // m: current much closer than previous => dynamic
    double max_range = 80.0;
};

class DynamicFilter
{
public:
    explicit DynamicFilter(const DynamicFilterParams &p) : p_(p)
    {
        range_.assign(size_t(p_.azimuth_bins) * size_t(p_.elev_bins), p_.max_range + 1.0);
    }

    // Drop points that appeared suddenly in front of the previous range image.
    // Updates the range image with the (filtered) cloud for the next call.
    PointCloud::Ptr filter(const PointCloud &body)
    {
        PointCloud::Ptr out(new PointCloud);
        if (!p_.enable || body.empty())
        {
            if (!body.empty())
                rebuild(body);
            *out = body;
            return out;
        }
        out->reserve(body.size());
        size_t kept = 0;
        for (const auto &pt : body.points)
        {
            int a, e;
            const double r = ray(pt.x, pt.y, pt.z, a, e);
            if (r < 0)
                continue;
            const double prev = range_[idx(a, e)];
            if (prev <= p_.max_range && r + p_.appear_thresh < prev)
            {
                dropped_++;
                continue; // appeared in front of a previously farther return
            }
            out->push_back(pt);
            kept++;
        }
        rebuild(*out);
        return out;
    }

    size_t dropped() const { return dropped_; }

private:
    DynamicFilterParams p_;
    std::vector<double> range_;
    size_t dropped_ = 0;

    size_t idx(int a, int e) const { return size_t(e) * size_t(p_.azimuth_bins) + size_t(a); }

    double ray(float x, float y, float z, int &a, int &e) const
    {
        const double r = std::sqrt(double(x) * x + double(y) * y + double(z) * z);
        if (r < 1e-3 || r > p_.max_range)
            return -1;
        double az = std::atan2(double(y), double(x)); // [-pi, pi]
        if (az < 0)
            az += 2 * M_PI;
        const double el = std::asin(double(z) / r);
        a = int(az / (2 * M_PI) * p_.azimuth_bins) % p_.azimuth_bins;
        if (el < p_.elev_min || el > p_.elev_max)
            return -1;
        e = int((el - p_.elev_min) / (p_.elev_max - p_.elev_min) * (p_.elev_bins - 1));
        e = std::max(0, std::min(p_.elev_bins - 1, e));
        return r;
    }

    void rebuild(const PointCloud &body)
    {
        std::fill(range_.begin(), range_.end(), p_.max_range + 1.0);
        for (const auto &pt : body.points)
        {
            int a, e;
            const double r = ray(pt.x, pt.y, pt.z, a, e);
            if (r < 0)
                continue;
            double &cell = range_[idx(a, e)];
            if (r < cell)
                cell = r;
        }
    }
};
} // namespace lio
