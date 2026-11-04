#pragma once

#include "Falcor.h"
#include <cmath>

namespace neural_controls
{
using Falcor::float3;
using Falcor::uint3;
using Falcor::Gui;
using Falcor::Properties;

/** Right-handed Euler rotation, applied in X, Y, Z order. Angles are in degrees. */
inline float3 rotateXYZ(float3 p, float3 degrees)
{
    constexpr float radiansPerDegree = 0.01745329251994329577f;
    const float3 a = degrees * radiansPerDegree;
    p = float3(p.x, std::cos(a.x) * p.y - std::sin(a.x) * p.z, std::sin(a.x) * p.y + std::cos(a.x) * p.z);
    p = float3(std::cos(a.y) * p.x + std::sin(a.y) * p.z, p.y, -std::sin(a.y) * p.x + std::cos(a.y) * p.z);
    return float3(std::cos(a.z) * p.x - std::sin(a.z) * p.y, std::sin(a.z) * p.x + std::cos(a.z) * p.y, p.z);
}

/** Deterministic motion evaluated at the Falcor scene time, never integrated per frame.
 * The caller owns the base transform and rebases it when editing motion controls.
 * Serialize this object's properties under the owning light's "motion" property.
 */
struct LightMotion
{
    bool autoRotate = false;
    uint3 rotationAxes = uint3(0, 0, 1);
    float3 rotationSpeed = float3(0.f, 0.f, 30.f);
    bool autoOrbit = false;
    uint3 orbitAxes = uint3(0, 0, 1);
    float3 orbitSpeed = float3(0.f, 0.f, 20.f);
    float3 orbitCenter = float3(0.f);
    double startTime = 0.0;

    static LightMotion fromProperties(const Properties& props)
    {
        LightMotion result;
        result.autoRotate = props.get<bool>("autoRotate", result.autoRotate);
        result.rotationAxes = props.get<uint3>("rotationAxes", result.rotationAxes);
        result.rotationSpeed = props.get<float3>("rotationSpeed", result.rotationSpeed);
        result.autoOrbit = props.get<bool>("autoOrbit", result.autoOrbit);
        result.orbitAxes = props.get<uint3>("orbitAxes", result.orbitAxes);
        result.orbitSpeed = props.get<float3>("orbitSpeed", result.orbitSpeed);
        result.orbitCenter = props.get<float3>("orbitCenter", result.orbitCenter);
        result.startTime = props.get<double>("startTime", result.startTime);
        result.validate();
        return result;
    }

    Properties toProperties() const
    {
        validate();
        Properties props;
        props["autoRotate"] = autoRotate;
        props["rotationAxes"] = rotationAxes;
        props["rotationSpeed"] = rotationSpeed;
        props["autoOrbit"] = autoOrbit;
        props["orbitAxes"] = orbitAxes;
        props["orbitSpeed"] = orbitSpeed;
        props["orbitCenter"] = orbitCenter;
        props["startTime"] = startTime;
        return props;
    }

    void validate() const
    {
        FALCOR_CHECK(std::isfinite(startTime), "Light motion startTime must be finite");
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            FALCOR_CHECK(rotationAxes[axis] <= 1 && orbitAxes[axis] <= 1, "Light motion axes must be 0 or 1");
            FALCOR_CHECK(std::isfinite(rotationSpeed[axis]) && std::isfinite(orbitSpeed[axis]), "Light motion speeds must be finite");
            FALCOR_CHECK(std::isfinite(orbitCenter[axis]), "Light motion orbit center must be finite");
        }
    }

    float3 spin(double time) const { return offset(autoRotate, rotationAxes, rotationSpeed, time); }
    float3 orbitOffset(double time) const { return offset(autoOrbit, orbitAxes, orbitSpeed, time); }
    float3 position(float3 base, double time) const
    {
        return autoOrbit ? orbitCenter + rotateXYZ(base - orbitCenter, orbitOffset(time)) : base;
    }
    float3 rotation(float3 base, double time) const { return base + spin(time); }

    bool renderUI(Gui::Widgets& widget, bool allowOrbit = true)
    {
        bool changed = widget.checkbox("Auto rotate", autoRotate);
        if (autoRotate)
        {
            const char* axisLabels[] = {"Rotate X", "Rotate Y", "Rotate Z"};
            const char* speedLabels[] = {"Rotation X (deg/s)", "Rotation Y (deg/s)", "Rotation Z (deg/s)"};
            changed |= renderAxes(widget, rotationAxes, rotationSpeed, axisLabels, speedLabels);
        }
        if (allowOrbit)
        {
            changed |= widget.checkbox("Auto orbit", autoOrbit);
            if (autoOrbit)
            {
                const char* axisLabels[] = {"Orbit X", "Orbit Y", "Orbit Z"};
                const char* speedLabels[] = {"Orbit X (deg/s)", "Orbit Y (deg/s)", "Orbit Z (deg/s)"};
                changed |= renderAxes(widget, orbitAxes, orbitSpeed, axisLabels, speedLabels);
                const float3 previous = orbitCenter;
                changed |= widget.var("Orbit center", orbitCenter, -1e6f, 1e6f, .01f);
                for (uint32_t axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(orbitCenter[axis])) orbitCenter[axis] = previous[axis];
            }
        }
        return changed;
    }

private:
    float3 offset(bool enabled, uint3 axes, float3 speed, double time) const
    {
        FALCOR_CHECK(std::isfinite(time) && std::isfinite(time - startTime), "Light motion time must be finite");
        float3 result(0.f);
        if (!enabled) return result;
        const double elapsed = time - startTime;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            if (!axes[axis] || speed[axis] == 0.f) continue;
            const double rate = double(speed[axis]);
            // Reduce time first to avoid overflowing speed * elapsed after a large timeline seek.
            result[axis] = float(std::fmod(rate * std::fmod(elapsed, 360.0 / std::abs(rate)), 360.0));
        }
        return result;
    }

    static bool renderAxes(Gui::Widgets& widget, uint3& axes, float3& speed, const char* const* axisLabels, const char* const* speedLabels)
    {
        bool changed = false;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            bool enabled = axes[axis] != 0;
            if (widget.checkbox(axisLabels[axis], enabled)) { axes[axis] = enabled ? 1u : 0u; changed = true; }
            const float previous = speed[axis];
            changed |= widget.var(speedLabels[axis], speed[axis], -3600.f, 3600.f, .1f);
            if (!std::isfinite(speed[axis])) speed[axis] = previous;
        }
        return changed;
    }
};
} // namespace neural_controls
