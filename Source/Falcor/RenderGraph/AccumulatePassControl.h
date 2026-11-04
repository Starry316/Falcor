#pragma once
#include <cstdint>

namespace Falcor
{
// Optional producer control for AccumulatePass. The producer writes uint32_t
// flags to its configured dictionary controlKey before AccumulatePass executes.
// Reset clears the previous estimate; Hold preserves it without adding a frame.
// An accumulator UI/reset/precision change requests a new producer sequence via
// the bool dictionary key controlKey + ".restart". The producer consumes it.
constexpr uint32_t kAccumulateReset = 1u;
constexpr uint32_t kAccumulateHold = 2u;
}
