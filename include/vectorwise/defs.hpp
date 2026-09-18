#pragma once

namespace vectorwise {
/// type to mark position in a vectorwise vector
#ifdef VW_POS_16
using pos_t = uint16_t;
#else
using pos_t = uint32_t;
#endif

#define RES __restrict__

/// Marker that causes noinline in case separation of interpretation code and
/// code that does actual work is required.
#ifndef INTERPRET_SEPARATE
#define INTERPRET_SEPARATE __attribute__((noinline))
#endif
}
