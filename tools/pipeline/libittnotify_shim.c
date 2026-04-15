// Shim to satisfy missing Intel ITT symbol required by some torch builds.
// This is a no-op implementation: it only exists so dynamic linking can succeed.
//
// The real ITT functionality (VTune/ITT) is not needed for correctness.

#ifdef __cplusplus
extern "C" {
#endif

// C variadic function needs at least one named parameter.
// We accept the first argument as an opaque pointer and ignore everything.
void iJIT_NotifyEvent(void* unused, ...) {
  // no-op
}

int iJIT_IsProfilingActive(void* unused, ...) {
  (void)unused;
  // Pretend profiling is disabled.
  return 0;
}

unsigned long iJIT_GetNewMethodID(void* unused, ...) {
  (void)unused;
  // Return a dummy method id.
  return 0;
}

#ifdef __cplusplus
}
#endif

