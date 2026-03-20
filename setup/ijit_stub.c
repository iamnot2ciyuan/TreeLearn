/* Stub for Intel ITT JIT symbols required by PyTorch when MKL 2024.0 is not available.
 * Compile: gcc -shared -fPIC -o libijit_stub.so ijit_stub.c
 * Use: LD_PRELOAD=/path/to/libijit_stub.so python -c "import torch"
 */

/* iJIT_NotifyEvent(int event_type, void *EventSpecificData) */
int iJIT_NotifyEvent(int event_type, void *EventSpecificData) {
    (void)event_type;
    (void)EventSpecificData;
    return 1;
}

/* iJIT_IsProfilingActive(void) returns enum, use 0 = iJIT_NOTHING_RUNNING */
int iJIT_IsProfilingActive(void) {
    return 0;
}

/* iJIT_GetNewMethodID(void) returns unsigned int */
unsigned int iJIT_GetNewMethodID(void) {
    return 0;
}

/* void FinalizeThread(void) */
void FinalizeThread(void) {
}

/* void FinalizeProcess(void) */
void FinalizeProcess(void) {
}

/* void iJIT_RegisterCallbackEx(void *userdata, void *NewModeCallBackFuncEx) */
void iJIT_RegisterCallbackEx(void *userdata, void *NewModeCallBackFuncEx) {
    (void)userdata;
    (void)NewModeCallBackFuncEx;
}
