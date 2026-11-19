"""Temporary Windows system-required request; kernel acknowledgments are audited."""
import ctypes as C
from ctypes import wintypes as W
from pilot.common import write_json
from .common import utc

class Detailed(C.Structure):
    _fields_=[('module',W.HMODULE),('id',W.ULONG),('count',W.ULONG),('strings',C.POINTER(W.LPWSTR))]
class Reason(C.Union):_fields_=[('simple',W.LPWSTR),('detailed',Detailed)]
class Context(C.Structure):_fields_=[('version',W.ULONG),('flags',W.DWORD),('reason',Reason)]

class Awake:
    def __init__(self,path):self.path=path;self.handle=None;self.active=False;self.records=[]
    def __enter__(self):
        self.k=C.WinDLL('kernel32',use_last_error=True)
        self.k.PowerCreateRequest.argtypes=[C.POINTER(Context)];self.k.PowerCreateRequest.restype=W.HANDLE
        self.k.PowerSetRequest.argtypes=[W.HANDLE,C.c_int];self.k.PowerSetRequest.restype=W.BOOL
        self.k.PowerClearRequest.argtypes=[W.HANDLE,C.c_int];self.k.PowerClearRequest.restype=W.BOOL
        self.k.CloseHandle.argtypes=[W.HANDLE];self.k.CloseHandle.restype=W.BOOL
        self.k.GetHandleInformation.argtypes=[W.HANDLE,C.POINTER(W.DWORD)];self.k.GetHandleInformation.restype=W.BOOL
        self.k.SetThreadExecutionState.argtypes=[W.DWORD];self.k.SetThreadExecutionState.restype=W.DWORD
        self.context=Context(0,1,Reason(simple='Kazakh NLP encoder_cont01 bounded continuation; shutdown disabled'))
        self.handle=self.k.PowerCreateRequest(C.byref(self.context))
        if not self.handle or self.handle==C.c_void_p(-1).value:raise C.WinError(C.get_last_error())
        if not self.k.PowerSetRequest(self.handle,1):
            self.k.CloseHandle(self.handle);self.handle=None;raise C.WinError(C.get_last_error())
        self.active=True
        try:self.verify('acquired')
        except BaseException:self.__exit__(None,None,None);raise
        return self
    def verify(self,event='held'):
        flags=W.DWORD()
        valid=bool(self.k.GetHandleInformation(self.handle,C.byref(flags)))
        execution=int(self.k.SetThreadExecutionState(0x80000001))
        entry={'utc':utc(),'event':event,'system_request_set_succeeded':self.active,'request_handle_valid':valid,'thread_request_nonzero_return':execution,'shutdown_enabled':False}
        self.records.append(entry);write_json(self.path,self.records)
        assert self.active and valid and execution!=0,'System-awake request verification failed'
    def __exit__(self,*unused):
        cleared=bool(self.k.PowerClearRequest(self.handle,1)) if self.active else True
        closed=bool(self.k.CloseHandle(self.handle)) if self.handle else True
        previous=int(self.k.SetThreadExecutionState(0x80000000));self.active=False;self.handle=None
        self.records.append({'utc':utc(),'event':'released','power_clear_succeeded':cleared,'handle_close_succeeded':closed,'thread_request_reset_nonzero_return':previous,'shutdown_enabled':False});write_json(self.path,self.records)
        assert cleared and closed and previous!=0,'Temporary awake request cleanup failed'

if __name__=='__main__':
    from .common import D
    with Awake(D/'audit/awake_probe.json') as request:request.verify('finite_preflight')
    print('Temporary awake request acquisition, verification and release passed.')
