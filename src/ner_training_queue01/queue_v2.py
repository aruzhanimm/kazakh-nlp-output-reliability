"""Controller-only correction: keep recovery report stdout outside its own manifest."""
from pathlib import Path
from pilot.common import ROOT,read_json,file_hash
from .common import Q
from . import queue as original

unchanged_stage=original.stage
def stage(q,entry,name,module,args,directory,deadline,awake):
    if module=='encoder_lr_screen02_recovery01.report':
        # Report manifests its own directory before its terminal stdout print.
        # An active stdout file there would become stale immediately after sealing.
        # Output/temp files belong to this queue; scientific report destination is
        # still the original module's own constant, with no source alteration.
        directory=Q/'external_finalization'
    return unchanged_stage(q,entry,name,module,args,directory,deadline,awake)

if __name__=='__main__':
    lock=read_json(Q/'CONTROLLER_V2.lock.json')
    for name,h in lock['files'].items():assert file_hash(ROOT/name)==h
    original.stage=stage
    original.main()
