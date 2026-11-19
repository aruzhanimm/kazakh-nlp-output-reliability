"""Capture EXACT bytes objects passed to SHA-256.update, preserving the wire format."""
import hashlib,json,os
from pathlib import Path
from pilot.common import file_hash,read_json
from encoder_lr_screen02_diag01.evidence import durable
from .common import COMPONENTS,utc

class Recorder:
    def __init__(self,base,hash_factory=hashlib.sha256):
        self.base=Path(base);self.base.parent.mkdir(parents=True,exist_ok=True)
        self.data=self.base.with_suffix('.bin').open('xb');self.index=self.base.with_suffix('.index.jsonl').open('x',encoding='utf-8',newline='\n')
        self.h=hash_factory();self.offset=0;self.ordinal=0
    def emit(self,buffer,path,part,**extra):
        assert type(buffer) is bytes,'Hash input must be an immutable bytes object'
        row={'ordinal':self.ordinal,'offset':self.offset,'length':len(buffer),'path':path,'part':part,
            'buffer_sha256':hashlib.sha256(buffer).hexdigest(),**extra}
        self.index.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        assert self.data.write(buffer)==len(buffer)
        # The exact SAME immutable object written above is passed here.
        self.h.update(buffer);self.offset+=len(buffer);self.ordinal+=1
    def close(self):
        for f in [self.data,self.index]:f.flush();os.fsync(f.fileno());f.close()

def fingerprint_stream(value,base,root='',hash_factory=hashlib.sha256):
    import torch,numpy as np
    r=Recorder(base,hash_factory)
    def visit(x,path):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous()
            r.emit(b'tensor',path,'tensor_tag');r.emit(str((a.dtype,tuple(a.shape))).encode(),path,'tensor_metadata')
            r.emit(a.reshape(-1).view(torch.uint8).numpy().tobytes(),path,'tensor_data',dtype=str(a.dtype),shape=list(a.shape),itemsize=a.element_size())
        elif isinstance(x,np.ndarray):
            r.emit(b'ndarray',path,'ndarray_tag');r.emit(str((x.dtype,x.shape)).encode(),path,'ndarray_metadata')
            r.emit(x.tobytes(),path,'ndarray_data',dtype=str(x.dtype),shape=list(x.shape),itemsize=x.dtype.itemsize)
        elif isinstance(x,dict):
            r.emit(b'dict',path,'dict_tag')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(k,path+'{key:'+repr(k)+'}');visit(x[k],path+'['+repr(k)+']')
        elif isinstance(x,(tuple,list)):
            r.emit(type(x).__name__.encode(),path,'sequence_tag')
            for i,item in enumerate(x):visit(item,path+f'[{i}]')
        else:r.emit(json.dumps([type(x).__name__,x],sort_keys=True).encode(),path,'scalar',python_type=type(x).__name__)
    error=None
    try:visit(value,root)
    except BaseException as e:error={'type':type(e).__name__,'message':str(e)};raise
    finally:
        r.close();durable(Path(base).with_suffix('.stream.json'),{'utc':utc(),'digest_observed_during_hash_updates':r.h.hexdigest(),
            'bytes':r.offset,'hash_update_calls':r.ordinal,'error':error,'format':'unchanged stage_preflight.state.fingerprint',
            'exact_immutable_buffers_captured_during_hashing':True,'index_sha256':file_hash(Path(base).with_suffix('.index.jsonl'))})
    return r.h.hexdigest()

def trace_components(state,folder,stage,sidecar=None):
    folder=Path(folder);target=folder/'streams'/stage;target.mkdir(parents=True,exist_ok=False)
    actual={};error=None
    try:
        for name in COMPONENTS:actual[name]=fingerprint_stream(state[name],target/name,name)
    except BaseException as e:error={'type':type(e).__name__,'message':str(e)};raise
    finally:
        embedded=state.get('state_digests');expected=sidecar.get('state_digests') if sidecar is not None else None
        differing=[k for k in COMPONENTS if k not in actual or (embedded is not None and actual[k]!=embedded.get(k)) or (expected is not None and actual[k]!=expected.get(k))]
        durable(folder/(stage+'.signatures.json'),{'utc':utc(),'stage':stage,'actual':actual,'embedded':embedded,'sidecar':expected,
            'every_differing_component':differing,'hash_error':error,'recorded_before_assertions':True})
    return actual

def rehash_stream(base,output):
    """Separate reader: independent file digest and update-boundary reconstruction."""
    base=Path(base);meta=read_json(base.with_suffix('.stream.json'));offset=0;ordinal=0;bad=[];boundary=hashlib.sha256()
    with base.with_suffix('.bin').open('rb') as data,base.with_suffix('.index.jsonl').open(encoding='utf-8') as idx:
        for line in idx:
            row=json.loads(line);assert row['offset']==offset and row['ordinal']==ordinal
            b=data.read(row['length']);h=hashlib.sha256(b).hexdigest()
            if len(b)!=row['length'] or h!=row['buffer_sha256']:bad.append({'ordinal':ordinal,'path':row['path'],'part':row['part'],'offset':offset,'actual_buffer_sha256':h,'recorded_buffer_sha256':row['buffer_sha256']})
            boundary.update(b);offset+=len(b);ordinal+=1
        assert not data.read(1)
    with base.with_suffix('.bin').open('rb') as f:streamed=hashlib.file_digest(f,'sha256').hexdigest()
    result={'utc':utc(),'base':str(base),'bytes':offset,'hash_update_calls':ordinal,'recorded_digest':meta['digest_observed_during_hash_updates'],
        'independent_file_digest':streamed,'rehash_original_boundaries':boundary.hexdigest(),'differing_buffers':bad,
        'index_hash_matches':file_hash(base.with_suffix('.index.jsonl'))==meta['index_sha256'],
        'passed':not bad and meta['error'] is None and offset==meta['bytes'] and ordinal==meta['hash_update_calls'] and streamed==boundary.hexdigest()==meta['digest_observed_during_hash_updates']}
    durable(output,result);assert result['passed'] and result['index_hash_matches'],'Captured stream rehash discrepancy'
    return result

def compare_streams(left,right,output):
    left=Path(left);right=Path(right);offset=0;first=None
    with left.with_suffix('.bin').open('rb') as a,right.with_suffix('.bin').open('rb') as b:
        while True:
            x=a.read(8*1024*1024);y=b.read(8*1024*1024)
            if x!=y:
                limit=min(len(x),len(y));position=next((i for i in range(limit) if x[i]!=y[i]),limit)
                first={'absolute_byte_offset':offset+position,'left_sample_hex':x[position:position+16].hex(),'right_sample_hex':y[position:position+16].hex()};break
            if not x:break
            offset+=len(x)
    if first:
        for side,base in [('left',left),('right',right)]:
            with base.with_suffix('.index.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if row['offset']<=first['absolute_byte_offset']<row['offset']+row['length']:
                        first[side+'_hash_update']=row;first[side+'_offset_in_buffer']=first['absolute_byte_offset']-row['offset'];break
    lm=read_json(left.with_suffix('.stream.json'));rm=read_json(right.with_suffix('.stream.json'))
    result={'utc':utc(),'left':str(left),'right':str(right),'input_bytes_identical':first is None,'first_difference':first,
        'left_recorded_digest':lm['digest_observed_during_hash_updates'],'right_recorded_digest':rm['digest_observed_during_hash_updates'],
        'identical_bytes_but_different_recorded_digests':first is None and lm['digest_observed_during_hash_updates']!=rm['digest_observed_during_hash_updates']}
    durable(output,result);assert first is None and not result['identical_bytes_but_different_recorded_digests'],'Hash input/digest divergence'
    return result
