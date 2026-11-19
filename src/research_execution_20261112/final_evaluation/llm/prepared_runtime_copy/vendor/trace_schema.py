"""Pass-through generation instrumentation; no tensor histories or score edits."""
import json
import math
import time
from collections import Counter


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if hasattr(value,"numel"):
        return value.detach().cpu().tolist() if value.numel()<=16 else scalar(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)


def scalar(x):
    if hasattr(x,"numel"):
        if x.numel()==1:return x.item()
        return {"shape":list(x.shape),"dtype":str(x.dtype),"device":str(x.device)}
    return x


def cache_info(cache):
    if cache is None:return {"type":None}
    info={"type":type(cache).__name__}
    if hasattr(cache,"get_seq_length"):
        try:info["seq_length"]=scalar(cache.get_seq_length())
        except Exception as e:info["length_error"]=type(e).__name__
    layers=getattr(cache,"layers",None)
    if layers:
        info["layers"]=len(layers);info["first_layer_type"]=type(layers[0]).__name__
        for key in ("keys","values"):
            value=getattr(layers[0],key,None)
            if value is not None:info[key]={"shape":list(value.shape),"dtype":str(value.dtype),"device":str(value.device)}
    return info


def stats(scores):
    import torch
    x=scores.detach().float().flatten()
    finite=torch.isfinite(x)
    n=int(finite.sum())
    vals=x[finite]
    clean=torch.where(finite,x,torch.full_like(x,-float('inf')))
    ids=clean.topk(min(3,n)).indices.tolist() if n else []
    return {"dtype":str(scores.dtype),"shape":list(scores.shape),"finite":n,
            "nan":int(torch.isnan(x).sum()),"positive_inf":int(torch.isposinf(x).sum()),"negative_inf":int(torch.isneginf(x).sum()),
            "finite_min":float(vals.min()) if n else None,"finite_max":float(vals.max()) if n else None,
            "argmax_including_nonfinite":int(x.argmax()),"token0":float(x[0]) if bool(finite[0]) else str(float(x[0])),
            "top_finite":[{"id":i,"value":float(x[i])} for i in ids]}


class Trace:
    def __init__(self,model,path):
        self.model=model;self.path=path;self.pending={};self.count=0;self.seen_prompt=False;self.metadata={}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open("x",encoding="utf-8")

    def put(self,value):
        if not self.seen_prompt:
            self.seen_prompt=True;return
        ids=value.flatten().tolist()
        record={"step":self.count,"selected_token_ids":ids,**self.pending}
        self.stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");self.stream.flush()
        self.pending={};self.count+=1

    def end(self):pass

    def __enter__(self):
        from transformers import LogitsProcessor,LogitsProcessorList
        owner=self
        class Recorder(LogitsProcessor):
            def __init__(self,kind):self.kind=kind
            def __call__(self,input_ids,scores):
                owner.pending[self.kind]=stats(scores)
                owner.pending["sequence_length"]=input_ids.shape[-1]
                return scores
        self.old_processors=self.model._get_logits_processor
        def get_processors(*args,**kwargs):
            original=owner.old_processors(*args,**kwargs)
            owner.metadata["native_logits_processors"]=[type(p).__name__ for p in original]
            c=kwargs.get("generation_config",args[0] if args else None)
            if c is not None:
                owner.metadata["actual_generation_config"]=json_safe(c.to_dict())
                owner.metadata["actual_compile_config"]=str(c.compile_config)
            return LogitsProcessorList([Recorder("raw"),*original,Recorder("processed")])
        def forward_hook(module,args,kwargs,output):
            owner.pending["cache_out"]=cache_info(getattr(output,"past_key_values",None))
            owner.pending["input_shape"]=list(kwargs["input_ids"].shape) if "input_ids" in kwargs else None
            owner.pending["use_cache"]=kwargs.get("use_cache")
            owner.pending["cache_position"]=scalar(kwargs.get("cache_position"))
            owner.pending["logits_shape"]=list(output.logits.shape)
        self.model._get_logits_processor=get_processors
        self.hook=self.model.register_forward_hook(forward_hook,with_kwargs=True)
        return self

    def __exit__(self,*exc):
        self.hook.remove();self.model._get_logits_processor=self.old_processors
        self.stream.close()


def attention_info(model):
    import torch
    return {"config":model.config._attn_implementation,
            "module_classes":dict(Counter(type(m).__name__ for n,m in model.named_modules() if n.endswith("self_attn"))),
            "is_quantized":getattr(model,"is_quantized",None),
            "cache_config":getattr(model.config,"use_cache",None),
            "sdpa_flags":{"flash":torch.backends.cuda.flash_sdp_enabled(),"memory_efficient":torch.backends.cuda.mem_efficient_sdp_enabled(),"math":torch.backends.cuda.math_sdp_enabled()},
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"torch":torch.__version__,"cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None}
