"""SA: summed review CE / actual optimizer-batch reviews, including partial batches."""
from contextlib import nullcontext

def collator(tokenizer):
    def apply(features,device):
        import torch
        assert features and all(type(f['labels']) is int and f['labels'] in (0,1) for f in features)
        length=max(len(f['input_ids']) for f in features);assert length<=384
        inputs={}
        for key,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('token_type_ids',0)]:
            if key=='token_type_ids' and not any(key in f for f in features):continue
            assert all(key in f for f in features)
            inputs[key]=torch.tensor([f[key]+[pad]*(length-len(f[key])) for f in features],dtype=torch.long,device=device)
        inputs['labels']=torch.tensor([f['labels'] for f in features],dtype=torch.long,device=device)
        return inputs
    return apply

def accumulate(model,features,microbatch,collate,device='cuda',amp=True):
    import torch
    assert features and microbatch>0
    denominator=len(features);loss_sum=0.
    for start in range(0,denominator,microbatch):
        batch=collate(features[start:start+microbatch],device);gold=batch.pop('labels')
        assert gold.ndim==1 and bool(((gold==0)|(gold==1)).all())
        context=torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
        with context:logits=model(**batch).logits
        assert logits.shape==(len(gold),2),'SA requires two logits per review'
        assert bool(torch.isfinite(logits).all()),'Nonfinite logits'
        summed=torch.nn.functional.cross_entropy(logits.float(),gold,reduction='sum')
        assert bool(torch.isfinite(summed))
        (summed/denominator).backward();loss_sum+=float(summed.detach())
    return {'loss_sum':loss_sum,'examples':denominator,'loss':loss_sum/denominator}

def update(model,optimizer,scheduler,features,microbatch,collate):
    import torch
    optimizer.zero_grad(set_to_none=True)
    result=accumulate(model,features,microbatch,collate)
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing,'Missing gradients: '+str(missing)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));lr=optimizer.param_groups[0]['lr']
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    return {**result,'gradient_norm':norm,'learning_rate':lr,'next_learning_rate':scheduler.get_last_lr()[0]}
