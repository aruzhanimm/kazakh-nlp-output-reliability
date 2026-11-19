"""Token-normalized NER accumulation: one denominator per complete optimizer batch."""
def valid_count(features):
    return sum(sum(int(y!=-100) for y in f['labels']) for f in features)

def accumulate(model,features,microbatch,collate,device='cuda',autocast=True):
    import torch
    from torch.nn import functional as F
    denom=valid_count(features)
    if denom<=0:raise ValueError('Optimizer batch has no supervised first-subword labels')
    numerator=0.;padded=0
    for start in range(0,len(features),microbatch):
        batch=collate(features[start:start+microbatch],device)
        labels=batch.pop('labels');padded+=labels.numel()
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast and device=='cuda'):
            logits=model(**batch).logits
        summed=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
        if not bool(torch.isfinite(summed)):raise RuntimeError('Nonfinite summed cross-entropy')
        (summed/denom).backward();numerator+=float(summed.detach())
    return {'loss_sum':numerator,'valid_labels':denom,'loss':numerator/denom,'examples':len(features),'padded_tokens':padded}
