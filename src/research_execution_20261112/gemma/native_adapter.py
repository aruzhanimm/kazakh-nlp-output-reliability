"""Preserve native Gemma channels; apply unchanged research parsers to final text."""
import existing_alignment
import existing_sa_scoring

def parse(processor,raw_token_ids,prefix_ids,item,allowed_types):
    raw=processor.tokenizer.decode(raw_token_ids,skip_special_tokens=False)
    native=processor.parse_response(raw_token_ids,prefix=prefix_ids)
    if isinstance(native,list):
        if len(native)!=1:raise ValueError('Expected exactly one native reply')
        native=native[0]
    if not isinstance(native,dict) or native.get('role','assistant')!='assistant' or native.get('tool_calls'):
        raise ValueError('Unexpected native reply/tool calls; preserve raw evidence')
    final=native.get('content','')
    if not isinstance(final,str):raise ValueError('Pinned tokenizer response template requires string content')
    parsed=existing_alignment.parse_mentions(final,item['tokens'],allowed_types) if item['task']=='ner' else existing_sa_scoring.parse_sa(final)
    return {'raw_token_ids':list(raw_token_ids),'raw_native_text':raw,'native_parsed':native,
            'task_text':final,'task_parsed':parsed,'unexpected_nonempty_thinking':bool(native.get('thinking'))}
