from datetime import datetime,timezone
from pilot.common import ROOT,read_json,write_json,file_hash,digest
D=ROOT/'encoder_lr_screen02_diag01'
ORIGINAL=ROOT/'encoder_lr_screen02'
CK=ORIGINAL/'run/checkpoints'
COMPONENTS=('model','optimizer','scheduler','rng','progress')
CASES={
    1000:('rolling_0_g000001.pt','generation_000001.json','bd481f3fd47088b236aa2705aab5f4870afce7c1663c40b1de7c2854c9c7bf01'),
    2000:('rolling_1_g000002.pt','generation_000002.json','7d52707f6f6edc1295aa43def279cea9d75cfec6d271f3429421e0b18467c9c2'),
    3000:('rolling_0_g000003.pt',None,'451e4aa4096b16d67de10cfce20c1de3b4cfa61fb0150b409b164650ee66d6cf')}
def utc():return datetime.now(timezone.utc).isoformat()
def verify_plan():
    binding=read_json(D/'plan.lock.json')
    for p,h in binding['files'].items():assert file_hash(ROOT/p)==h,p
    return digest(binding)
