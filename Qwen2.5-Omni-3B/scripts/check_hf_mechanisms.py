"""Execute selected archived HF functions on CPU without importing Transformers or weights.

Only signatures/decorators are adapted to make isolated functions loadable.
Function bodies come directly from the byte-identical archived source.
"""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(symbol, namespace):
    index = json.loads((ROOT / 'sources/code_index.json').read_text())
    entry = index[symbol]
    tree = ast.parse((ROOT / entry['local_path']).read_text())
    node = copy.deepcopy(next(n for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef) and n.lineno == entry['start_line']))
    for n in ast.walk(node):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, ast.FunctionDef):
            n.returns = None
            n.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / entry['local_path']), 'exec'), namespace)
    return namespace[node.name]

def run():
    ns = dict(torch=torch, np=np, math=math)
    load('rotate_half', ns)
    rotate = load('apply_multimodal_rotary_pos_emb', ns)
    torch.manual_seed(7)
    q = torch.randn(1,2,3,128,dtype=torch.float64)
    k = torch.randn_like(q)
    positions = torch.arange(3,dtype=torch.float64)
    inv = 1000000.0 ** (-torch.arange(0,128,2,dtype=torch.float64)/128)
    angle = positions[:,None] * inv[None,:]
    angle = torch.cat([angle,angle],dim=-1)
    cos = angle.cos()[None,None].expand(3,1,3,128)
    sin = angle.sin()[None,None].expand(3,1,3,128)
    qr,kr = rotate(q,k,cos,sin,[16,24,24])
    ordinary = q * angle.cos()[None,None] + ns['rotate_half'](q) * angle.sin()[None,None]
    assert torch.allclose(qr,ordinary)
    assert torch.allclose(qr.norm(dim=-1),q.norm(dim=-1))
    lengths = load('Qwen2_5OmniAudioEncoder._get_feat_extract_output_lengths',ns)
    mel = torch.tensor([4,5,7,8,199,200,201,400])
    after_conv,after_pool = lengths(None,mel)
    assert after_conv.tolist() == [2,3,4,4,100,100,101,200]
    assert after_pool.tolist() == [1,1,2,2,50,50,50,100]
    cfg = json.loads((ROOT / 'sources/checkpoint/config.json').read_text())['thinker_config']
    fake = SimpleNamespace(config=SimpleNamespace(**cfg),spatial_merge_size=2)
    base = 'Qwen2_5OmniPreTrainedModelForConditionalGeneration.'
    for name in ['get_llm_pos_ids_for_vision','get_chunked_index','get_rope_index']:
        setattr(fake,name,MethodType(load(base+name,ns),fake))
    text = torch.tensor([[10,11,12]])
    p,d = fake.get_rope_index(text,attention_mask=torch.ones_like(text))
    assert p[:,0].tolist() == [[0,1,2]]*3 and d.item()==0
    # Left padding must not advance valid text positions.
    padded=torch.tensor([[0,0,10,11,12]])
    pp,pd=fake.get_rope_index(padded,attention_mask=torch.tensor([[0,0,1,1,1]]))
    assert pp[:,0,2:].tolist()==[[0,1,2]]*3
    vs,ve,im,au,ab,ae,vi = [cfg[x] for x in ['vision_start_token_id','vision_end_token_id','image_token_index','audio_token_index','audio_start_token_id','audio_end_token_id','video_token_index']] if 'vision_start_token_id' in cfg else [151652,151653,cfg['image_token_index'],cfg['audio_token_index'],cfg['audio_start_token_id'],cfg['audio_end_token_id'],cfg['video_token_index']]
    # These tokens are constructor defaults missing from the serialized thinker dict.
    fake.config.vision_start_token_id=vs
    image = torch.tensor([[10,vs,im,im,im,im,ve,11]])
    ip,idel=fake.get_rope_index(image,image_grid_thw=torch.tensor([[1,4,4]]),attention_mask=torch.ones_like(image))
    assert ip[:,0,2:6].tolist()==[[2,2,2,2],[2,2,3,3],[2,3,2,3]]
    audio=torch.tensor([[10,ab]+[au]*50+[ae,11]])
    ap,ad=fake.get_rope_index(audio,audio_seqlens=torch.tensor([200]),attention_mask=torch.ones_like(audio))
    assert ap[:,0].tolist()==[list(range(audio.shape[1]))]*3
    processor_chunk=load('Qwen2_5OmniProcessor.get_chunked_index',ns)
    intended=processor_chunk(None,np.arange(100),50)
    historical=fake.get_chunked_index(torch.arange(100).repeat(3,1),50,0)
    assert intended==[(0,50),(50,100)]
    assert historical==[(0,100)], 'Snapshot behavior changed: investigate the source before reusing this finding.'
    # The helper also indexes beyond the end for a grid with fewer than 3 tokens.
    short_grid_error=False
    try:
        fake.get_chunked_index(torch.tensor([[0,50],[0,0],[0,0]]),50,0)
    except IndexError:
        short_grid_error=True
    assert short_grid_error
    # Two video temporal grids, four merged spatial tokens each, spaced by 2 s.
    av=torch.tensor([[10,vs,ab]+[vi]*4+[au]*50+[vi]*4+[au]*50+[ae,ve,11]])
    avp,avd=fake.get_rope_index(av,video_grid_thw=torch.tensor([[2,4,4]]),
                              audio_seqlens=torch.tensor([400]),second_per_grids=torch.tensor([2.0]),
                              attention_mask=torch.ones_like(av),use_audio_in_video=True)
    video_locations=(av[0]==vi).nonzero().flatten()[::4]
    actual_video_times=avp[0,0,video_locations].tolist()
    expected_video_times=[2,52]
    assert actual_video_times!=expected_video_times
    result=dict(torch_version=torch.__version__,cpu_only=True,weights_loaded=False,
        text_positions=True,left_padding=True,image_positions=True,audio_positions=True,
        text_tmrope_matches_rope=True,rotation_norm_preserved=True,audio_lengths=True,
        processor_two_second_chunks=intended,model_helper_chunks=historical,
        audiovisual_expected_temporal_positions=expected_video_times,
        audiovisual_actual_temporal_positions=actual_video_times,
        known_snapshot_interleaving_discrepancy_confirmed=True,
        short_visual_grid_index_error_confirmed=short_grid_error,
        note='The model helper loops over len(3,N)=3 rather than N, unlike the 1-D processor helper. Archived upstream code is not patched.')
    (ROOT / 'hf_mechanism_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
