"""Validate the public tool contract through Pi's argument validator."""

import json
from pathlib import Path
import subprocess

from src.harness.memory import MemoryStore
from src.harness.tools.specs import specifications


def test_unified_references_and_action_modes_pass_pi_validation(tmp_path):
    tools = {tier: {tool["name"]: tool for tool in specifications(tier)}
             for tier in ("basic", "super")}
    script = """
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateToolArguments} from '@earendil-works/pi-ai';
const tools = JSON.parse(readFileSync(0, 'utf8'));
const invoke = (tier,name,args) => validateToolArguments(tools[tier][name],
    {id:'fixture',name,arguments:args});
const edit = (type,fields) => ({operations:[{
    op:'create',id:{object:'O01',room:'R01',door:'D01'}[type],type,fields:{name:'fixture',...fields},
}]});
for (const tier of ['basic','super']) {
    for (const type of ['object','room','door']) {
        assert.doesNotThrow(()=>invoke(tier,'memory_edit',edit(type,{
            location:{image:'m_0001',pixel:[384,207]},
        })));
        assert.doesNotThrow(()=>invoke(tier,'memory_edit',edit(type,{
            location:{uv:[.5,.6]},
        })));
    }
    assert.doesNotThrow(()=>invoke(tier,'memory_edit',edit('door',{point:[.5,.6]})));
    assert.throws(()=>invoke(tier,'memory_edit',edit('door',{point:[384,207]})),/Validation failed/);
    assert.throws(()=>invoke(tier,'memory_edit',edit('door',{segment:[[.4,.5],[.5,.5]]})),/Validation failed/);
    assert.throws(()=>invoke(tier,'memory_edit',edit('door',{at:{frame:'f_0001',pixel:[320,240]}})),/Validation failed/);
    assert.doesNotThrow(()=>invoke(tier,'move',{mode:'steps',actions:['move_forward']}));
    assert.doesNotThrow(()=>invoke(tier,'view_images',{images:[{image:'i_0001',crop:[0,0,.5,.5]}]}));
    assert.doesNotThrow(()=>invoke(tier,'map_view',{project:{source:'m_0001',points:[[384,207]]}}));
    for (const [name,args] of [
        ['observe',{mode:'current',grid:true}],
        ['view_images',{frames:[{frame:'f_0001'}]}],
        ['view_images',{images:[{image:'f_0001',grid:true}]}],
    ]) {
        assert.throws(()=>invoke(tier,name,args),/Validation failed/);
    }
}
assert.doesNotThrow(()=>invoke('super','move',{mode:'face',target:{image:'f_0001',pixel:[320,240]}}));
assert.doesNotThrow(()=>invoke('super','move',{mode:'approach',target:{entity:'O01'},after:'panorama'}));
assert.throws(()=>invoke('basic','move',{mode:'approach',target:{entity:'O01'}}),/Validation failed/);
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            input=json.dumps(tools), text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr
    memory = MemoryStore(tmp_path)
    result = memory.edit([{"op": "create", "id": "O01", "type": "object",
                           "fields": {"name": "invalid UV", "point": [384, 207]}}])
    assert "Coordinates must" in result["error"]
    assert memory.entities == []
