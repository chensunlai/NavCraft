"""Validate Python's tool schemas through Pi's actual argument validator."""

import json
from pathlib import Path
import subprocess

from src.harness.memory import MemoryStore
from src.harness.tools.specs import specifications


def test_map_pixel_geometry_passes_pi_validation_only_in_super_tier(tmp_path):
    tools = {tier: next(tool for tool in specifications(tier) if tool["name"] == "memory_edit")
             for tier in ("basic", "super")}
    script = """
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateToolArguments} from '@earendil-works/pi-ai';
const tools = JSON.parse(readFileSync(0, 'utf8'));
function call(type, fields, coordinate_ref) {
    return {id:'fixture',name:'memory_edit',arguments:{operations:[{
        op:'create',id:{object:'O01',room:'R01',door:'D01'}[type],type,fields:{name:'fixture',...fields},
        ...(coordinate_ref ? {coordinate_ref} : {}),
    }]}};
}
for (const [type, fields] of [
    ['object',{point:[384,207]}],
    ['room',{anchor:[384,207]}],
    ['door',{segment:[[100,207],[300,207]]}],
]) {
    assert.doesNotThrow(()=>validateToolArguments(tools.super,call(type,fields,'m_0001')));
    assert.throws(()=>validateToolArguments(tools.basic,call(type,fields)),/Validation failed/);
}
for (const tier of ['basic','super']) {
    assert.doesNotThrow(()=>validateToolArguments(tools[tier],call('object',{point:[.5,.6]})));
}
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            input=json.dumps(tools), text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr
    # Widening the transport schema must not widen the stored global UV range.
    memory = MemoryStore(tmp_path)
    result = memory.edit([{"op": "create", "id": "O01", "type": "object",
                           "fields": {"name": "invalid UV", "point": [384, 207]}}])
    assert "Coordinates must" in result["error"]
    assert memory.entities == []
