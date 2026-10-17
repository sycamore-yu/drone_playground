"""The preview must match chosen native commands and never mislabel a policy."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_task_preview_selection_uses_exact_method_and_scene():
    """Browser-side matching has no checkpoint/algorithm substitution."""
    if shutil.which("node") is None:
        pytest.skip("Node is needed for browser selection logic")
    source = Path(__file__).parents[1] / "docs/task-preview.js"
    script = source.read_text()
    assertions = r"""
const fake=[
  {id:0,mode:'play',method:'planner',task:'static_oa',scene:'S01',sensor:'depth',
   planner:'ego',controller:'sampling_mpc',
   recordings:[{scene:'S01',event:'SUCCESS',elapsed_seconds:32.4}]},
  {id:1,mode:'play',method:'planner',task:'static_oa',scene:'S06',sensor:'depth',
   planner:'ego',controller:'sampling_mpc',
   recordings:[{scene:'S06',event:'METHOD_FAILURE',elapsed_seconds:6.4,
   failure_detail:'NO_SOLUTION'}]},
  {id:2,mode:'play',method:'planner',task:'static_oa',scene:'S06',sensor:'lidar',
   planner:'super',controller:'attitude_mpc',
   recordings:[{scene:'S06',event:'SUCCESS',elapsed_seconds:33.8}]},
  {id:3,mode:'benchmark',method:'planner',task:'static_oa',suite:'core',
   sensor:'depth',planner:'ego',controller:'sampling_mpc',
   recordings:[{scene:'S01',event:'METHOD_FAILURE'},{scene:'S02',event:'SUCCESS'}]},
];
const chooser=TaskPreviewCatalog;
const selected={mode:'play',method:'planner',task:'static_oa',scene:'S06',
  planner:'ego',controller:'sampling_mpc',sensor:'depth'};
const exact=chooser.caseForSelection(selected,fake);
if(!exact.exact||exact.recordedCase.id!==1||
   exact.recordedCase.recordings[0].event!=='METHOD_FAILURE')
  throw Error('Incorrect exact selection');
const benchmark=chooser.caseForSelection({...selected,mode:'benchmark',suite:'core'},fake);
if(!benchmark.exact||benchmark.recordedCase.id!==3||
   chooser.defaultRecording(benchmark.recordedCase)!==1)
  throw Error('Incorrect benchmark selection');
const policy=chooser.caseForSelection(
  {...selected,method:'policy',algorithm:'shac',checkpoint:'best'},fake);
if(policy.exact||policy.recordedCase.id!==1)
  throw Error('Policy must receive only a labelled task example');
const other=chooser.caseForSelection({...selected,controller:'full_state_nmpc'},fake);
if(other.recordedCase!==null)
  throw Error('Cannot masquerade as an unrecorded native controller');
process.stdout.write('preview matching passed');
"""
    result = subprocess.run(
        ["node"], input=script + "\n" + assertions, text=True, capture_output=True, check=True
    )
    assert "preview matching passed" in result.stdout
