"""ROS-specific parameters travel through the common configuration interface."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest


def test_native_factory_forwards_algorithm_parameters_without_rewriting_limits(monkeypatch, tmp_path):
    import drone_playground.integrations.native_planner as native
    from drone_playground.integrations.native_service import create_native_planner

    captured = {}
    class Client:
        latencies = []
        def __init__(self, method, **kwargs):
            captured.update(kwargs)
        def reset(self, **kwargs):
            return dict(launch_xml='<launch/>')
        def close(self):
            pass
    monkeypatch.setattr(native, 'NativeClient', Client)
    monkeypatch.setattr(native.subprocess, 'check_output', lambda _: json.dumps([
        dict(NetworkSettings=dict(Networks=dict(private=dict(IPAddress='172.17.0.2'))))]).encode())
    settings = dict(implementation='native_ego', method='ego', container='runtime',
                    parameters=dict(occupancy_inflation_m=.199))
    planner = create_native_planner(settings, tmp_path, 12345, '/runtime/worker.py')
    try:
        limits = dict(max_velocity_mps=4., max_acceleration_mps2=3., planning_horizon_m=7.5)
        planner.start({}, [10., 0., 3.], limits)
        assert captured['parameters'] == dict(occupancy_inflation_m=.199, limits=limits)
        assert settings['parameters'] == dict(occupancy_inflation_m=.199)
    finally:
        planner.close()


def test_ego_inflation_is_explicit_and_default_launch_keeps_original_value(monkeypatch, tmp_path):
    path = Path(__file__).parents[1]/'native_planners/bridge/worker.py'
    spec = importlib.util.spec_from_file_location('parameter_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    source = '<launch><node pkg="ego_planner" name="ego"><param name="grid_map/obstacles_inflation" value="0.099"/></node></launch>'
    monkeypatch.setattr(worker.ET, 'parse', lambda _: SimpleNamespace(getroot=lambda: ET.fromstring(source)))
    calibration = dict(intrinsics=dict(cx_px=60, cy_px=45, fx_px=65., fy_px=65.))
    for parameters, expected in [(None, .099), (dict(occupancy_inflation_m=.199), .199)]:
        target = worker.launch_file('ego', calibration, [10, 0, 3], tmp_path, parameters=parameters)
        root = ET.fromstring(Path(target).read_text())
        value = root.find("node/param[@name='grid_map/obstacles_inflation']").get('value')
        assert float(value) == expected
    for value in [0, -1, float('nan'), float('inf'), True, '0.2']:
        with pytest.raises(ValueError, match='finite and positive'):
            worker.launch_file('ego', calibration, [10, 0, 3], tmp_path,
                               parameters=dict(occupancy_inflation_m=value))
    with pytest.raises(ValueError, match='EGO only'):
        worker.launch_file('super', {}, [10, 0, 3], tmp_path,
                           parameters=dict(occupancy_inflation_m=.199))
