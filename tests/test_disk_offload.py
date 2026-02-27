import torch
from torch import nn

from mechinterp_qwen3.utils.model_utils import (
    cleanup_all_offload_files,
    cpu_offload_module,
    disk_offload_module,
    offload_modules,
)


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(3, 3))


def test_cpu_offload():
    module = SimpleModule()
    org_device = next(module.parameters()).device

    reload_handle = cpu_offload_module(module)
    assert next(module.parameters()).device.type == "cpu"

    reload_handle()
    assert next(module.parameters()).device == org_device


def test_disk_offload():
    module = SimpleModule()
    org_device = next(module.parameters()).device

    reload_handle = disk_offload_module(module)
    # meta device means it's offloaded
    assert next(module.parameters()).device.type == "meta"

    reload_handle()
    assert next(module.parameters()).device == org_device


def test_offload_modules_list():
    m1 = SimpleModule()
    m2 = SimpleModule()
    org_device = next(m1.parameters()).device

    handles = offload_modules([m1, m2], "cpu")
    assert len(handles) == 2
    assert next(m1.parameters()).device.type == "cpu"
    assert next(m2.parameters()).device.type == "cpu"

    for h in handles:
        h()
    assert next(m1.parameters()).device == org_device


def test_cleanup_offload_files():
    # This is a bit hard to test directly without mocking NamedTemporaryFile
    # But we can at least ensure it doesn't crash
    n = cleanup_all_offload_files()
    assert isinstance(n, int)
