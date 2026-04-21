"""Debug script: test SSD transfer with real KV cache tensors."""
import os
import torch
import tempfile

# Set up vLLM environment
os.environ["VLLM_USE_PRECOMPILED"] = "1"

from vllm import LLM, SamplingParams

# Create a minimal LLM to get the KV cache tensors
print("Creating LLM...")
llm = LLM(
    model="/root/autodl-fs/model/OPT-6.7B",
    max_num_seqs=4,
    gpu_memory_utilization=0.5,
    disable_log_stats=True,
)

# Get engine internals - we need the KV cache tensors
engine = llm.llm_engine.engine_core
print(f"Engine type: {type(engine)}")

# Run a small generation to initialize KV cache
print("Running small generation to init KV cache...")
outputs = llm.generate(
    ["Hello world, how are you today?", "Tell me about machine learning."],
    SamplingParams(max_tokens=30, temperature=0.0),
)
print(f"Generated {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")

# Now let's check the canonicalized KV cache
from vllm.v1.kv_offload.spec import CanonicalKVCaches, CanonicalKVCacheTensor

# Get KV cache config from worker
print("\nChecking KV cache structure...")
worker = engine._core_engine.engine_core.workers[0]
print(f"Worker type: {type(worker)}")

# Get the KV cache tensors
kv_cache_config = worker.model_runner.kv_cache_config
print(f"KV cache config: {kv_cache_config}")

# Get canonicalized KV caches
canonical = kv_cache_config.canonicalize()
print(f"Canonical KV caches: {len(canonical.tensors)} tensors")
for i, t in enumerate(canonical.tensors):
    print(f"  Tensor {i}: shape={t.tensor.shape}, dtype={t.tensor.dtype}, device={t.tensor.device}, page_size={t.page_size_bytes}")

# Now test a transfer
from vllm.bidaw.ssd.mediums import SSDLoadStoreSpec
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.bidaw.ssd.worker import SSDDirectionHandler

tmpdir = tempfile.mkdtemp(prefix="bidaw_debug_")
print(f"\nUsing tmpdir: {tmpdir}")

# Create handler
handler = SSDDirectionHandler(
    kv_caches=canonical,
    ssd_cache_dir=tmpdir,
    io_threads=2,
    gpu_to_ssd=True,
)

# Create a transfer spec for block 0
block_ids = [0, 1]
src_spec = GPULoadStoreSpec(block_ids, group_sizes=(len(block_ids),))
dst_spec = SSDLoadStoreSpec(cache_dir=tmpdir, io_threads=2)
for i, bid in enumerate(block_ids):
    dst_spec.add_block_path(f"block_{i:08x}", os.path.join(tmpdir, f"block_{bid}.bin"))

print(f"\nSubmitting transfer for blocks {block_ids}...")
print(f"  src_spec block_ids: {src_spec.block_ids}")
print(f"  dst_spec block_paths: {list(dst_spec.block_paths.values())}")

result = handler.transfer_async(job_id=1, spec=(src_spec, dst_spec))
print(f"  transfer_async returned: {result}")

# Wait for completion
import time
for _ in range(10):
    finished = handler.get_finished()
    if finished:
        for f in finished:
            print(f"  Finished: job_id={f.job_id}, success={f.success}, size={f.transfer_size}, time={f.transfer_time:.3f}s")
        break
    time.sleep(0.5)
else:
    print("  TIMEOUT: transfer did not complete!")
    # Check what files were created
    files = os.listdir(tmpdir)
    print(f"  Files in {tmpdir}: {files}")
    for f in files:
        fpath = os.path.join(tmpdir, f)
        print(f"    {f}: {os.path.getsize(fpath)} bytes")

# Now test load back
print("\n--- Testing SSD->GPU load ---")
handler_load = SSDDirectionHandler(
    kv_caches=canonical,
    ssd_cache_dir=tmpdir,
    io_threads=2,
    gpu_to_ssd=False,
)

files = sorted(os.listdir(tmpdir))
print(f"  Files to load: {files}")

src_spec_load = SSDLoadStoreSpec(cache_dir=tmpdir, io_threads=2)
for i, f in enumerate(files):
    src_spec_load.add_block_path(f"block_{i:08x}", os.path.join(tmpdir, f))

dst_spec_load = GPULoadStoreSpec(list(range(len(files))), group_sizes=(len(files),))

print(f"  Loading blocks {list(range(len(files)))} from {files}")
result2 = handler_load.transfer_async(job_id=2, spec=(src_spec_load, dst_spec_load))
print(f"  transfer_async returned: {result2}")

for _ in range(10):
    finished = handler_load.get_finished()
    if finished:
        for f in finished:
            print(f"  Finished: job_id={f.job_id}, success={f.success}, size={f.transfer_size}, time={f.transfer_time:.3f}s")
        break
    time.sleep(0.5)
else:
    print("  TIMEOUT: load transfer did not complete!")

print("\nDebug done!")
