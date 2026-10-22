# DeepSWE ripgrep upload retry probe

This diagnostic exercises the exact Gym `79901c5761df43645056f7b5a482cfbc34c4d271`
`OpenCodeSandboxedAgent._stage_ripgrep` method with one real OpenSandbox task-image pod.
It makes **no model requests** and produces **no DeepSWE result row**.

After the first real provider upload completes, the process raises a synthetic
`SandboxIOTimeoutError` in place of its acknowledgement. The agent must retry at
a different UUID path. The probe downloads both remote files, checks their bytes
against the pinned ripgrep SHA-256, installs the second file into OpenCode's
normal `$HOME/.opencode/bin/rg` path, checks its checksum and runs `rg --version`.
The pod is stopped in a bounded, shielded `finally` block and has a required
finite TTL with renewal disabled. The evidence records the sandbox ID, upload
paths, stop confirmation and failure class even when the route fails.

The probe is opt-in. Run it only after an independent all-user DeepSeek DeepSWE
no-writer and cluster-capacity check. Use the resolved provider and task-image
settings from the same exact EFB `e4e71e5fa337ac58f0cdead6593a8c5e4019ade9`
trial. The manifest must live outside this checkout and include:

```json
{
  "efb_root": "/absolute/path/to/efb-e4e-checkout",
  "source_root": "/absolute/path/to/frozen-task-source-checkout",
  "source_commit": "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9",
  "source_file": "/absolute/path/to/frozen/task/source/file",
  "source_sha256": "<64-character SHA-256>",
  "task_list_file": "/absolute/path/to/canonical/task-list/file",
  "task_list_sha256": "<64-character SHA-256>",
  "resolved_run_file": "/absolute/path/to/resolved/run.sub",
  "resolved_run_sha256": "<64-character SHA-256>",
  "ripgrep_file": "/opt/deepswe-ripgrep/rg",
  "ripgrep_sha256": "ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2",
  "provider_config": {"opensandbox": {"connection": {"domain": "<reviewed resolved endpoint>"}}},
  "sandbox_spec": {
    "image": "<exact task image>",
    "ttl_s": 1200,
    "ready_timeout_s": 120,
    "resources": {"cpu": 2, "memory_mib": 8192},
    "provider_options": {
      "network_policy": {"defaultAction": "deny", "egress": []}
    }
  }
}
```

The displayed provider and sandbox blocks are **placeholders**. Copy the
resolved run's OpenSandbox endpoint and a canonical task-image resource limit.
The manifest's provider and image fields **cannot be derived reliably by this
script from the shell-formatted `run.sub`**. A separate reviewer must compare
them with that exact resolved run and the canonical task source, then compute
the SHA-256 of the reviewed manifest. The `run.sub` token checks are sanity
checks, not a proof that those fields match. Live execution requires an explicit
attestation of this comparison.

The probe rejects missing SHA pins, a changed probe script or Gym/EFB/source
checkout, a changed manifest/source/task-list/ripgrep file, an unbounded pod,
any egress exception, startup files, sidecars, ports and GPUs. It allows only
connection, create and operations provider blocks, so provider setup commands,
image rewrites, network setup commands and TTL renewal cannot be used. A
disposable copy sets `create.retries=0` and `operations.command_retries=0`
before it creates the provider. The pod network policy must be exactly
`{"defaultAction":"deny","egress":[]}`. The exact task image is reviewer
attested; this probe does not verify a registry digest or all 113 images.

First run without `--execute` to check provenance without contacting OpenSandbox:

```text
uv run --extra dev python scripts/diagnostics/deepswe_rg_retry_probe.py \
  --manifest /absolute/path/to/probe.json \
  --reviewed-manifest-sha256 <reviewed-manifest-sha256> \
  --script-sha256 <independently-computed-script-sha256>
```

After checking writer/capacity state and independently reviewing the manifest
mapping, add `--execute --attest-resolved-mapping-reviewed` and a fresh
`--evidence` path. Preserve that JSON with the run's logs. A passed status
requires stop confirmation for the same sandbox ID. A failed status records
the sandbox ID, attempted paths, failure stage/type and cleanup status without
provider credentials. An unconfirmed stop remains a failure; the finite TTL is
the last cleanup bound.

This proves the retry path can recover from a simulated lost acknowledgement
through a real provider and sandbox. It does **not** measure spontaneous
300-second upload timeouts, four-worker load, provider-status failures, OOM
classification or model/verifier behavior. A separate model-backed concurrency
pilot and complete 113-task audit remain necessary for release claims.
