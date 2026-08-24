# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end orchestration for `gym eval compare`.

Ties loading, diffing and rendering together so the CLI entry point stays a thin wrapper.
"""

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.comparison.diff import compare_runs
from nemo_gym.comparison.loading import build_loaded_run, load_run_file, resolve_agent_selections, run_labels
from nemo_gym.comparison.report import write_reports
from nemo_gym.comparison.schema import CompareConfig, ComparisonResult, RunRef
from nemo_gym.package_info import __version__


def invoked_command() -> str:
    """The `gym eval compare` invocation, for provenance in the report.

    `dispatch` rewrites `sys.argv` to the resolved Hydra overrides before the entry point runs, so
    this records the resolved form -- which reproduces the run -- rather than the flags as typed.
    """
    return shlex.join(["gym", "eval", "compare", *sys.argv[1:]])


def build_comparison_result(config: CompareConfig, *, command: str) -> ComparisonResult:
    """Load both sides, pick the agent(s), and assemble the full comparison."""
    baseline_file = load_run_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        flag_label="--baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    candidate_files = [
        load_run_file(
            fpath,
            role="candidate",
            index=index,
            flag_label="--candidates",
            aggregate_metrics_fpath_override=(
                config.candidate_aggregate_metrics_fpaths[index] if config.candidate_aggregate_metrics_fpaths else None
            ),
        )
        for index, fpath in enumerate(config.candidate_rollouts_jsonl_fpaths)
    ]

    selections, warnings, skipped = resolve_agent_selections(
        baseline_file,
        candidate_files,
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        candidate_agent_names=config.candidate_agent_names,
    )

    labels = run_labels([baseline_file.rollouts_jsonl_fpath, *(f.rollouts_jsonl_fpath for f in candidate_files)])
    baseline_label, candidate_labels = labels[0], labels[1:]

    comparisons = []
    for selection in selections:
        baseline_run = build_loaded_run(baseline_file, selection.baseline_agent, label=baseline_label)
        candidate_runs = [
            build_loaded_run(run_file, agent, label=candidate_labels[index])
            for index, (run_file, agent) in enumerate(zip(candidate_files, selection.candidate_agents))
        ]
        comparisons.append(compare_runs(baseline_run, candidate_runs))

    def run_ref(run_file, label: str, repeats) -> RunRef:
        return RunRef(
            role=run_file.role,
            label=label,
            rollouts_jsonl_fpath=str(run_file.rollouts_jsonl_fpath),
            aggregate_metrics_fpath=str(run_file.aggregate_metrics_fpath),
            agents=run_file.agent_names,
            num_repeats=repeats,
        )

    first = comparisons[0]
    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        baseline=run_ref(baseline_file, baseline_label, first.baseline_repeat_count),
        candidates=[
            run_ref(run_file, candidate_labels[index], first.candidate_repeat_counts[index])
            for index, run_file in enumerate(candidate_files)
        ],
        comparisons=comparisons,
        skipped_agents=skipped,
        # Run-level warnings only. Per-comparison observations stay on their `AgentComparison.notes`
        # so the report renders each one once, under the agent it applies to.
        warnings=warnings,
    )


def resolve_output_dir(config: CompareConfig) -> Path:
    """`--output-dir`, defaulting to the candidate run's own directory.

    The default resolves the candidate path the same way loading does, so the report lands next to
    the metrics file that was actually read rather than at a same-named path under the cwd.
    """
    if config.output_dirpath:
        return Path(config.output_dirpath)
    return _resolve_under_cwd_or_install(config.candidate_rollouts_jsonl_fpaths[-1]).parent


def run_comparison(config: CompareConfig, *, command: str) -> Tuple[ComparisonResult, List[Path]]:
    """Build the comparison and write its report artifacts."""
    result = build_comparison_result(config, command=command)
    return result, write_reports(result, resolve_output_dir(config), config.report_format)
