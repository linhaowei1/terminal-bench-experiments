import argparse
import asyncio
from contextlib import ExitStack, contextmanager
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import harbor
import yaml
from dotenv import dotenv_values, load_dotenv
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from litellm import model_cost
from supabase import acreate_client
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

sys.path.append(str(Path(__file__).resolve().parent.parent))

from db.schema_public_latest import (
    AgentInsert,
    JobInsert,
    ModelInsert,
    TrialInsert,
    TrialModelInsert,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
ENV_FILE_VALUES = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
ANTHROPIC_PROXY_ENV_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")
RESUME_UNORDERED_LIST_PATHS = {
    ("retry", "include_exceptions"),
    ("retry", "exclude_exceptions"),
}

load_dotenv(ENV_FILE if ENV_FILE.exists() else None)


def _job_uses_codex(config: JobConfig) -> bool:
    return any((agent.name or "").lower() == "codex" for agent in config.agents)


def _job_uses_official_anthropic(config: JobConfig) -> bool:
    return any((agent.model_name or "").startswith("anthropic/") for agent in config.agents)


def _job_explicitly_sets_env(config: JobConfig, key: str) -> bool:
    if key in config.environment.env:
        return True
    return any(key in agent.env for agent in config.agents)


def _sanitize_ambient_anthropic_proxy_env(config: JobConfig) -> None:
    """
    Prefer repo-declared Anthropic credentials over unrelated shell-level proxy vars.

    These jobs target official ``anthropic/*`` models. If the repo's ``.env`` does not
    declare a proxy URL/token, ambient shell exports like ``ANTHROPIC_BASE_URL`` can
    silently reroute requests to a different upstream and cause 401s.
    """
    if not _job_uses_official_anthropic(config):
        return
    if "ANTHROPIC_API_KEY" not in os.environ:
        return

    cleared_vars: list[str] = []
    for env_var in ANTHROPIC_PROXY_ENV_VARS:
        if ENV_FILE_VALUES.get(env_var) is not None:
            continue
        if _job_explicitly_sets_env(config, env_var):
            continue
        if os.environ.pop(env_var, None) is not None:
            cleared_vars.append(env_var)

    if cleared_vars:
        print(
            "Cleared ambient Anthropic proxy settings for official anthropic/* "
            f"models: {', '.join(cleared_vars)}"
        )


def _sanitize_ambient_codex_auth(config: JobConfig) -> None:
    """
    Prefer repo-declared OPENAI_API_KEY over ambient ~/.codex/auth.json for Codex jobs.

    Harbor's Codex agent defaults to injecting ~/.codex/auth.json when it exists,
    which silently overrides the repository's OPENAI_API_KEY. Users can still opt
    into auth.json explicitly with CODEX_AUTH_JSON_PATH or manage precedence with
    an explicit CODEX_FORCE_API_KEY setting.
    """
    if not _job_uses_codex(config):
        return
    if ENV_FILE_VALUES.get("OPENAI_API_KEY") is None:
        return
    if _job_explicitly_sets_env(config, "CODEX_FORCE_API_KEY"):
        return
    if _job_explicitly_sets_env(config, "CODEX_AUTH_JSON_PATH"):
        return
    if os.environ.get("CODEX_FORCE_API_KEY") is not None:
        return
    if os.environ.get("CODEX_AUTH_JSON_PATH"):
        return

    default_auth_json = Path.home() / ".codex" / "auth.json"
    if not default_auth_json.is_file():
        return

    os.environ["CODEX_FORCE_API_KEY"] = "1"
    print(
        "Set CODEX_FORCE_API_KEY=1 so Codex jobs use OPENAI_API_KEY from the repo "
        "environment instead of ~/.codex/auth.json"
    )


def _configs_match_for_resume(
    existing_config: JobConfig, requested_config: JobConfig
) -> bool:
    """
    Compare configs using a normalized JSON form so resume tolerates benign drift.

    Harbor serializes sensitive env vars in ``config.json`` with masking. Direct model
    equality compares the in-memory raw values and incorrectly rejects resume attempts
    when the requested config still contains the original secret. Some config fields
    are set-like but serialized as JSON arrays, so order-only differences should not
    block resume either.
    """
    return _resume_values_match(
        existing_config.model_dump(mode="json"),
        requested_config.model_dump(mode="json"),
        ignore_paths={("n_concurrent_trials",)},
    )


def _models_match_for_resume(existing_model: object, requested_model: object) -> bool:
    if not hasattr(existing_model, "model_dump") or not hasattr(
        requested_model, "model_dump"
    ):
        return existing_model == requested_model

    return _resume_values_match(
        existing_model.model_dump(mode="json"),
        requested_model.model_dump(mode="json"),
    )


def _resume_values_match(
    existing_value: object,
    requested_value: object,
    path: tuple[str, ...] = (),
    *,
    ignore_paths: set[tuple[str, ...]] | None = None,
) -> bool:
    if ignore_paths and path in ignore_paths:
        return True

    if existing_value == requested_value:
        return True

    if _masked_string_matches(existing_value, requested_value):
        return True

    if isinstance(existing_value, dict) and isinstance(requested_value, dict):
        if set(existing_value) != set(requested_value):
            return False

        return all(
            _resume_values_match(
                existing_value[key],
                requested_value[key],
                path + (str(key),),
                ignore_paths=ignore_paths,
            )
            for key in existing_value
        )

    if isinstance(existing_value, list) and isinstance(requested_value, list):
        if len(existing_value) != len(requested_value):
            return False

        if path in RESUME_UNORDERED_LIST_PATHS:
            return sorted(existing_value) == sorted(requested_value)

        return all(
            _resume_values_match(
                existing_item,
                requested_item,
                path + (str(index),),
                ignore_paths=ignore_paths,
            )
            for index, (existing_item, requested_item) in enumerate(
                zip(existing_value, requested_value, strict=True)
            )
        )

    return False


def _masked_string_matches(existing_value: object, requested_value: object) -> bool:
    if not isinstance(existing_value, str) or not isinstance(requested_value, str):
        return False

    return _masked_string_matches_one_way(
        masked_value=existing_value, actual_value=requested_value
    ) or _masked_string_matches_one_way(
        masked_value=requested_value, actual_value=existing_value
    )


def _masked_string_matches_one_way(masked_value: str, actual_value: str) -> bool:
    if "****" not in masked_value:
        return False

    prefix, _, suffix = masked_value.partition("****")
    return (
        actual_value.startswith(prefix)
        and actual_value.endswith(suffix)
        and len(actual_value) >= len(prefix) + len(suffix)
    )


@contextmanager
def _patched_model_equality_for_resume(model_type: type[object]):
    original_eq = model_type.__eq__

    def _model_eq_for_resume(self: object, other: object):
        if not isinstance(other, model_type):
            return NotImplemented
        return _models_match_for_resume(self, other)

    model_type.__eq__ = _model_eq_for_resume  # type: ignore[method-assign]
    try:
        yield
    finally:
        model_type.__eq__ = original_eq  # type: ignore[method-assign]


@contextmanager
def _patched_trial_config_equality_for_resume():
    original_eq = TrialConfig.__eq__

    def _trial_config_eq_for_resume(self: object, other: object):
        if not isinstance(self, TrialConfig) or not isinstance(other, TrialConfig):
            return NotImplemented

        return (
            _models_match_for_resume(self.task, other.task)
            and self.trials_dir == other.trials_dir
            and self.timeout_multiplier == other.timeout_multiplier
            and self.agent_timeout_multiplier == other.agent_timeout_multiplier
            and self.verifier_timeout_multiplier == other.verifier_timeout_multiplier
            and self.agent_setup_timeout_multiplier
            == other.agent_setup_timeout_multiplier
            and self.environment_build_timeout_multiplier
            == other.environment_build_timeout_multiplier
            and _models_match_for_resume(self.agent, other.agent)
            and _models_match_for_resume(self.environment, other.environment)
            and _models_match_for_resume(self.verifier, other.verifier)
            and _resume_values_match(self.artifacts, other.artifacts)
        )

    TrialConfig.__eq__ = _trial_config_eq_for_resume  # type: ignore[method-assign]
    try:
        yield
    finally:
        TrialConfig.__eq__ = original_eq  # type: ignore[method-assign]


@contextmanager
def _patched_resume_equalities():
    with ExitStack() as stack:
        stack.enter_context(_patched_model_equality_for_resume(JobConfig))
        stack.enter_context(_patched_trial_config_equality_for_resume())
        yield


def _force_overwrite_configs(job_path: Path, config: JobConfig, config_dict: dict) -> None:
    # Overwrite existing job's config.json, and all trial config.json files with the latest environment and task variables.
    job_config_data = json.loads((job_path / "config.json").read_text())
    job_config_data["environment"] = config_dict["environment"]
    n_concurrent = (config_dict.get("orchestrator") or {}).get("n_concurrent_trials")
    if n_concurrent is not None:
        job_config_data["n_concurrent_trials"] = n_concurrent
    (job_path / "config.json").write_text(json.dumps(job_config_data, indent=4))
    print("finish overwrite")
    new_task_by_name: dict[str, dict] = {
        Path(task.path).name: task.model_dump(mode="json")
        for dataset in config.datasets
        for task in dataset.get_task_configs()
    }
    for trial_dir in job_path.iterdir():
        if not trial_dir.is_dir():
            continue
        trial_config_path = trial_dir / "config.json"
        if trial_config_path.exists():
            trial_config_data = json.loads(trial_config_path.read_text())
            trial_config_data["environment"] = job_config_data["environment"]
            task_name = Path(trial_config_data["task"]["path"]).name
            if task_name in new_task_by_name:
                trial_config_data["task"] = new_task_by_name[task_name]
                print(trial_config_data["task"])
            trial_config_path.write_text(json.dumps(trial_config_data, indent=4))


async def create_job_compat(config: JobConfig) -> Job:
    """
    Create a Harbor job across old and new Harbor versions.

    Harbor >= 2026-03-27 requires ``await Job.create(config)`` because job
    initialization now performs async dataset and task resolution. Older Harbor
    versions still support direct instantiation.
    """
    create = getattr(Job, "create", None)
    with _patched_resume_equalities():
        if callable(create):
            maybe_job = create(config)
            if inspect.isawaitable(maybe_job):
                return await maybe_job
            return maybe_job

        return Job(config=config)


async def create_job_compat(config: JobConfig) -> Job:
    """
    Create a Harbor job across old and new Harbor versions.

    Harbor >= 2026-03-27 requires ``await Job.create(config)`` because job
    initialization now performs async dataset and task resolution. Older Harbor
    versions still support direct instantiation.
    """
    create = getattr(Job, "create", None)
    if callable(create):
        maybe_job = create(config)
        if inspect.isawaitable(maybe_job):
            return await maybe_job
        return maybe_job

    return Job(config=config)


async def upload_trial_to_storage(result: TrialResult) -> str | None:
    """
    Upload trial directory as a tar.gz archive to Supabase storage and return public URL.
    Also uploads trajectory.json if it exists.

    Returns:
        Public URL of the trial archive in storage, or None if upload failed.
    """
    trial_path = Path(urlparse(result.trial_uri).path)

    if not trial_path.exists():
        print(f"Trial directory not found: {trial_path}")
        return None

    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    bucket_name = "trials"
    trial_id = str(result.id)

    # Create a temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Create tar.gz archive of the trial directory
        print(f"Creating tar.gz archive for trial {trial_id}...")
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(trial_path, arcname=trial_path.name)

        archive_size = tmp_path.stat().st_size
        print(f"Archive created: {archive_size / (1024 * 1024):.2f} MB")

        # Upload the archive with retry logic
        async def upload_archive():
            """Upload the tar.gz file with retry logic."""
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    with open(tmp_path, "rb") as f:
                        storage_path = f"{trial_id}.tar.gz"
                        response = await client.storage.from_(bucket_name).upload(
                            file=f, path=storage_path, file_options={"upsert": "true"}
                        )
                    return response

        await upload_archive()
        print(f"Successfully uploaded {trial_id}.tar.gz")

        # Upload trajectory.json if it exists
        trajectory_path = trial_path / "agent" / "trajectory.json"
        if trajectory_path.exists():
            print(f"Found trajectory.json for trial {trial_id}, uploading...")

            async def upload_trajectory():
                """Upload the trajectory.json file with retry logic."""
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    reraise=True,
                ):
                    with attempt:
                        with open(trajectory_path, "rb") as f:
                            storage_path = f"{trial_id}-traj.json"
                            response = await client.storage.from_(bucket_name).upload(
                                file=f,
                                path=storage_path,
                                file_options={"upsert": "true"},
                            )
                        return response

            try:
                await upload_trajectory()
                print(f"Successfully uploaded {trial_id}-traj.json")
            except Exception as e:
                print(f"Failed to upload trajectory.json for trial {trial_id}: {e}")

        # Get the public URL for the archive
        public_url = await client.storage.from_(bucket_name).get_public_url(
            f"{trial_id}.tar.gz"
        )

        return public_url

    except Exception as e:
        print(f"Failed to upload trial archive {trial_id}.tar.gz: {e}")
        return None

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def cleanup_codex_temp_runtime_dirs(result: TrialResult) -> None:
    """Drop Codex runtime dirs before archive upload."""
    trial_path = Path(urlparse(result.trial_uri).path)
    agent_tmp_path = trial_path / "agent" / "tmp"

    if not agent_tmp_path.exists():
        return

    removed_count = 0
    for codex_dir in agent_tmp_path.glob("arg*/codex-*"):
        if codex_dir.is_dir():
            try:
                shutil.rmtree(codex_dir)
                removed_count += 1
            except OSError as exc:
                print(f"Failed to remove Codex temp runtime dir {codex_dir}: {exc}")

    if removed_count:
        print(f"Removed {removed_count} Codex temp runtime dirs from {trial_path}")


def cleanup_terminus_session_artifacts(result: TrialResult) -> None:
    """Drop > 10MB Terminus-2 session logs before archive upload."""
    trial_path = Path(urlparse(result.trial_uri).path)
    agent_path = trial_path / "agent"

    total = 0
    artifact_paths = [
        agent_path / "recording.cast",
        agent_path / "terminus_2.pane",
    ]

    for path in artifact_paths:
        if path.is_file():
            total += path.stat().st_size

    if total > 10 * 1024 * 1024:
        print(
            f"Removing Terminus-2 session artifacts totaling "
            f"{total / (1024 * 1024):.2f} MB from {trial_path}"
        )
        for path in artifact_paths:
            try:
                if path.is_file():
                    path.unlink()
            except OSError as exc:
                print(f"Failed to remove Terminus-2 artifact {path}: {exc}")

async def insert_trial_into_db(event: TrialHookEvent):
    result = event.result
    cleanup_codex_temp_runtime_dirs(result)
    cleanup_terminus_session_artifacts(result)
    storage_url = await upload_trial_to_storage(result)

    trial_uri = storage_url
    if storage_url:
        pass
    else:
        print("Upload failed - trial_uri will be null")

    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    async def insert_agent():
        agent_insert = AgentInsert(
            name=result.agent_info.name,
            version=result.agent_info.version,
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("agent")
                    .upsert(
                        agent_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    async def get_dataset_task():
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("dataset_task")
                    .select("*, task!inner(name)")
                    .eq("dataset_name", "terminal-bench")
                    .eq("dataset_version", "2.0")
                    .eq("task.name", result.task_name)
                    .single()
                    .execute()
                )

    async def insert_trial():
        trial_insert = TrialInsert(
            id=result.id,
            agent_name=result.agent_info.name,
            agent_version=result.agent_info.version,
            config=result.config.model_dump(mode="json"),
            task_checksum=result.task_checksum,
            trial_name=result.trial_name,
            trial_uri=trial_uri,
            agent_execution_started_at=(
                result.agent_execution.started_at if result.agent_execution else None
            ),
            agent_execution_ended_at=(
                result.agent_execution.finished_at if result.agent_execution else None
            ),
            agent_setup_started_at=(
                result.agent_setup.started_at if result.agent_setup else None
            ),
            agent_setup_ended_at=(
                result.agent_setup.finished_at if result.agent_setup else None
            ),
            environment_setup_started_at=(
                result.environment_setup.started_at
                if result.environment_setup
                else None
            ),
            environment_setup_ended_at=(
                result.environment_setup.finished_at
                if result.environment_setup
                else None
            ),
            verifier_started_at=result.verifier.started_at if result.verifier else None,
            verifier_ended_at=result.verifier.finished_at if result.verifier else None,
            exception_info=(
                result.exception_info.model_dump(mode="json")
                if result.exception_info
                else None
            ),
            job_id=result.config.job_id,
            reward=(
                Decimal(result.verifier_result.rewards.get("reward", 0))
                if result.verifier_result and result.verifier_result.rewards is not None
                else None
            ),
            started_at=result.started_at,
            ended_at=result.finished_at,
            agent_metadata=(
                result.agent_result.metadata if result.agent_result else None
            ),
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("trial")
                    .insert(
                        trial_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    async def insert_model(name, provider, input_cost_per_token, output_cost_per_token):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("model")
                    .upsert(
                        ModelInsert(
                            name=name,
                            provider=provider,
                            cents_per_million_input_tokens=(
                                round(input_cost_per_token * 1e8)
                                if input_cost_per_token
                                else None
                            ),
                            cents_per_million_output_tokens=(
                                round(output_cost_per_token * 1e8)
                                if output_cost_per_token
                                else None
                            ),
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                    .execute()
                )

    async def insert_trial_model(name, provider):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("trial_model")
                    .insert(
                        TrialModelInsert(
                            trial_id=result.id,
                            model_name=name,  # type: ignore
                            model_provider=provider,  # type: ignore
                            n_cache_tokens=(
                                result.agent_result.n_cache_tokens
                                if result.agent_result
                                else None
                            ),
                            n_input_tokens=(
                                result.agent_result.n_input_tokens
                                if result.agent_result
                                else None
                            ),
                            n_output_tokens=(
                                result.agent_result.n_output_tokens
                                if result.agent_result
                                else None
                            ),
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                    .execute()
                )

    try:
        await insert_agent()
        await insert_trial()

        if result.agent_info.model_info:
            name = result.agent_info.model_info.name
            provider = result.agent_info.model_info.provider

            key = f"{provider}/{name}"
            token_costs = model_cost.get(key) or model_cost.get(name)

            input_cost_per_token = (
                token_costs.get("input_cost_per_token") if token_costs else None
            )
            output_cost_per_token = (
                token_costs.get("output_cost_per_token") if token_costs else None
            )

            if input_cost_per_token is None or output_cost_per_token is None:
                print(f"Could not find token costs for model: {key} or {name}")

            await insert_model(
                name, provider, input_cost_per_token, output_cost_per_token
            )
            await insert_trial_model(name, provider)

    except Exception as e:
        print(
            f"Failed to insert trial {result.trial_name} into database after 3 retries: {e}"
        )
        return


async def insert_job_into_db(job_insert: JobInsert):
    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    async def insert_job():
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("job")
                    .upsert(
                        job_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    try:
        await insert_job()
    except Exception as e:
        print(
            f"Failed to insert job {getattr(job_insert, 'job_name', 'unknown')} into database after 3 retries: {e}"
        )
        return


async def main():
    parser = argparse.ArgumentParser(
        description="Run a job with configurable job config"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="configs/job.yaml",
        help="Path to the job configuration file (default: configs/job.yaml)",
    )
    parser.add_argument(
        "-f",
        "--filter-error-types",
        action="append",
        help="Filter error types",
    )
    parser.add_argument(
        "--force-config",
        action="store_true",
        help="Overwrite existing job & trial config.json with the new config to avoid conflict",
    )

    args = parser.parse_args()

    config_path = args.config
    config_text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        config_dict = json.loads(config_text)
    else:
        config_dict = yaml.safe_load(config_text)

    config = JobConfig.model_validate(config_dict)

    job_path = config.jobs_dir / config.job_name

    if args.force_config and (job_path / "config.json").exists():
        _force_overwrite_configs(job_path, config, config_dict)

    if (job_path / "config.json").exists() and args.filter_error_types:
        existing_config = JobConfig.model_validate_json(
            (job_path / "config.json").read_text()
        )

        if existing_config != config:
            config_diff = "\n".join(
                format_config_diff(
                    existing_config.model_dump(mode="json"),
                    config.model_dump(mode="json"),
                )
            )
            raise ValueError(
                f"Job directory {job_path} already exists and cannot be "
                "resumed with a different config.\n\n"
                f"Config diff:\n{config_diff}"
            )

        filter_error_types_set = set(args.filter_error_types)
        for trial_dir in job_path.iterdir():
            if not trial_dir.is_dir():
                continue

            trial_paths = TrialPaths(trial_dir)

            if not trial_paths.result_path.exists():
                continue

            try:
                trial_result = TrialResult.model_validate_json(
                    trial_paths.result_path.read_text()
                )
            except Exception:
                print(
                    f"Failed to parse trial result {trial_dir.name}. Removing trial "
                    "directory."
                )
                shutil.rmtree(trial_dir)
                continue

            if (
                trial_result.exception_info is not None
                and trial_result.exception_info.exception_type in filter_error_types_set
            ):
                print(
                    f"Removing trial directory with "
                    f"{trial_result.exception_info.exception_type}: {trial_dir.name}"
                )
                shutil.rmtree(trial_dir)

    job = await create_job_compat(config)

    job_insert = JobInsert(
        id=job._id,
        config=config.model_dump(mode="json"),
        job_name=config.job_name,
        n_trials=len(job),
        username=os.environ.get("USER", "unknown"),
        git_commit_id=(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(harbor.__file__).parent,
            )
            .decode("utf-8")
            .strip()
        ),
        package_version=harbor.__version__,
    )

    if not job.is_resuming:
        await insert_job_into_db(job_insert)

    job.add_hook(
        event=TrialEvent.END,
        callback=insert_trial_into_db,
    )

    result = await job.run()

    job_insert.started_at = result.started_at
    job_insert.ended_at = result.finished_at
    job_insert.stats = result.stats.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )

    await insert_job_into_db(job_insert)


if __name__ == "__main__":
    asyncio.run(main())
