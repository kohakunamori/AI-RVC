"""角色资产校验：文件存在不等于有效模型，不以文件名推断 RVC 架构。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def model_files(directory: Path) -> tuple[Path, Path | None]:
    directory = Path(directory)
    weights = sorted(directory.rglob('*.pth')) if directory.is_dir() else []
    if not weights:
        raise FileNotFoundError(f'角色目录没有 .pth 权重：{directory}')
    if len(weights) != 1:
        raise ValueError(f'角色目录包含多个 .pth，必须明确拆分，不能任选第一个：{directory}')
    indices = sorted(directory.rglob('*.index'))
    for path in weights + indices:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'角色资产路径越界：{path}')
    if len(indices) == 2:
        added = [path for path in indices if path.name.startswith('added_')]
        if len(added) == 1:
            trained = added[0].with_name('trained_' + added[0].name[len('added_'):])
            if trained in indices and _is_empty_training_index(_stamp(trained)):
                indices = added
    matching = [path for path in indices if path.stem.casefold() == weights[0].stem.casefold()]
    if len(matching) == 1 and len(indices) == 1:
        index = matching[0]
    elif len(indices) == 1:
        index = indices[0]
    elif not indices:
        index = None
    else:
        raise ValueError(f'角色目录包含多个索引，不能猜测对应关系：{directory}')
    return weights[0], index


def _stamp(path: Path | None):
    if path is None:
        return None
    from tools.model_assets import file_sha256
    return str(path.resolve()), file_sha256(path)


@lru_cache(maxsize=64)
def _is_empty_training_index(stamp: tuple) -> bool:
    import faiss

    index = faiss.read_index(stamp[0])
    return (type(index) is faiss.IndexIVFFlat and index.is_trained
            and index.metric_type == faiss.METRIC_L2 and index.ntotal == 0)


@lru_cache(maxsize=64)
def _inspect_cached(weight_stamp: tuple, index_stamp: tuple | None) -> dict:
    import torch
    from infer.contracts import inspect_checkpoint, read_index
    from infer.rvc_checkpoint import safe_load_rvc_checkpoint, prepare_rvc_checkpoint

    path = Path(weight_stamp[0])
    checkpoint, compatibility = prepare_rvc_checkpoint(
        safe_load_rvc_checkpoint(path), model_label=str(path), validate_structure=True,
    )
    contract = inspect_checkpoint(checkpoint, str(path))
    result = {'version': contract.version, 'feature_dim': contract.feature_dim,
              'sample_rate': contract.sample_rate, 'speakers': contract.speaker_count,
              'uses_f0': contract.uses_f0,
              'dtypes': sorted({str(v.dtype) for v in checkpoint['weight'].values()}),
              'model_path': str(path), 'index_path': index_stamp[0] if index_stamp else None,
              'index_vectors': 0, 'runtime_verified': False,
              'vocoder': compatibility.vocoder,
              'runtime_architecture': compatibility.runtime_architecture}
    if index_stamp:
        index, _ = read_index(index_stamp[0], contract.feature_dim, min_vectors=8)
        result['index_vectors'] = int(index.ntotal)
    return result


def inspect_model_assets(weight_path: Path, index_path: Path | None = None, *, require_index: bool = False) -> dict:
    if require_index and index_path is None:
        raise FileNotFoundError('该角色下载配置包含索引，但本地缺少 .index 文件')
    return dict(_inspect_cached(_stamp(Path(weight_path)), _stamp(Path(index_path)) if index_path else None))


def inspect_character_directory(directory: Path, *, require_index: bool = False) -> dict:
    weight, index = model_files(directory)
    return inspect_model_assets(weight, index, require_index=require_index)
