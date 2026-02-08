import os
from typing import Union, List, Dict, Any

import fsspec
import tosfs
import yaml


class Cfg():
    cfg: Dict[str, Any] = None

    @staticmethod
    def load(cfg_path: str) -> Dict[str, Any]:
        if Cfg.cfg is None:
            assert os.path.exists(cfg_path), f"{cfg_path} not exist"
            with open(cfg_path) as f:
                Cfg.cfg = yaml.safe_load(f)
        return Cfg.cfg

def get_filesystem(path: Union[str, List[str]], cfg: Dict[str, Any]) -> fsspec.AbstractFileSystem:
    path = path
    if isinstance(path, list):
        path = path[0]
    if str(path).startswith("s3://"):
        assert "internal" in cfg["oss"]["endpoint_url"] or cfg["oss"]["endpoint_url"]
        # file_system = s3fs.S3FileSystem(
        #     access_key=cfg["oss"]["ak"],
        #     secret_key=cfg["oss"]["sk"],
        #     endpoint_url=cfg["oss"]["endpoint_url"]
        # )
        file_system = fsspec.filesystem(
            "s3",
            key=cfg["oss"]["ak"],
            secret=cfg["oss"]["sk"],
            client_kwargs={
                "endpoint_url": cfg["oss"]["endpoint_url"]
            },
            region_name=cfg["oss"]["region"]
        )
    elif str(path).startswith("tos"):
        assert "ivolces" in cfg["tos"]["endpoint"]
        file_system = tosfs.TosFileSystem(
            key=cfg["tos"]["ak"],
            secret=cfg["tos"]["sk"],
            # 只需要tos服务地址，不包含bucket
            endpoint=cfg["tos"]["endpoint"],
            region=cfg["tos"]["region"]
        )

    else:

        return fsspec.filesystem("file")
    return file_system



