"""提供仅插件构建场景下的兼容命令行入口。"""

import sys


def main() -> int:
    """提示当前构建需由宿主插件系统加载。

    Returns:
        int: 兼容 CLI 的退出码，固定返回非零值表示不可直接启动。
    """
    sys.stderr.write(
        "na_memorix is packaged as a Nekro Agent plugin only.\n"
        "Load it through the host runtime instead of launching the amemorix module directly.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
