
from src.plugin_system.base.base_command import BaseCommand
from src.common.logger import get_logger

logger = get_logger("A_Memorix.DebugServerCommand")

class DebugServerCommand(BaseCommand):
    command_name = "debug_server"
    command_description = "调试启动 Web Server"
    command_pattern = r"^/debug_server$"

    async def execute(self):
        try:
            from ...plugin import A_MemorixPlugin
            plugin = A_MemorixPlugin.get_global_instance()
            
            if not plugin:
                return True, "❌ 无法获取插件实例", 0
                
            status = []
            status.append(f"Plugin initialized: {plugin._initialized}")
            status.append(f"Server instance: {plugin.server}")

            # 先确保核心存储已初始化，避免只启动了 WebUI 但导入依赖为空
            if not getattr(plugin, "_initialized", False):
                status.append("Storage not initialized, trying async initialization ...")
                try:
                    await plugin._ensure_initialized()
                    status.append(f"Storage initialized: {plugin._initialized}")
                except Exception as e:
                    status.append(f"❌ _ensure_initialized failed: {e}")

            # 尝试强制启动
            if not plugin.server:
                status.append("Attempting to start server...")
                try:
                    from ...server import MemorixServer
                    plugin.server = MemorixServer(plugin)
                    plugin.server.start()
                    status.append("Server thread started.")
                except Exception as e:
                    status.append(f"❌ Start failed: {e}")
                    import traceback
                    status.append(traceback.format_exc())
            else:
                status.append("Server already exists.")
                
            return True, "\n".join(status), 0
            
        except Exception as e:
            return True, f"Command Error: {e}", 0
