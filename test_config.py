"""测试NeMo Guardrails配置加载"""

from nemoguardrails import RailsConfig

try:
    config = RailsConfig.from_path("configs/rag_guard")
    print("✅ 配置加载成功!")
    print(f"   Models: {config.models}")
    print(f"   Rails: {config.rails}")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")
