"""
Custom Actions for RAG Security Guardrails
CSonEmbedding Project
"""

from typing import Optional
from nemoguardrails.actions import action


@action(is_system_action=True)
async def self_check_input(context: Optional[dict] = None) -> bool:
    """
    检查用户输入是否安全
    返回 True 表示允许，False 表示阻止
    """
    user_input = context.get("user_message", "") if context else ""
    
    # 基础关键词检测
    blocked_patterns = [
        "ignore previous",
        "forget your",
        "system prompt",
        "reveal instructions",
        "you are now",
        "act as",
        "pretend to be",
        "jailbreak",
        "DAN mode",
    ]
    
    user_input_lower = user_input.lower()
    for pattern in blocked_patterns:
        if pattern in user_input_lower:
            return False
    
    return True


@action(is_system_action=True)
async def self_check_output(context: Optional[dict] = None) -> bool:
    """
    检查模型输出是否安全
    返回 True 表示允许，False 表示阻止
    """
    bot_response = context.get("bot_message", "") if context else ""
    
    # 检测敏感输出
    blocked_patterns = [
        "system prompt:",
        "my instructions are:",
        "i was programmed to",
        "here is my configuration",
    ]
    
    bot_response_lower = bot_response.lower()
    for pattern in blocked_patterns:
        if pattern in bot_response_lower:
            return False
    
    return True


@action(is_system_action=True)
async def self_check_retrieval(context: Optional[dict] = None) -> bool:
    """
    检查RAG检索内容是否安全
    返回 True 表示允许，False 表示阻止
    """
    retrieved_content = context.get("relevant_chunks", []) if context else []
    
    # 检测检索内容中的恶意指令
    injection_patterns = [
        "ignore the above",
        "disregard previous",
        "new instructions:",
        "system override",
        "<script>",
        "eval(",
        "exec(",
    ]
    
    for chunk in retrieved_content:
        chunk_text = str(chunk).lower()
        for pattern in injection_patterns:
            if pattern in chunk_text:
                return False
    
    return True


@action()
async def retrieve_relevant_chunks(
    context: Optional[dict] = None,
    kb: Optional[object] = None
) -> list:
    """
    从知识库检索相关内容
    """
    user_input = context.get("user_message", "") if context else ""
    
    if kb is None:
        return []
    
    # 使用知识库检索
    try:
        results = await kb.search(user_input, max_results=5)
        return results
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
