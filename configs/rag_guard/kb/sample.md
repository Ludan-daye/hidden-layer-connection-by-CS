# CSonEmbedding 知识库示例文档

## 项目简介

CSonEmbedding 是一个基于压缩感知技术的RAG安全防御系统。该系统旨在保护大语言模型应用免受各种安全威胁。

## 主要功能

1. **Embedding安全防御**: 防止embedding反演攻击和成员推断攻击
2. **Prompt注入防护**: 检测和阻止恶意prompt注入
3. **检索内容过滤**: 确保RAG检索的内容安全可靠
4. **输入输出审核**: 对用户输入和模型输出进行安全检查

## 技术架构

- 使用NeMo Guardrails作为安全防护框架
- 支持多种LLM后端（OpenAI、本地模型等）
- 集成FastEmbed进行高效向量检索

## 使用指南

启动Guardrails服务：
```bash
nemoguardrails server --config configs/rag_guard --port 8000
```

## 安全策略

本系统遵循以下安全原则：
- 最小权限原则
- 输入验证
- 输出过滤
- 日志审计
