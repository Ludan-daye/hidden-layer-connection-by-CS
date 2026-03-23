"""测试Embedding输入输出维度"""

from fastembed import TextEmbedding

# 使用NeMo Guardrails默认的embedding模型
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# 测试输入
test_inputs = [
    "Hello world",
    "这是一个中文测试句子",
    "This is a longer sentence to test the embedding dimension output.",
]

print("=" * 60)
print("Embedding 维度测试")
print("=" * 60)

for text in test_inputs:
    embeddings = list(model.embed([text]))
    emb = embeddings[0]
    
    print(f"\n输入: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    print(f"  输入字符数: {len(text)}")
    print(f"  输出维度: {len(emb)}")
    print(f"  存储大小: {len(emb) * 4} bytes (float32)")

print("\n" + "=" * 60)
print("常用Embedding模型维度对比:")
print("=" * 60)
print(f"  BAAI/bge-small-en-v1.5:  384 维  -> {384*4} bytes/向量")
print(f"  BAAI/bge-base-en-v1.5:   768 维  -> {768*4} bytes/向量")
print(f"  BAAI/bge-large-en-v1.5:  1024 维 -> {1024*4} bytes/向量")
print(f"  OpenAI text-embedding-3-small: 1536 维 -> {1536*4} bytes/向量")
print(f"  OpenAI text-embedding-3-large: 3072 维 -> {3072*4} bytes/向量")
