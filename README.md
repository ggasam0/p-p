# p-p
# test_all_chunkers.py
```python
import sys
import traceback

from chonkie import (
    TokenChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    LateChunker,
    CodeChunker,
    NeuralChunker,
    SlumberChunker,
)

def test_chunker(chunker_class, test_text, **kwargs):
    print(f"Testing {chunker_class.__name__} with kwargs={kwargs} …")
    try:
        chunker = chunker_class(**kwargs)
        chunks = chunker(test_text)
        print(f" → OK: got {len(chunks)} chunks")
        # 可选：打印前几个 chunk 的文本 + token_count
        for idx, ch in enumerate(chunks[:3]):
            print(f"    chunk[{idx}].text = {repr(ch.text)}")
            print(f"    chunk[{idx}].token_count = {ch.token_count}")
        return True
    except Exception as e:
        print(f" ✘ ERROR: {e}")
        traceback.print_exc(limit=2, file=sys.stdout)
        return False

def main():
    test_text = (
        "This is a sample text to test chunking. "
        "It contains multiple sentences, and maybe some code: `print('hello')`. "
        "We want to ensure each chunker works."
    )
    # 针对 CodeChunker，可能用代码样本
    code_text = (
        "def foo(x):\n    return x * x\n\n"
        "class Bar:\n    def __init__(self, y):\n        self.y = y\n"
    )

    # 列出要测试的 chunker 和特定文本
    tests = [
        (TokenChunker, test_text, {}),
        (SentenceChunker, test_text, {}),
        (RecursiveChunker, test_text, {}),
        (SemanticChunker, test_text, {"chunk_size": 50}),  # 示例传参
        (LateChunker, test_text, {"chunk_size": 50}),
        (CodeChunker, code_text, {}),
        (NeuralChunker, test_text, {}),
        (SlumberChunker, test_text, {}),
    ]

    results = {}
    for cls, text, kwargs in tests:
        ok = test_chunker(cls, text, **kwargs)
        results[cls.__name__] = ok

    print("\nSummary:")
    for name, ok in results.items():
        print(f" {name}: {'✅' if ok else '❌'}")

    # Exit non-zero if any failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
```
