# Tokenizer Design

## Tokenizer class

### `from_files()`
The `from_files()` method is designed to be a class method, which is bound to the class and not the instance of the class. For a class method, we should put a `@classmethod` as a decorator above the method's name.


### param `self.special_pattern`
```python
self.special_pattern = ""
if self.special_tokens:
	self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
	self.special_pattern = "|".join(re.escape(st) for st in self.special_tokens)
```

The parameter is used in the `encode` method to split the input text into different parts, ensuring that special tokens are identified and handled separately from regualr text.

1. Check if special_tokens is not None or empty
2. Sorts the list of special_tokens in descending order of their lengths. This ensures that longer tokens are matched first when using the regular expression. For example, for special tokens ["\<pad>", "\<p>"], soring ensures "\<pad>" is matched before "\<p>"
3. `self.special_pattern = "|".join(re.escape(st) for st in self.special_tokens)`: 
	- `re.escape()` ensures that any special characteres in the token are escaped so that they are treated as literal chararcters in the regex.
	- `"|".join(...)` conbines all the excaped tokens into **a single regex pattern** separated by `|` (**logical OR**)


## `json` 库

- `json.loads()`：处理 **内存中** 的 JSON **字符串** (string)。
- `json.load()`：处理 **文件流** 中的 JSON **数据** (file-like object)。

函数名结尾的 **`s`** 就是 **`string`** 的意思。有 `s` 的处理字符串，没 `s` 的处理文件。

一个json数据可能作为一个字符串变量储存在python代码里，也有可能作为一个.json文件保存在电脑上。json.loads() 和 json.load() 就是分别用来处理这两种情况的。

当JSON数据是一个Python字符串时，就用json.loads()

## bytes type

**ATTENTION**: 
- bytes 类型的变量在进行迭代的时候，返回的不是 bytes 而是一个int！ `for b in b'hello'` -> b 是 int

- 而 bytes() 构造函数接收的是一个**包含整数的可迭代对象**!
	- `[bytes([b]) for b in b'hello']` 可以正确构造一个由每个bytes构成的列表. 如: bytes([72]) 的意思是：创建一个 bytes 对象，它的内容由列表 [72] 中的整数决定。结果是 `b'h'`。

	- `[bytes(b) for b in b'hello']` 错误, 每次会产生一个长度为 b,内容全部为 0x00 的bytes 对象. 如: bytes(72) 的意思是：创建一个长度为72的 bytes 对象，并用零字节 (\x00) 填充它。结果是 `b'\x00\x00\x00\x00\x00\x00...'` (总共72个 \x00)。

