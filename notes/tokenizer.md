# Tokenizer 设计

## Tokenizer class

The `from_files()` method is designed to be a class method, which is bound to the class and not the instance of the class. For a class method, we should put a `@classmethod` as a decorator above the method's name.


## `json` 库

- `json.loads()`：处理 **内存中** 的 JSON **字符串** (string)。
- `json.load()`：处理 **文件流** 中的 JSON **数据** (file-like object)。

函数名结尾的 **`s`** 就是 **`string`** 的意思。有 `s` 的处理字符串，没 `s` 的处理文件。

一个json数据可能作为一个字符串变量储存在python代码里，也有可能作为一个.json文件保存在电脑上。json.loads() 和 json.load() 就是分别用来处理这两种情况的。

当你的JSON数据是一个Python字符串时，就用json.loads()

## bytes type

ATTENTION: bytes 在进行迭代的时候，返回的不是bytes而是一个整数！

