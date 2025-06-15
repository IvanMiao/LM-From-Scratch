## 2 BPE Tokenizer

### 2.1 Unicode

(a) What Unicode character does `chr(0)` return?
`'\x00'`

(b) How does this character's string representation (__repr__()) differ from its printed representation?
`chr(0).__repr__()` -> `"'\\x00'"`
`print(chr(0))` -> `(nothing)`

(C) 
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

`chr(0)` -> `'\x00'`
`print(chr(0))` -> `(nothing)`
`"this is a test" + chr(0) + "string"` -> `'this is a test\x00string'`
`print("this is a test" + chr(0) + "string")` -> `this is a teststring`

This character will be ignored when calling `print()` to print it.

### 2.2 Unicode Encodings

(a) It's more economic when using UTF-8, since UTF-16 always use more than 2 bytes and UTF-32 4 bytes.

(b) Incorrect function:
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Suppose: The function will decode each single byte, but thers are some characters in UTF-8 are encoded by >=2 bytes.

(c) Two byte sequence that does not decode to any Unicode character(s):
#1 

#2 

### 2.3 Subword Tokenization

