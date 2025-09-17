# Tokenizer 设计

## Tokenizer class

The `from_files()` method is designed to be a class method, which is bound to the class and not the instance of the class. For a class method, we should put a `@classmethod` as a decorator above the method's name.


