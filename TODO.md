TODO List for DIRE update
=========================

- [ ] Restructure layout so (our) data and AST API is more portable. Right now
      it's under
      `dataset-gen/decompiler/{function.py,ida_ast.py,typeinfo.py,variable.py}`,
      but it should be top-level to decouple it from data collection.
- [ ] Test data deserialization.
- [ ] Implement `TypeLib.get_replacements()`. This function should take as input
      memory layout and output a generator of lists of possible type
      replacements.
- [x] Output serialized data directly to compressed files.
- [ ] (Maybe) API calls for `CollectedFunction` that aligns debug and decompiler
      variable names. At the moment you can just compare `Location`s; if they're
      in the same location then they're the same variable. Note that it's
      possible to have multiple variables at the same location. This is fine,
      they use the same memory location but their lifetimes don't overlap.
- [ ] Serialization is a bit all-over-the-place. To serialize `Function`s, you
      just call `Function.to_json()` on an instance of a `Function` object and
      it returns a dictionary that can be passed to a standard JSON encoder. To
      serialize `TypeLib`s you have to use `TypeLibCodec.encode()` (or worse,
      `TypeLibCodec.read_metadata()` if you want to just deserialize a single
      `TypeInfo`). I don't like
      this (older) method, so I'd like to change it to be more in line with
      everything else.
