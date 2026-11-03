"""What the handed-callable analysis does with each form of code it reads,
and with each kind of value it meets: one closed table, which the analysis
consults (:mod:`.ticking`, :mod:`.handed`) and a test holds against every
node class of the running Python.

Each node class has one :class:`Rule`; a class the table does not name gets
:attr:`Rule.FALLBACK`, so a form no one listed makes the play tick
everything. The analysis is flow-insensitive: it reads every node of a
function the play reaches (its callbacks and the functions of the scene's
they call), in any order, as if every branch ran.

The value kinds the rules name are listed below the table
(:data:`FREE_KINDS`, :data:`INERT_KINDS`, :data:`NATIVE_HOLDERS`,
:data:`ITEM_KINDS`), with the names reaching a namespace
(:data:`NAMESPACE_ATTRIBUTES`, :data:`NAMESPACE_BUILTINS`).

What the analysis resolves when the play is planned stands for the whole
play only when no code can change it. A value of a kind not inert (what a
list, a dict, an object or a class holds) read then makes the play tick
everything, whatever would change it
(:meth:`.ticking._Writes.plan_reads_hold`); a binding (a closure's cell, a
module's global, a function's default) changes only through forms the
rules close.

:func:`proven_pure` and :func:`result_kept` answer from the same table for
callers outside the stepping check, as filters for precision only: nothing
in the stepping check rests on them, and each has known limits (the
tech-debt tracker lists them).
"""

from __future__ import annotations

__all__ = ["OPERATIONS", "Rule", "proven_pure", "result_kept", "rule_of"]

import ast
import builtins
import collections
import enum
import fractions
import inspect
import pathlib
import re
import types
from typing import Any

import numpy as np


class Rule(enum.Enum):
    """What the analysis does with a node of a class, as an expression
    whose value may be handed to a call (*held*: the callables it may be or
    hold, or "anything"; *inert*: a value of an :data:`INERT_KINDS` kind or
    a NumPy array of numbers; *carried*: the parameters of a callee whose
    objects it may give back) and as a statement of a function the play
    reaches.
    """

    CONSTANT = (
        "a literal or a formatted string: holds no callable; inert when its"
        " value is of an INERT_KINDS kind (a formatted string is a str);"
        " carries nothing; a formatted value runs its PROTOCOL"
    )
    LAMBDA = (
        "a function made in place: its body is read as code of the function"
        " holding it (its calls are that function's calls), so, called where"
        " it is made, it holds no callable the analysis does not read; stored,"
        " it is found by what reads what holds it, which no plan rests on"
        " (ATTRIBUTE, SUBSCRIPT); never inert; carries nothing"
    )
    PARTS = (
        "an operator, a display, a comprehension, a slice, a starred or"
        " named value: holds every callable any part holds (anything, if any"
        " part may hold anything): an operator may give back an operand, a"
        " [f] * n among them; inert when every part is; carries what its"
        " parts carry; an operator runs its operands' PROTOCOL (a boolean"
        " operator their truth test), a slice its bounds', and a starred"
        " value or a mapping unpacked into a display is iterated (ITERATION)"
    )
    GENERATOR = (
        "a generator expression: holds what its item holds (its code is read"
        " in place, like a lambda's body); never inert (a generator runs code"
        " as it is read); carries what its parts carry; its iterables are"
        " iterated (ITERATION)"
    )
    COMPARE = (
        "a comparison: a truth value unless an operand's own comparison"
        " makes another, so it holds nothing unless an operand may hold"
        " anything; inert when every operand is; runs its operands' PROTOCOL,"
        " and the right operand of in and not in is iterated (ITERATION)"
    )
    NAME = (
        "a name: a parameter holds what its call site hands it (judged"
        " there); the owner, what it holds; a local name, every value any of"
        " its stores binds (anything for a binding no expression gives);"
        " another name, what its binding (a cell, a global, a default) holds"
        " when the play is planned, which changes only by OUTER_REBIND, by a"
        " store into an attribute of that name (STORE), or through a"
        " namespace (NAMESPACE_ATTRIBUTES, NAMESPACE_BUILTINS); naming a"
        " NAMESPACE_BUILTINS builtin makes the play tick everything"
    )
    ATTRIBUTE = (
        "an attribute: the object's own attribute when the analysis resolves"
        " it at plan time; that reads what the object holds, which makes the"
        " play tick everything unless the object is a module (a global's"
        " binding, NAME) or the attribute is a mobject's member list (which"
        " every stepping tick compares); one the object does not hold is read"
        " through its class, a binding a store of that name may change"
        " (STORE); else anything; inert only when it resolves to an inert"
        " value; reading it runs its object's PROTOCOL; a"
        " NAMESPACE_ATTRIBUTES attribute makes the play tick everything"
    )
    SUBSCRIPT = (
        "an item: read only on a base of an ITEM_KINDS kind, an inert value"
        " or a mobject's members; on any other base, a load, store or"
        " deletion of an item counts as a call of unknown code (its class's"
        " __getitem__, __missing__, __setitem__); holds what its base holds;"
        " its key runs its PROTOCOL (hash, index)"
    )
    CALL = (
        "a call: the callables its arguments hold are read as called where"
        " its callee may call them, and an argument not shown free of"
        " callables (a generator or an iterator among them, which a native"
        " consuming it runs; an object of the scene's class, whose PROTOCOL"
        " it may run) makes it an unknown call, whatever its callee; its"
        " value holds nothing when it is inert (len, math, the engine's"
        " getters, a method of an inert value, NumPy's makers and ufuncs of"
        " inert values), else anything; a function of the scene's never"
        " gives an inert value, whatever its code (a proof over code, not a"
        " kind); a starred argument is iterated (ITERATION)"
    )
    SENT = (
        "a value sent into a generator or awaited (await, yield, yield"
        " from): holds anything; never inert; carries what its parts carry;"
        " awaiting is an unknown call, and yield from iterates (ITERATION)"
    )
    STORE = (
        "an assignment or deletion: a name target binds the name (NAME); a"
        " store into an object's attribute or item changes what the object"
        " holds, on which no plan rests (ATTRIBUTE); a store of a value not"
        " inert into an attribute of a module or a class named as such makes"
        " the play tick everything, and into an attribute of any other"
        " object (an operator in place's whatever its operand), makes it tick"
        " everything when the plan read a global, or an attribute through a"
        " class, of that name; an item target follows SUBSCRIPT; a tuple or"
        " list target iterates the value (ITERATION); an operator in place"
        " runs its operands' PROTOCOL"
    )
    BINDING = (
        "a statement or clause binding names (a loop, a with, an import, a"
        " definition, an except clause, a pattern, a type parameter): a name"
        " it binds holds what NAME says (a loop variable, its iterable's"
        " items; anything for a binding no expression gives); an attribute"
        " or item it binds is a STORE; a loop's or a comprehension clause's"
        " iterable is iterated, and a tuple or list target iterates each"
        " item (ITERATION); a comprehension clause's conditions run their"
        " truth test (PROTOCOL); entering a with is an unknown call"
    )
    OUTER_REBIND = (
        "a global or nonlocal declaration: the play ticks everything unless"
        " every value the function binds the declared names to is inert"
    )
    CONTROL = (
        "control flow, a test, a raise, an expression statement, a return"
        " (whose value a function's returns are read from): the analysis is"
        " flow-insensitive, so it reads every branch and adds nothing; a"
        " test (if, while, assert) runs its truth test (PROTOCOL)"
    )
    PART = (
        "a part of the node holding it (a context, an operator, a"
        " comprehension clause, arguments, a keyword, an alias, a with item,"
        " a match case or pattern part), read by that node's rule"
    )
    NOT_PARSED = (
        "never in a parsed function body: an abstract base, a module root, a"
        " type comment, a class kept only for old code"
    )
    FALLBACK = "a form the analysis does not read: the play ticks everything"


#: What iterating does wherever a form iterates (a loop, a comprehension,
#: ``in``, a starred value, an unpacking target, ``yield from``), which the
#: rules above name: iterating a value runs its kind's code, so it counts as
#: calling it. Nothing for an inert value, a display, an exact native holder
#: or range, an object of the engine's class, the result of a builtin making
#: a container or an iterator (what it consumes is its argument, judged where
#: it is handed) or of the engine's code, a parameter (its call site hands a
#: value shown free of callables); an unknown call on anything else (a
#: generator, an iterator, an object of the scene's class).
ITERATION = "ITERATION"
#: What Python runs on an operand by syntax, which the rules above name: an
#: operator's, a comparison's, a truth test's, a formatting's, a hash's or
#: an index's method of its class, and an attribute read's
#: (``__getattribute__``, ``__getattr__``, a property's getter, another
#: descriptor's ``__get__``). An unknown call when the operand may be an
#: object whose class defines it in the scene's code, or a container
#: (beyond a truth test, which only counts its items) or an array of objects
#: whose items may; nothing for an inert value, a function made in place,
#: an operation's result (its operands are judged where it is), the
#: engine's code's result, a parameter.
PROTOCOL = "PROTOCOL"


_R = Rule
#: Each node class of :mod:`ast`, by name, and its rule.
OPERATIONS: dict[str, Rule] = {
    # Values.
    "Constant": _R.CONSTANT,
    "JoinedStr": _R.CONSTANT,
    "Lambda": _R.LAMBDA,
    "BinOp": _R.PARTS,
    "UnaryOp": _R.PARTS,
    "BoolOp": _R.PARTS,
    "IfExp": _R.PARTS,
    "NamedExpr": _R.PARTS,
    "Starred": _R.PARTS,
    "List": _R.PARTS,
    "Tuple": _R.PARTS,
    "Set": _R.PARTS,
    "Dict": _R.PARTS,
    "ListComp": _R.PARTS,
    "SetComp": _R.PARTS,
    "DictComp": _R.PARTS,
    "Slice": _R.PARTS,
    "TemplateStr": _R.PARTS,
    "GeneratorExp": _R.GENERATOR,
    "Compare": _R.COMPARE,
    "Name": _R.NAME,
    "Attribute": _R.ATTRIBUTE,
    "Subscript": _R.SUBSCRIPT,
    "Call": _R.CALL,
    "Await": _R.SENT,
    "Yield": _R.SENT,
    "YieldFrom": _R.SENT,
    # Statements.
    "Assign": _R.STORE,
    "AnnAssign": _R.STORE,
    "AugAssign": _R.STORE,
    "Delete": _R.STORE,
    "For": _R.BINDING,
    "AsyncFor": _R.BINDING,
    "With": _R.BINDING,
    "AsyncWith": _R.BINDING,
    "Import": _R.BINDING,
    "ImportFrom": _R.BINDING,
    "FunctionDef": _R.BINDING,
    "AsyncFunctionDef": _R.BINDING,
    "ClassDef": _R.BINDING,
    "TypeAlias": _R.BINDING,
    "ExceptHandler": _R.BINDING,
    "comprehension": _R.BINDING,
    "withitem": _R.BINDING,
    "MatchAs": _R.BINDING,
    "MatchStar": _R.BINDING,
    "MatchMapping": _R.BINDING,
    "TypeVar": _R.BINDING,
    "ParamSpec": _R.BINDING,
    "TypeVarTuple": _R.BINDING,
    "Global": _R.OUTER_REBIND,
    "Nonlocal": _R.OUTER_REBIND,
    "Return": _R.CONTROL,
    "If": _R.CONTROL,
    "While": _R.CONTROL,
    "Try": _R.CONTROL,
    "TryStar": _R.CONTROL,
    "Raise": _R.CONTROL,
    "Assert": _R.CONTROL,
    "Pass": _R.CONTROL,
    "Break": _R.CONTROL,
    "Continue": _R.CONTROL,
    "Expr": _R.CONTROL,
    "Match": _R.CONTROL,
    # Parts of the node holding them.
    "FormattedValue": _R.PART,
    "Interpolation": _R.PART,
    "Load": _R.PART,
    "Store": _R.PART,
    "Del": _R.PART,
    "And": _R.PART,
    "Or": _R.PART,
    "Add": _R.PART,
    "Sub": _R.PART,
    "Mult": _R.PART,
    "MatMult": _R.PART,
    "Div": _R.PART,
    "Mod": _R.PART,
    "Pow": _R.PART,
    "LShift": _R.PART,
    "RShift": _R.PART,
    "BitOr": _R.PART,
    "BitXor": _R.PART,
    "BitAnd": _R.PART,
    "FloorDiv": _R.PART,
    "Invert": _R.PART,
    "Not": _R.PART,
    "UAdd": _R.PART,
    "USub": _R.PART,
    "Eq": _R.PART,
    "NotEq": _R.PART,
    "Lt": _R.PART,
    "LtE": _R.PART,
    "Gt": _R.PART,
    "GtE": _R.PART,
    "Is": _R.PART,
    "IsNot": _R.PART,
    "In": _R.PART,
    "NotIn": _R.PART,  # codespell:ignore
    "arguments": _R.PART,
    "arg": _R.PART,
    "keyword": _R.PART,
    "alias": _R.PART,
    "match_case": _R.PART,
    "MatchValue": _R.PART,
    "MatchSingleton": _R.PART,
    "MatchSequence": _R.PART,
    "MatchClass": _R.PART,
    "MatchOr": _R.PART,
    # Never in a parsed function body.
    "AST": _R.NOT_PARSED,
    "mod": _R.NOT_PARSED,
    "Module": _R.NOT_PARSED,
    "Interactive": _R.NOT_PARSED,
    "Expression": _R.NOT_PARSED,
    "FunctionType": _R.NOT_PARSED,
    "Suite": _R.NOT_PARSED,
    "stmt": _R.NOT_PARSED,
    "expr": _R.NOT_PARSED,
    "expr_context": _R.NOT_PARSED,
    "AugLoad": _R.NOT_PARSED,
    "AugStore": _R.NOT_PARSED,
    "Param": _R.NOT_PARSED,
    "boolop": _R.NOT_PARSED,
    "operator": _R.NOT_PARSED,
    "unaryop": _R.NOT_PARSED,
    "cmpop": _R.NOT_PARSED,
    "excepthandler": _R.NOT_PARSED,
    "pattern": _R.NOT_PARSED,
    "type_ignore": _R.NOT_PARSED,
    "TypeIgnore": _R.NOT_PARSED,
    "type_param": _R.NOT_PARSED,
    "slice": _R.NOT_PARSED,
    "Index": _R.NOT_PARSED,
    "ExtSlice": _R.NOT_PARSED,
    "Num": _R.NOT_PARSED,
    "Str": _R.NOT_PARSED,
    "Bytes": _R.NOT_PARSED,
    "NameConstant": _R.NOT_PARSED,
    "Ellipsis": _R.NOT_PARSED,
}
del _R


def rule_of(node: ast.AST) -> Rule:
    """The rule for ``node``'s class: :attr:`Rule.FALLBACK` for a class the
    table does not name.
    """
    return OPERATIONS.get(type(node).__name__, Rule.FALLBACK)


def node_classes() -> list[type]:
    """Every node class of the running Python's :mod:`ast`."""
    found: list[type] = []
    stack: list[type] = [ast.AST]
    while stack:
        cls = stack.pop()
        if cls in found:
            continue
        found.append(cls)
        stack.extend(cls.__subclasses__())
    # A library's own subclass (a node it decorates) is never parsed.
    return [cls for cls in found if cls.__module__ in ("ast", "_ast")]


# Value kinds.

#: Kinds whose objects hold no callable (a NumPy scalar or array of them
#: too, unless its dtype holds objects); a subclass from the scene's code is
#: read as any object.
FREE_KINDS: tuple[type, ...] = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    type(Ellipsis),
    fractions.Fraction,
    pathlib.PurePath,
    re.Pattern,
)
#: The kinds of an inert value: one holding no callable whose items,
#: operators and methods give inert values again. NumPy arrays and scalars
#: of numbers (no object dtype) are inert too, and so are a compiled
#: pattern and its match (their methods give strings, lists and matches; a
#: replacement function handed to ``sub`` is a handed callable of its own).
INERT_KINDS: tuple[type, ...] = (type(None), bool, int, float, complex, str, bytes)
INERT_OBJECT_KINDS: tuple[type, ...] = (fractions.Fraction, re.Pattern, re.Match)
#: The native kinds whose objects hold only their items (and a
#: ``defaultdict``'s factory): what a snapshot of one reads.
NATIVE_HOLDERS: tuple[type, ...] = (
    list,
    tuple,
    set,
    frozenset,
    dict,
    collections.deque,
    collections.defaultdict,
    collections.OrderedDict,
)
#: The kinds whose items a subscript reads without running code of a class
#: of its own (exact kinds: a subclass may define ``__getitem__`` or
#: ``__missing__``); a NumPy array is one too.
ITEM_KINDS: tuple[type, ...] = (list, tuple, dict, str, bytes, range)
#: Builtins whose result is of an inert kind whatever they are handed
#: (Python holds an argument's ``__float__``, ``__int__``, ``__index__``,
#: ``__bool__``, ``__str__``, ``__repr__``, ``__len__``, ``__hash__`` to
#: give one).
INERT_RESULT_BUILTINS: tuple[Any, ...] = (
    len,
    float,
    int,
    bool,
    complex,
    str,
    repr,
    ascii,
    hash,
    id,
    isinstance,
    issubclass,
    callable,
    ord,
    chr,
    format,
    bin,
    hex,
    oct,
)
#: Attributes reaching a namespace: a function's globals, cells, defaults
#: and code, an object's dict and class, a frame's names. Reading or storing
#: one makes the play tick everything (a binding the plan read may change).
NAMESPACE_ATTRIBUTES = frozenset(
    {
        "__globals__",
        "__closure__",
        "cell_contents",
        "__defaults__",
        "__kwdefaults__",
        "__code__",
        "__dict__",
        "__class__",
        "__bases__",
        "__builtins__",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "gi_frame",
        "cr_frame",
        "ag_frame",
        "tb_frame",
    }
)
#: Builtins reaching a namespace by a name the code does not spell; naming
#: one makes the play tick everything.
NAMESPACE_BUILTINS = frozenset(
    {
        "globals",
        "locals",
        "vars",
        "setattr",
        "delattr",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
    }
)
#: Methods of the native holders that store what they are handed into
#: their receiver (``setattr`` and ``object.__setattr__`` store too).
STORING_METHODS = frozenset(
    {
        "append",
        "appendleft",
        "extend",
        "extendleft",
        "insert",
        "update",
        "setdefault",
        "add",
        "__setitem__",
        "__setattr__",
        "__iadd__",
        "__ior__",
    }
)


def inert_value(value: Any) -> bool:
    """Whether ``value``, resolved, is of an inert kind."""
    if type(value) in INERT_KINDS or type(value) in INERT_OBJECT_KINDS:
        return True
    return (type(value) is np.ndarray or isinstance(value, np.generic)) and not (
        value.dtype.hasobject
    )


def item_kind(value: Any) -> bool:
    """Whether a subscript of ``value`` (resolved) reads an item without
    running code of a class of its own: an exact :data:`ITEM_KINDS` kind,
    any NumPy array.
    """
    return type(value) in ITEM_KINDS or type(value) is np.ndarray


# Purity.

#: Builtins whose result depends only on their arguments, which they
#: never call (beyond an argument's own operators).
_PURE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "complex",
        "divmod",
        "float",
        "int",
        "len",
        "max",
        "min",
        "pow",
        "range",
        "round",
        "str",
        "sum",
        "tuple",
        "zip",
        "enumerate",
        "reversed",
        "sorted",
        "isinstance",
        "list",
        "dict",
        "set",
        "frozenset",
    }
)


def proven_pure(function: Any, _seen: frozenset[int] = frozenset()) -> bool | str:
    """``True`` when ``function`` reads only its arguments and values no
    code can store into, read through the table's rules; else the first
    read that defeats the proof, as a reason.

    Its parameters are its inputs, operated on and indexed as they are;
    reading an attribute of one reads state, and fails the proof. A name it
    reads beyond its own locals must be a module (whose attributes it reads
    are judged the same way), a pure builtin, a function of :mod:`math`, a
    NumPy ufunc or a function of NumPy's outside :mod:`numpy.random` (never
    handed ``out=``), a function of the scene's proven pure itself, or a
    global bound to an inert value (not a NumPy array, which a store may
    change in place, nor a name any ``global`` declaration in its module
    rebinds). Every node it holds must have a rule other than
    :attr:`Rule.SENT`, :attr:`Rule.OUTER_REBIND` and :attr:`Rule.FALLBACK`;
    a store into an attribute or an item, or a call of a storing method,
    fails the proof.
    """
    from . import purity
    from .ticking import _source_node

    function = getattr(function, "__func__", function)
    if not isinstance(function, types.FunctionType):
        return f"{function!r} is not a function whose code can be read"
    if id(function) in _seen:
        return True  # what a call back into itself reads, its other reads say
    seen = _seen | {id(function)}
    where = function.__qualname__
    try:
        node = _source_node(function)
    except purity.Refused as refused:
        return f"{where}: its source can't be read ({refused})"
    code = function.__code__
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    parameters = set(code.co_varnames[:count])
    local = set(code.co_varnames) | set(code.co_cellvars)
    rebound = _globals_rebound(inspect.getmodule(function))
    cells = dict(zip(code.co_freevars, function.__closure__ or (), strict=False))
    nodes = list(ast.walk(node))
    inner = {id(each.value) for each in nodes if isinstance(each, ast.Attribute)}
    for item in nodes:
        rule = rule_of(item)
        if rule in (Rule.SENT, Rule.OUTER_REBIND, Rule.FALLBACK):
            return f"{where}: a {type(item).__name__} ({rule.name})"
        if isinstance(item, (ast.Attribute, ast.Subscript)) and isinstance(
            item.ctx, (ast.Store, ast.Del)
        ):
            return f"{where}: stores into an attribute or an item"
        if isinstance(item, ast.Call):
            if isinstance(item.func, ast.Attribute) and (
                item.func.attr in STORING_METHODS
            ):
                return f"{where}: calls .{item.func.attr}, which stores"
            if any(keyword.arg == "out" for keyword in item.keywords):
                return f"{where}: hands a call out= to write into"
        if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Load):
            root = item.value
            if isinstance(root, ast.Name) and root.id in parameters:
                return f"{where}: reads .{item.attr} of its argument {root.id}"
        if (
            not isinstance(item, (ast.Name, ast.Attribute))
            or not isinstance(item.ctx, ast.Load)
            or id(item) in inner
        ):
            continue
        chain = _outer_chain(item, local)
        if chain is None:
            continue  # rooted in a local or an argument
        name, attributes = chain
        if name in cells:
            try:
                value = cells[name].cell_contents
            except ValueError:
                return f"{where}: reads {name}, an empty cell"
        elif name in function.__globals__:
            if name in rebound or "*" in rebound:
                return f"{where}: reads {name}, a global its module may rebind"
            value = function.__globals__[name]
        elif hasattr(builtins, name):
            if attributes or name not in _PURE_BUILTINS:
                return f"{where}: reads the builtin {name}"
            continue
        else:
            return f"{where}: reads {name}, which is bound to nothing"
        for attribute in attributes:
            if not isinstance(value, types.ModuleType):
                break  # a value's own attribute: judged with the value
            value = getattr(value, attribute, _MISSING)
            name = f"{name}.{attribute}"
        verdict = _pure_value(value, name, seen)
        if verdict is not True:
            return f"{where}: {verdict}"
    return True


_MISSING = object()


def _outer_chain(item: ast.AST, local: set[str]) -> tuple[str, list[str]] | None:
    """The name ``item`` (a name, or attributes of one) is rooted in and
    the attributes it reads in order, when that name is not a local.
    """
    attributes: list[str] = []
    node = item
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id in local:
        return None
    return node.id, attributes[::-1]


def _pure_value(value: Any, name: str, seen: frozenset[int]) -> bool | str:
    """Whether reading ``value`` (bound to ``name`` outside the function)
    keeps the function pure.
    """
    if value is _MISSING:
        return f"reads {name}, which is bound to nothing"
    if isinstance(value, types.ModuleType):
        return True  # what it reads through the module is judged as read
    if isinstance(value, np.ufunc):
        return True
    if isinstance(value, np.ndarray):
        return f"reads {name}, an array a store may change in place"
    if inert_value(value):
        return True
    if type(value) in (tuple, frozenset) and all(
        _pure_value(each, name, seen) is True for each in value
    ):
        return True
    if isinstance(value, types.BuiltinFunctionType):
        owner = getattr(value, "__module__", "") or ""
        if owner in ("math", "cmath") or (
            owner == "builtins" and value.__name__ in _PURE_BUILTINS
        ):
            return True
        return f"reads {name}, a native it can't show pure"
    module = (getattr(value, "__module__", "") or "").split(".")
    if callable(value) and module[0] == "numpy" and "random" not in module:
        return True  # NumPy's own functions of their arguments
    if isinstance(value, types.FunctionType):
        inner = proven_pure(value, seen)
        return True if inner is True else f"calls {name}, not pure ({inner})"
    return f"reads {name}, a {type(value).__name__} whose state may change"


def _globals_rebound(module: types.ModuleType | None) -> set[str]:
    """The names any ``global`` declaration in ``module``'s source names;
    ``{"*"}`` when that source can't be read.
    """
    if module is None:
        return {"*"}
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return {"*"}
    names: set[str] = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.Global):
            names.update(item.names)
    return names


# What becomes of a function's result.

#: Methods that change their receiver in place: the storing ones, and those
#: that remove, reorder or overwrite (a list's, a dict's, a set's, a
#: deque's, a NumPy array's).
MUTATING_METHODS = STORING_METHODS | frozenset(
    {
        "pop",
        "popleft",
        "popitem",
        "remove",
        "discard",
        "clear",
        "sort",
        "reverse",
        "rotate",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
        "__delitem__",
        "__imul__",
        "fill",
        "put",
        "resize",
        "itemset",
        "partition",
        "setfield",
        "byteswap",
        "setflags",
    }
)
#: Methods of a list, a tuple, a dict or a NumPy array that leave it as it
#: is and give a new value (no view of it): what a result may be read by.
_READING_METHODS = frozenset(
    {
        "copy",
        "count",
        "index",
        "get",
        "keys",
        "values",
        "items",
        "tolist",
        "sum",
        "mean",
        "min",
        "max",
        "prod",
        "any",
        "all",
        "argmin",
        "argmax",
        "dot",
        "item",
        "astype",
        "round",
        "clip",
        "cumsum",
        "std",
        "var",
        "nonzero",
        "tobytes",
    }
)


#: Builtins that change nothing they are handed and give a container of
#: its items, or one of them (what they give holds what they were handed).
_ITEM_GIVERS: tuple[Any, ...] = (
    list,
    tuple,
    sorted,
    reversed,
    zip,
    enumerate,
    dict,
    set,
    frozenset,
    max,
    min,
    iter,
    next,
)


def result_kept(function: Any, reachable: Any) -> str | None:
    """``None`` when nothing in ``reachable`` (functions of the scene's whose
    code may run while ``function``'s results live) can change a result of
    ``function`` in place: its results are inert by kind (every return
    inert, whatever it is handed), or no code changes one; else the first
    store or call that may, as a reason.

    A parameter holds what a caller hands it (a result handed by code in
    ``reachable`` is flagged where it is handed; the engine's own arguments
    to a callback, its mobject and numbers, are not taken for results); a
    name bound outside the function to a value that may change (not inert,
    not a module, class or function) may hold one. A result, and every
    name, item or attribute taken from it (an alias),
    may be read only by an operator, a comparison, a reading method
    (``copy``, ``sum``, ``tolist``: a new value), ``len`` and the builtins
    giving inert values, and a NumPy ufunc not handed ``out=``. Anything
    else (a store into it, an operator in place, a changing method
    (:data:`MUTATING_METHODS`) or any method not listed as reading, being
    handed to another call, stored, returned or put in a display) defeats
    the proof; so does a call whose callee can't be told apart from
    ``function`` when its result is used so.
    """
    from . import purity
    from .ticking import _source_node

    function = getattr(function, "__func__", function)
    if _immutable_returns(function):
        return None
    for each in reachable:
        each = getattr(each, "__func__", each)
        if not isinstance(each, types.FunctionType):
            return f"{each!r} is no function whose code can be read"
        try:
            node = _source_node(each)
        except purity.Refused as refused:
            return f"{each.__qualname__}: its source can't be read ({refused})"
        why = _changes_result(each, node, function)
        if why is not None:
            return f"{each.__qualname__}: {why}"
    return None


def _immutable_returns(function: Any) -> bool:
    """Whether every value ``function`` returns, whatever it is handed, is
    of a kind no code changes in place: a constant, what a builtin of
    :data:`INERT_RESULT_BUILTINS` or a function of :mod:`math` gives, a
    comparison, and operators, conditionals and tuples of those.
    """
    from . import purity
    from .ticking import _source_node

    if not isinstance(function, types.FunctionType):
        return False
    try:
        node = _source_node(function)
    except purity.Refused:
        return False
    cells = dict(
        zip(function.__code__.co_freevars, function.__closure__ or (), strict=False)
    )

    def outer(name: str) -> Any:
        if name in cells:
            try:
                return cells[name].cell_contents
            except ValueError:
                return _MISSING
        if name in function.__globals__:
            return function.__globals__[name]
        return getattr(builtins, name, _MISSING)

    def fixed(expr: ast.AST | None) -> bool:
        if expr is None:
            return True  # None
        if isinstance(expr, (ast.Constant, ast.JoinedStr, ast.Compare)):
            return True
        if isinstance(expr, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp, ast.Tuple)):
            parts = [
                each
                for each in ast.iter_child_nodes(expr)
                if isinstance(each, ast.expr)
                and not (isinstance(expr, ast.IfExp) and each is expr.test)
            ]
            return all(fixed(part) for part in parts)
        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Name):
                target = outer(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                holder = outer(func.value.id)
                target = getattr(holder, func.attr, _MISSING)
                if isinstance(holder, types.ModuleType) and holder.__name__ in (
                    "math",
                    "cmath",
                ):
                    return True
            else:
                return False
            return any(target is each for each in INERT_RESULT_BUILTINS)
        return False  # a name, an item, an attribute: what it holds may change

    if isinstance(node, ast.Lambda):
        return fixed(node.body)
    returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
    if any(isinstance(item, (ast.Yield, ast.YieldFrom)) for item in ast.walk(node)):
        return False
    return all(fixed(item.value) for item in returns)


def _calls_of(value: Any, function: Any) -> bool | None:
    """Whether calling ``value`` calls ``function`` (``None``: can't tell)."""
    if value is _MISSING:
        return None
    return value is function or getattr(value, "__wrapped__", None) is function


def _changes_result(
    owner: types.FunctionType, node: ast.AST, function: Any
) -> str | None:
    """Why ``owner``'s code (``node``) may change a result of ``function``
    in place; ``None`` when it never does.
    """
    nodes = list(ast.walk(node))
    cells = dict(zip(owner.__code__.co_freevars, owner.__closure__ or (), strict=False))
    code = owner.__code__
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    parameters = set(code.co_varnames[:count])
    bound: dict[str, list[ast.AST | None]] = {}
    for item in nodes:
        pairs: list[tuple[ast.AST, ast.AST | None]] = []
        if isinstance(item, ast.Assign):
            pairs = [(target, item.value) for target in item.targets]
        elif isinstance(item, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            pairs = [(item.target, item.value)]
        elif isinstance(item, (ast.For, ast.AsyncFor, ast.comprehension)):
            pairs = [(item.target, item.iter)]
        elif isinstance(item, ast.withitem) and item.optional_vars is not None:
            pairs = [(item.optional_vars, None)]
        for target, value in pairs:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    bound.setdefault(name.id, []).append(value)

    def outer(name: str) -> Any:
        if name in cells:
            try:
                return cells[name].cell_contents
            except ValueError:
                return _MISSING
        if name in owner.__globals__:
            return owner.__globals__[name]
        return getattr(builtins, name, _MISSING)

    def callee(func: ast.AST, seen: set[str]) -> bool | None:
        """Whether ``func`` is ``function`` (``None``: can't tell)."""
        if isinstance(func, ast.Name):
            if func.id in bound:
                if func.id in seen:
                    return False
                values = bound[func.id]
                verdicts = [
                    None if value is None else callee(value, seen | {func.id})
                    for value in values
                ]
                if True in verdicts:
                    return True
                return None if None in verdicts else False
            return _calls_of(outer(func.id), function)
        if isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name) and base.id not in bound:
                holder = outer(base.id)
                if isinstance(holder, types.ModuleType):
                    return _calls_of(getattr(holder, func.attr, _MISSING), function)
            return None
        return None if not isinstance(func, ast.Lambda) else False

    def result(expr: ast.AST, seen: set[str]) -> bool | None:
        """Whether ``expr`` may be (or be part of) a result of ``function``:
        ``True``, ``False``, or ``None`` when that can't be told.
        """
        if isinstance(expr, ast.Call):
            is_it = callee(expr.func, set())
            if is_it is not False:
                return is_it
            if (
                isinstance(expr.func, ast.Name)
                and expr.func.id not in bound
                and any(outer(expr.func.id) is each for each in _ITEM_GIVERS)
            ):
                # A container of its argument's items, or one of them.
                verdicts = [
                    result(part, seen)
                    for part in [*expr.args, *(k.value for k in expr.keywords)]
                ]
                if True in verdicts:
                    return True
                return None if None in verdicts else False
            if isinstance(expr.func, ast.Attribute):
                inner = result(expr.func.value, seen)
                if inner is not False and expr.func.attr not in _READING_METHODS:
                    return inner  # a view, or what the method gives back
            return False
        if isinstance(expr, ast.Name):
            if expr.id in parameters or expr.id in seen:
                return False  # a caller's result is flagged where handed
            if expr.id not in bound:
                value = outer(expr.id)
                if (
                    value is _MISSING
                    or inert_value(value)
                    or isinstance(
                        value, (types.ModuleType, type, types.FunctionType, *FREE_KINDS)
                    )
                ):
                    return False
                return None  # state held outside: it may hold a result
            verdicts = [
                False if value is None else result(value, seen | {expr.id})
                for value in bound[expr.id]
            ]
            if True in verdicts:
                return True
            return None if None in verdicts else False
        if isinstance(expr, (ast.Subscript, ast.Attribute, ast.Starred, ast.NamedExpr)):
            return result(expr.value, seen)
        if isinstance(expr, (ast.IfExp, ast.BoolOp)):
            parts = (
                [expr.body, expr.orelse] if isinstance(expr, ast.IfExp) else expr.values
            )
            verdicts = [result(part, seen) for part in parts]
            if True in verdicts:
                return True
            return None if None in verdicts else False
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            verdicts = [
                result(part, seen)
                for part in ast.iter_child_nodes(expr)
                if isinstance(part, ast.expr)
            ]
            if True in verdicts:
                return True
            return None if None in verdicts else False
        return False  # an operator, a comparison, a constant: a new value

    def may_be(expr: ast.AST) -> bool:
        return result(expr, set()) is not False

    parents: dict[int, ast.AST] = {}
    for item in nodes:
        for child in ast.iter_child_nodes(item):
            parents[id(child)] = item
    for item in nodes:
        rule = rule_of(item)
        if rule in (Rule.STORE, Rule.BINDING):
            targets: list[ast.AST] = []
            if isinstance(item, ast.Assign):
                targets = list(item.targets)
            elif isinstance(item, (ast.AnnAssign, ast.AugAssign)):
                targets = [item.target]
            elif isinstance(item, ast.Delete):
                targets = list(item.targets)
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, (ast.Attribute, ast.Subscript)) and may_be(
                        part.value
                    ):
                        return f"stores into {ast.unparse(part)}"
                if isinstance(item, ast.AugAssign) and may_be(item.target):
                    return f"changes {ast.unparse(item.target)} in place"
        if rule is not Rule.CALL:
            continue
        assert isinstance(item, ast.Call)
        func = item.func
        if (
            isinstance(func, ast.Attribute)
            and may_be(func.value)
            and (func.attr in MUTATING_METHODS or func.attr not in _READING_METHODS)
        ):
            return f"calls {ast.unparse(func)}, which may change it"
        handed = [*item.args, *(keyword.value for keyword in item.keywords)]
        if not any(may_be(argument) for argument in handed):
            continue
        callee_value: Any = None
        if isinstance(func, ast.Name) and func.id not in bound:
            callee_value = outer(func.id)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id not in bound
            and isinstance(outer(func.value.id), types.ModuleType)
        ):
            callee_value = getattr(outer(func.value.id), func.attr, _MISSING)
        reads = (
            any(callee_value is each for each in INERT_RESULT_BUILTINS)
            or any(callee_value is each for each in _ITEM_GIVERS)
            or (callee_value is sum)
            or (
                isinstance(callee_value, np.ufunc)
                and not any(keyword.arg == "out" for keyword in item.keywords)
                and len(item.args) <= callee_value.nin  # codespell:ignore
            )
        )
        if not reads:
            return f"hands it to {ast.unparse(func)}, which may change or keep it"
    for item in nodes:
        if not isinstance(item, ast.expr) or not may_be(item):
            continue
        parent = parents.get(id(item))
        if isinstance(parent, ast.Return) or (
            isinstance(parent, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(part, (ast.Attribute, ast.Subscript))
                for target in getattr(
                    parent, "targets", [getattr(parent, "target", None)]
                )
                for part in ast.walk(target)
            )
            and parent.value is item
        ):
            return f"lets {ast.unparse(item)} out of the function"
        if isinstance(parent, (ast.Yield, ast.YieldFrom)):
            return f"lets {ast.unparse(item)} out of the function"
    return None
