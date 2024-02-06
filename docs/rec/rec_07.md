# torchrec.fx

> 原文：[`pytorch.org/torchrec/torchrec.fx.html`](https://pytorch.org/torchrec/torchrec.fx.html)

Torchrec 跟踪器

torchrec 的自定义 FX 跟踪器

请参阅[Torch.FX 文档](https://pytorch.org/docs/stable/fx.html)

torchrec.fx.tracer

```py
class torchrec.fx.tracer.Tracer(leaf_modules: Optional[List[str]] = None)¶
```

基类：`跟踪器`

torchrec 的自定义 FX 跟踪器

请参阅[Torch.FX 文档](https://pytorch.org/docs/stable/fx.html)

我们创建了一个自定义的 FX 跟踪器来跟踪基于 torchrec 的模型。自定义跟踪器处理 Python 通用类型（即 NoWait[T]，Awaitable[T]），并在需要时将其降级为 TorchScript

```py
create_arg(a: Any) → Optional[Union[Tuple[Any, ...], List[Any], Dict[str, Any], slice, range, Node, str, int, float, bool, complex, dtype, Tensor, device, memory_format, layout, OpOverload]]¶
```

一种指定跟踪行为的方法，用于准备作为“图形”中节点参数使用的值

除了默认跟踪器外，还添加了对 NoWait 类型的支持

参数：

**a**（*任何*）- 要作为“图形”中的`参数`发出的值。

返回：

将值`a`转换为适当的`参数`

返回类型：

参数

```py
graph: Graph¶
```

```py
is_leaf_module(m: Module, module_qualified_name: str) → bool¶
```

覆盖 FX 定义以包括量化嵌入袋

```py
module_stack: OrderedDict[str, Tuple[str, Any]]¶
```

```py
node_name_to_scope: Dict[str, Tuple[str, type]]¶
```

```py
path_of_module(mod: Module) → str¶
```

允许跟踪非注册模块。这通常用于使表批量嵌入看起来像 nn.EmbeddingBags

```py
scope: Scope¶
```

```py
trace(root: Union[Module, Callable[[...], Any]], concrete_args: Optional[Dict[str, Any]] = None) → Graph¶
```

注意

此 API 的向后兼容性得到保证。

```py
torchrec.fx.tracer.is_fx_tracing() → bool¶
```

```py
torchrec.fx.tracer.symbolic_trace(root: Union[Module, Callable], concrete_args: Optional[Dict[str, Any]] = None, leaf_modules: Optional[List[str]] = None) → GraphModule¶
```

符号跟踪 API

给定一个`nn.Module`或函数实例`root`，此函数将返回通过跟踪`root`时看到的操作构建的`GraphModule`。

`concrete_args`允许您部分专门化您的函数，无论是删除控制流还是数据结构。

参数：

+   **root**（*Union**[**torch.nn.Module**,* *Callable**]*）- 要跟踪并转换为图形表示的模块或函数。

+   **concrete_args**（*可选**[**Dict**[**str**,* *任何**]**]*）- 要部分专门化的输入

返回：

从`root`中记录的操作创建的模块。

返回类型：

图形模块

Torchrec 跟踪器

torchrec 的自定义 FX 跟踪器

请参阅[Torch.FX 文档](https://pytorch.org/docs/stable/fx.html)
