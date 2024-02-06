# torchrec.distributed

> 原文：[`pytorch.org/torchrec/torchrec.distributed.html`](https://pytorch.org/torchrec/torchrec.distributed.html)

Torchrec Distributed

Torchrec distributed 提供了必要的模块和操作来实现模型并行处理。

这些包括：

+   通过 DistributedModelParallel 进行模型并行处理。

+   用于通信的集合操作，包括 All-to-All 和 Reduce-Scatter。

    +   用于稀疏特征、KJT 和各种嵌入类型的集合操作包装器。

+   包括 ShardedEmbeddingBag 用于 nn.EmbeddingBag 的各种模块的分片实现，ShardedEmbeddingBagCollection 用于 EmbeddingBagCollection

    +   定义任何分片模块实现的分片器。

    +   支持各种计算内核，针对计算设备（CPU/GPU）进行优化，可能包括将嵌入表和/或优化器融合在一起进行批处理。

+   通过 TrainPipelineSparseDist 进行流水线训练，可以重叠数据加载设备传输（复制到 GPU）、设备间通信（input_dist）和计算（前向、后向）以提高性能。

+   支持减少精度训练和推断的量化。

## torchrec.distributed.collective_utils

此文件包含用于构建基于集合的控制流的实用程序。

```py
torchrec.distributed.collective_utils.invoke_on_rank_and_broadcast_result(pg: ProcessGroup, rank: int, func: Callable[[...], T], *args: Any, **kwargs: Any) → T
```

在指定的 rank 上调用函数，并将结果广播给组内的所有成员。

示例：

```py
id = invoke_on_rank_and_broadcast_result(pg, 0, allocate_id) 
```

```py
torchrec.distributed.collective_utils.is_leader(pg: Optional[ProcessGroup], leader_rank: int = 0) → bool
```

检查当前进程是否为领导者。

参数：

+   **pg** (*Optional**[**dist.ProcessGroup**]*) – pg 内的进程排名用于确定进程是否为领导者。pg 为 None 意味着进程是组中唯一的成员（例如，单个进程程序）。

+   **leader_rank** (*int*) – 领导者的定义（默认为 0）。调用者可以使用特定于上下文的定义进行覆盖。

```py
torchrec.distributed.collective_utils.run_on_leader(pg: ProcessGroup, rank: int)
```  ## torchrec.distributed.comm

```py
torchrec.distributed.comm.get_group_rank(world_size: Optional[int] = None, rank: Optional[int] = None) → int
```

获取工作组的组排名。也可通过 GROUP_RANK 环境变量获得，介于 0 和 get_num_groups()之间（参见[`pytorch.org/docs/stable/elastic/run.html`](https://pytorch.org/docs/stable/elastic/run.html)）

```py
torchrec.distributed.comm.get_local_rank(world_size: Optional[int] = None, rank: Optional[int] = None) → int
```

获取本地进程的本地排名（参见[`pytorch.org/docs/stable/elastic/run.html`](https://pytorch.org/docs/stable/elastic/run.html)）通常是工作节点上的工作进程的排名

```py
torchrec.distributed.comm.get_local_size(world_size: Optional[int] = None) → int
```

```py
torchrec.distributed.comm.get_num_groups(world_size: Optional[int] = None) → int
```

获取工作组的数量。通常等同于 max_nnodes（参见[`pytorch.org/docs/stable/elastic/run.html`](https://pytorch.org/docs/stable/elastic/run.html)）

```py
torchrec.distributed.comm.intra_and_cross_node_pg(device: Optional[device] = None, backend: Optional[str] = None) → Tuple[Optional[ProcessGroup], Optional[ProcessGroup]]
```

创建子进程组（节点内和跨节点）  ## torchrec.distributed.comm_ops

```py
class torchrec.distributed.comm_ops.All2AllDenseInfo(output_splits: List[int], batch_size: int, input_shape: List[int], input_splits: List[int])
```

基类：`object`

在调用 alltoall_dense 操作时收集属性的数据类。

```py
batch_size: int
```

```py
input_shape: List[int]
```

```py
input_splits: List[int]
```

```py
output_splits: List[int]
```

```py
class torchrec.distributed.comm_ops.All2AllPooledInfo(batch_size_per_rank: List[int], dim_sum_per_rank: List[int], dim_sum_per_rank_tensor: Optional[Tensor], cumsum_dim_sum_per_rank_tensor: Optional[Tensor], codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`object`

在调用 alltoall_pooled 操作时收集属性的数据类。

```py
batch_size_per_rank
```

每个 rank 中的批处理大小

类型：

List[int]

```py
dim_sum_per_rank
```

每个 rank 中嵌入的特征数量（维度之和）。

类型：

List[int]

```py
dim_sum_per_rank_tensor
```

dim_sum_per_rank 的张量版本，仅由 _recat_pooled_embedding_grad_out 的快速内核使用。

类型：

Optional[Tensor]

```py
cumsum_dim_sum_per_rank_tensor
```

dim_sum_per_rank 的累积和，仅由 _recat_pooled_embedding_grad_out 的快速内核使用。

类型：

Optional[Tensor]

```py
codecs
```

量化通信编解码器。

类型：

Optional[QuantizedCommCodecs]

```py
batch_size_per_rank: List[int]
```

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
cumsum_dim_sum_per_rank_tensor: Optional[Tensor]
```

```py
dim_sum_per_rank: List[int]
```

```py
dim_sum_per_rank_tensor: Optional[Tensor]
```

```py
class torchrec.distributed.comm_ops.All2AllSequenceInfo(embedding_dim: int, lengths_after_sparse_data_all2all: Tensor, forward_recat_tensor: Optional[Tensor], backward_recat_tensor: Tensor, input_splits: List[int], output_splits: List[int], variable_batch_size: bool = False, codecs: Optional[QuantizedCommCodecs] = None, permuted_lengths_after_sparse_data_all2all: Optional[Tensor] = None)
```

基类：`object`

在调用 alltoall_sequence 操作时收集属性的数据类。

```py
embedding_dim
```

嵌入维度。

类型：

int

```py
lengths_after_sparse_data_all2all
```

AlltoAll 后稀疏特征的长度。

类型：

张量

```py
forward_recat_tensor
```

前向传递的 recat 张量。

类型：

Optional[Tensor]

```py
backward_recat_tensor
```

为后向传递的 recat 张量。

类型：

张量

```py
input_splits
```

输入分割。

类型：

List[int]

```py
output_splits
```

输出分割。

类型：

List[int]

```py
variable_batch_size
```

是否启用可变批处理大小。

类型：

布尔值

```py
codecs
```

量化通信编解码器。

类型：

Optional[QuantizedCommCodecs]

```py
permuted_lengths_after_sparse_data_all2all
```

AlltoAll 之前稀疏特征的长度。

类型：

Optional[Tensor]

```py
backward_recat_tensor: Tensor
```

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
embedding_dim: int
```

```py
forward_recat_tensor: Optional[Tensor]
```

```py
input_splits: List[int]
```

```py
lengths_after_sparse_data_all2all: Tensor
```

```py
output_splits: List[int]
```

```py
permuted_lengths_after_sparse_data_all2all: Optional[Tensor] = None
```

```py
variable_batch_size: bool = False
```

```py
class torchrec.distributed.comm_ops.All2AllVInfo(dims_sum_per_rank: ~typing.List[int], B_global: int, B_local: int, B_local_list: ~typing.List[int], D_local_list: ~typing.List[int], input_split_sizes: ~typing.List[int] = <factory>, output_split_sizes: ~typing.List[int] = <factory>, codecs: ~typing.Optional[~torchrec.distributed.types.QuantizedCommCodecs] = None)
```

基类：`object`

调用 alltoallv 操作时收集属性的数据类。

```py
dim_sum_per_rank
```

每个排名中嵌入的特征数量（维度之和）。

类型：

List[int]

```py
B_global
```

每个排名的全局批量大小。

类型：

int

```py
B_local
```

分散之前的本地批量大小。 

类型：

int

```py
B_local_list
```

(List[int])：每个嵌入表在我的当前排名中的本地批量大小。

类型：

List[int]

```py
D_local_list
```

每个嵌入表的嵌入维度（在我的当前排名中）。

类型：

List[int]

```py
input_split_sizes
```

每个排名的输入分割大小，这记住了在执行 all_to_all_single 操作时如何分割输入。

类型：

List[int]

```py
output_split_sizes
```

每个排名的输出分割大小，这记住了在执行 all_to_all_single 操作时如何填充输出。

类型：

List[int]

```py
B_global: int
```

```py
B_local: int
```

```py
B_local_list: List[int]
```

```py
D_local_list: List[int]
```

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
dims_sum_per_rank: List[int]
```

```py
input_split_sizes: List[int]
```

```py
output_split_sizes: List[int]
```

```py
class torchrec.distributed.comm_ops.All2All_Pooled_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused) → Tuple[None, None, None, Tensor]
```

定义使用反向模式自动微分区分操作的公式。

此函数将被所有子类覆盖。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，后面是`forward()`返回的尽可能多的输出（对于前向函数的非张量输出将传入 None），并且应该返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，可以为该输入传递 None 作为梯度。

上下文可用于检索在前向传递期间保存的张量。它还有一个属性`ctx.needs_input_grad`，是一个布尔值元组，表示每个输入是否需要梯度。例如，`backward()`将在第一个输入需要计算相对于输出的梯度时具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], a2ai: All2AllPooledInfo, input_embeddings: Tensor) → Tensor
```

定义自定义自动求导函数的前向。

此函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，后面是任意数量的参数（张量或其他类型）。

+   有关详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管出于向后兼容性目的目前未强制执行）。相反，如果打算在`backward`中使用它们（等效地，`vjp`），则应使用`ctx.save_for_backward()`保存张量，或者如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.All2All_Pooled_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_output: Tensor) → Tuple[None, None, Tensor]
```

定义使用反向模式自动微分区分操作的公式。

此函数将被所有子类覆盖。（定义此函数等效于定义`vjp`函数。）

它必须接受一个上下文 `ctx` 作为第一个参数，后面是与 `forward()` 返回的输出一样多的输出（对于前向函数的非张量输出，将传入 None），并且应该返回与 `forward()` 的输入一样多的张量。每个参数是相对于给定输出的梯度，每个返回值应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可以用来检索在前向传递期间保存的张量。它还有一个属性 `ctx.needs_input_grad`，是一个布尔值元组，表示每个输入是否需要梯度。例如，如果 `forward()` 的第一个输入需要计算相对于输出的梯度，则 `backward()` 将具有 `ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义 autograd 函数的前向。

这个函数应该被所有子类重写。有两种定义前向的方式：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，后面是任意数量的参数（张量或其他类型）。

+   查看 combining-forward-context 以获取更多详细信息

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写 `torch.autograd.Function.setup_context()` 静态方法来处理设置 `ctx` 对象。`output` 是前向的输出，`inputs` 是前向的输入的元组。

+   查看 extending-autograd 以获取更多详细信息

上下文可以用来存储任意数据，然后在反向传播期间检索。张量不应直接存储在 ctx 上（尽管为了向后兼容性，目前没有强制执行）。相反，如果打算在 `backward` 中使用张量，则应使用 `ctx.save_for_backward()` 保存它们（等效地，`vjp`），或者如果打算在 `jvp` 中使用，则应使用 `ctx.save_for_forward()` 保存它们。

```py
class torchrec.distributed.comm_ops.All2All_Seq_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused) → Tuple[None, None, None, Tensor]
```

定义一个用于反向模式自动微分的操作的公式。

这个函数应该被所有子类重写。（定义这个函数等同于定义 `vjp` 函数。）

它必须接受一个上下文 `ctx` 作为第一个参数，后面是与 `forward()` 返回的输出一样多的输出（对于前向函数的非张量输出，将传入 None），并且应该返回与 `forward()` 的输入一样多的张量。每个参数是相对于给定输出的梯度，每个返回值应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可以用来检索在前向传递期间保存的张量。它还有一个属性 `ctx.needs_input_grad`，是一个布尔值元组，表示每个输入是否需要梯度。例如，如果 `forward()` 的第一个输入需要计算相对于输出的梯度，则 `backward()` 将具有 `ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], a2ai: All2AllSequenceInfo, sharded_input_embeddings: Tensor) → Tensor
```

定义自定义自动微分函数的前向。

这个函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，后面是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而没有强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.All2All_Seq_Req_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, sharded_grad_output: Tensor) → Tuple[None, None, Tensor]
```

定义一个公式，用于对反向模式自动微分的操作进行微分。

这个函数将被所有子类覆盖。（定义此函数等效于定义`vjp`函数。）

它必须接受一个上下文`ctx`作为第一个参数，后面是`forward()`返回的输出数量（对于前向函数的非张量输出将传递 None），并且应返回与`forward()`的输入数量相同的张量。每个参数是相对于给定输出的梯度，每个返回值应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为一个布尔值元组，表示每个输入是否需要梯度。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义自动微分函数的前向。

这个函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，后面是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而没有强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.All2Allv_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *grad_output)
```

定义一个公式，用于对反向模式自动微分的操作进行微分。

所有子类都必须重写此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是`forward()`返回的输出数量（对于前向函数的非张量输出将传递 None），并且应返回与`forward()`的输入数量相同的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为布尔值元组，表示每个输入是否需要梯度。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], a2ai: All2AllVInfo, inputs: List[Tensor]) → Tensor
```

定义自定义 autograd 函数的前向。

所有子类都必须重写此函数。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管出于向后兼容性目的目前未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.All2Allv_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *grad_outputs) → Tuple[None, None, Tensor]
```

定义使用反向模式自动微分的操作的微分公式。

所有子类都必须重写此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是`forward()`返回的输出数量（对于前向函数的非张量输出将传递 None），并且应返回与`forward()`的输入数量相同的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有属性 `ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果 `forward()` 的第一个输入需要计算相对于输出的梯度，则 `backward()` 将具有 `ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tuple[Tensor]
```

定义自定义 autograd 函数的前向传播。

此函数将被所有子类覆盖。有两种定义前向的方式：

用法 1（合并前向传播和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向传播和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖 `torch.autograd.Function.setup_context()` 静态方法来处理设置 `ctx` 对象。`output` 是前向传播的输出，`inputs` 是前向传播的输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在 `backward` 中使用，则应使用 `ctx.save_for_backward()` 保存张量（等效地，`vjp`），或者如果打算在 `jvp` 中使用，则应使用 `ctx.save_for_forward()` 保存张量。

```py
class torchrec.distributed.comm_ops.AllGatherBaseInfo(input_size: Size, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`object`

在调用 all_gatther_base_pooled 操作时收集属性的数据类。

```py
input_size
```

输入张量的大小。

类型：

整数

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
input_size: Size
```

```py
class torchrec.distributed.comm_ops.AllGatherBase_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused: Tensor) → Tuple[Optional[Tensor], ...]
```

定义使用反向模式自动微分区分操作的公式。

此函数将被所有子类覆盖。（定义此函数等同于定义 `vjp` 函数。）

它必须接受上下文 `ctx` 作为第一个参数，然后是与 `forward()` 返回的输出一样多（对于前向函数的非张量输出，将传递 None），并且应返回与 `forward()` 的输入一样多的张量。每个参数是相对于给定输出的梯度，每个返回值应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有属性 `ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果 `backward()` 的第一个输入需要计算相对于输出的梯度，则 `forward()` 将具有 `ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], agi: AllGatherBaseInfo, input: Tensor) → Tensor
```

定义自定义 autograd 函数的前向传播。

此函数将被所有子类覆盖。有两种定义前向的方式：

用法 1（合并前向传播和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向传播和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法以处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传播期间可以检索的任意数据。不应直接在 ctx 上存储张量（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用张量，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用张量，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.AllGatherBase_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_outputs: Tensor) → Tuple[None, None, Tensor]
```

为使用后向模式自动微分区分操作定义一个公式。

所有子类都必须覆盖此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受一个上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传递为 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应是相对应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有一个属性`ctx.needs_input_grad`，作为布尔值元组，表示每个输入是否需要梯度。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义 autograd 函数的前向。

所有子类都必须覆盖此函数。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法以处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传播期间可以检索的任意数据。不应直接在 ctx 上存储张量（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用张量，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用张量，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatterBaseInfo(input_sizes: Size, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`object`

在调用 reduce_scatter_base_pooled 操作时收集属性的数据类。

```py
input_sizes
```

输入展平张量的大小。

类型：

torch.Size

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
input_sizes: Size
```

```py
class torchrec.distributed.comm_ops.ReduceScatterBase_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused: Tensor) → Tuple[Optional[Tensor], ...]
```

为使用后向模式自动微分区分操作定义一个公式。

所有子类都必须覆盖此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受一个上下文`ctx`作为第一个参数，后面是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传入 None），并且应该返回与`forward()`中的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，您可以为该输入传递 None 作为梯度。

上下文可以用来检索在前向传递期间保存的张量。它还有一个属性`ctx.needs_input_grad`，是一个布尔值元组，表示每个输入是否需要梯度。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], rsi: ReduceScatterBaseInfo, inputs: Tensor) → Tensor
```

定义自定义自动微分函数的前向。

这个函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，后面是任意数量的参数（张量或其他类型）。

+   查看 combining-forward-context 以获取更多详细信息

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   查看 extending-autograd 以获取更多详细信息

上下文可以用来存储在反向传递期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管为了向后兼容性，目前没有强制执行）。相反，如果打算在`backward`中使用它们，则应使用`ctx.save_for_backward()`保存张量（等效地，`vjp`），或者如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatterBase_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_output: Tensor) → Tuple[None, None, Tensor]
```

为使用反向模式自动微分区分操作定义一个公式。

这个函数将被所有子类覆盖。（定义这个函数等同于定义`vjp`函数。）

它必须接受一个上下文`ctx`作为第一个参数，后面是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传入 None），并且应该返回与`forward()`中的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，您可以为该输入传递 None 作为梯度。

上下文可用于检索在前向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_Tensor: Tensor) → Tensor
```

定义自定义 autograd Function 的前向。

此函数应该被所有子类重写。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传播期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管出于向后兼容性目的目前尚未强制执行）。相反，如果打算在`backward`（等效地，`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，或者如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatterInfo(input_sizes: List[Size], codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`object`

在调用 reduce_scatter_pooled 操作时收集属性的数据类。

```py
input_sizes
```

输入张量的大小。这会在运行反向传播并产生梯度时记住输入张量的大小。

类型：

列表[torch.Size]

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
input_sizes: List[Size]
```

```py
class torchrec.distributed.comm_ops.ReduceScatterVInfo(input_sizes: List[Size], input_splits: List[int], equal_splits: bool, total_input_size: List[int], codecs: Optional[QuantizedCommCodecs])
```

基类：`object`

在调用 reduce_scatter_v_pooled 操作时收集属性的数据类。

```py
input_sizes
```

输入张量的大小。这会在运行反向传播并产生梯度时保存输入张量的大小。

类型：

列表[torch.Size]

```py
input_splits
```

输入张量沿 dim 0 的拆分。

类型：

列表[int]

```py
total_input_size
```

（列表[int]）：总输入大小。

类型：

列表[int]

```py
codecs: Optional[QuantizedCommCodecs]
```

```py
equal_splits: bool
```

```py
input_sizes: List[Size]
```

```py
input_splits: List[int]
```

```py
total_input_size: List[int]
```

```py
class torchrec.distributed.comm_ops.ReduceScatterV_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused: Tensor) → Tuple[Optional[Tensor], ...]
```

定义使用反向模式自动微分区分操作的公式。

此函数应该被所有子类重写。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出数量相同（对于前向函数的非张量输出将传入 None），并且应该返回与`forward()`的输入数量相同的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], rsi: ReduceScatterVInfo, input: Tensor) → Tensor
```

定义自定义 autograd Function 的前向。

此函数应被所有子类重写。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可以用来存储任意数据，然后在反向传播期间检索这些数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而没有强制执行）。相反，如果打算在`backward`（或等效的`vjp`）中使用张量，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用张量，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatterV_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_output: Tensor) → Tuple[None, None, Tensor]
```

为使用反向模式自动微分定义一个操作的求导公式。

此函数应被所有子类重写。（定义此函数等同于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出，将传入 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还有一个属性`ctx.needs_input_grad`，作为布尔值元组，表示每个输入是否需要梯度。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义 autograd 函数的前向。

此函数应被所有子类重写。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可以用来存储任意数据，然后在反向传播期间检索这些数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而没有强制执行）。相反，如果打算在`backward`（或等效的`vjp`）中使用张量，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用张量，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatter_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused: Tensor) → Tuple[Optional[Tensor], ...]
```

为使用反向模式自动微分定义一个操作的求导公式。

此函数应被所有子类重写。（定义此函数等同于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传入 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传播期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], rsi: ReduceScatterInfo, *inputs: Any) → Tensor
```

定义自定义自动微分函数的前向。

所有子类都必须覆盖此函数。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向传播的输出，`inputs`是前向传播的输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传播期间可以检索的任意数据。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在`backward`中使用它们，则应使用`ctx.save_for_backward()`保存张量，或者如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.ReduceScatter_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_output: Tensor) → Tuple[None, None, Tensor]
```

定义用于使用反向模式自动微分区分操作的公式。

所有子类都必须覆盖此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传入 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传播期间保存的张量。它还具有属性`ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义自动微分函数的前向。

此函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向不再接受 ctx 参数。

+   相反，您还必须覆盖`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是前向的输出，`inputs`是前向的输入元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可以用来存储任意数据，然后在反向传播期间检索。张量不应直接存储在 ctx 上（尽管为了向后兼容性，目前没有强制执行）。相反，如果打算在`backward`（等效地，`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.Request(pg: ProcessGroup, device: device)
```

基类：`Awaitable`[`W`]

定义了一个张量在进程组上的集体操作请求。

参数：

**pg**（*dist.ProcessGroup*）- 请求所属的进程组。

```py
class torchrec.distributed.comm_ops.VariableBatchAll2AllPooledInfo(batch_size_per_rank_per_feature: List[List[int]], batch_size_per_feature_pre_a2a: List[int], emb_dim_per_rank_per_feature: List[List[int]], codecs: Optional[QuantizedCommCodecs] = None, input_splits: Optional[List[int]] = None, output_splits: Optional[List[int]] = None)
```

基类：`object`

调用 variable_batch_alltoall_pooled 操作时收集属性的数据类。

```py
batch_size_per_rank_per_feature
```

每个秩每个特征的批量大小。

类型：

List[List[int]]

```py
batch_size_per_feature_pre_a2a
```

散播之前的本地批量大小。

类型：

List[int]

```py
emb_dim_per_rank_per_feature
```

每个秩每个特征的嵌入维度

类型：

List[List[int]]

```py
codecs
```

量化通信编解码器。

类型：

可选[QuantizedCommCodecs]

```py
input_splits
```

张量的所有输入拆分到所有。

类型：

可选[List[int]]

```py
output_splits
```

张量的所有输出拆分到所有。

类型：

可选[List[int]]

```py
batch_size_per_feature_pre_a2a: List[int]
```

```py
batch_size_per_rank_per_feature: List[List[int]]
```

```py
codecs: Optional[QuantizedCommCodecs] = None
```

```py
emb_dim_per_rank_per_feature: List[List[int]]
```

```py
input_splits: Optional[List[int]] = None
```

```py
output_splits: Optional[List[int]] = None
```

```py
class torchrec.distributed.comm_ops.Variable_Batch_All2All_Pooled_Req(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, *unused) → Tuple[None, None, None, Tensor]
```

定义使用反向模式自动微分区分操作的公式。

此函数将被所有子类覆盖。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传入 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可以用来检索在前向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，表示每个输入是否需要梯度的布尔值元组。例如，如果第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], a2ai: VariableBatchAll2AllPooledInfo, input_embeddings: Tensor) → Tensor
```

定义自定义自动微分函数的前向。

此函数将被所有子类覆盖。有两种定义前向的方法：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向传递不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法以处理设置`ctx`对象。`output`是前向传递的输出，`inputs`是前向传递的输入的元组。

+   有关更多详细信息，请参见扩展自动微分

上下文可用于存储任意数据，然后在反向传递期间检索。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.comm_ops.Variable_Batch_All2All_Pooled_Wait(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx, grad_output: Tensor) → Tuple[None, None, Tensor]
```

定义使用反向模式自动微分区分操作的公式。

所有子类都必须重写此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受一个上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于前向函数的非张量输出将传递 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在前向传递期间保存的张量。它还具有一个属性`ctx.needs_input_grad`，作为表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx, pg: ProcessGroup, myreq: Request[Tensor], *dummy_tensor: Tensor) → Tensor
```

定义自定义自动微分函数的前向传递。

所有子类都必须重写此函数。定义前向有两种方式：

用法 1（合并前向和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受一个上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见组合前向上下文

用法 2（分离前向和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   前向传递不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法以处理设置`ctx`对象。`output`是前向传递的输出，`inputs`是前向传递的输入的元组。

+   有关更多详细信息，请参见扩展自动微分

上下文可用于存储任意数据，然后在反向传递期间检索。张量不应直接存储在 ctx 上（尽管目前为了向后兼容性而未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用它们，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用它们，则应使用`ctx.save_for_forward()`保存张量。

```py
torchrec.distributed.comm_ops.all_gather_base_pooled(input: Tensor, group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

从组中的所有进程中聚集张量以形成扁平化的汇总嵌入张量。输入张量的大小为 output_tensor_size / world_size。

参数：

+   **input**（*张量*）-要收集的张量。

+   **group**（*可选**[**dist.ProcessGroup**]）-要处理的进程组。如果为 None，则将使用默认进程组。

返回：

异步工作句柄（可等待），稍后可以等待()以获取生成的张量。

返回类型：

Awaitable[Tensor]

警告

all_gather_base_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.alltoall_pooled(a2a_pooled_embs_tensor: Tensor, batch_size_per_rank: List[int], dim_sum_per_rank: List[int], dim_sum_per_rank_tensor: Optional[Tensor] = None, cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None, group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

对单个池化嵌入张量执行 AlltoAll 操作。每个进程根据世界大小拆分输入池化嵌入张量，然后将拆分列表分发给组中的所有进程。然后将来自组中所有进程的接收张量连接起来并返回单个输出张量。

参数：

+   **a2a_pooled_embs_tensor**（*Tensor*） - 输入池化嵌入。在传递到此函数之前必须将其汇总在一起。其形状为 B x D_local_sum，其中 D_local_sum 是所有本地嵌入表的维度总和。

+   **batch_size_per_rank**（*List**[**int**]*） - 每个 rank 中的批量大小。

+   **dim_sum_per_rank**（*List**[**int**]*） - 每个 rank 中嵌入的特征数（维度之和）。

+   **dim_sum_per_rank_tensor**（*Optional**[**Tensor**]*） - dim_sum_per_rank 的张量版本，仅由 _recat_pooled_embedding_grad_out 的快速内核使用。

+   **cumsum_dim_sum_per_rank_tensor**（*Optional**[**Tensor**]*） - dim_sum_per_rank 的累积和，仅由 _recat_pooled_embedding_grad_out 的快速内核使用。

+   **group**（*Optional**[**dist.ProcessGroup**]*） - 要处理的进程组。如果为 None，则将使用默认进程组。

+   **codecs**（*Optional***[*QuantizedCommCodecs**]*） - 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待()以获取结果张量。

返回类型：

Awaitable[Tensor]

警告

alltoall_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.alltoall_sequence(a2a_sequence_embs_tensor: Tensor, forward_recat_tensor: Tensor, backward_recat_tensor: Tensor, lengths_after_sparse_data_all2all: Tensor, input_splits: List[int], output_splits: List[int], variable_batch_size: bool = False, group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

对序列嵌入执行 AlltoAll 操作。每个进程根据世界大小拆分输入张量，然后将拆分列表分发给组中的所有进程。然后将来自组中所有进程的接收张量连接起来并返回单个输出张量。

注意

用于序列嵌入张量的 AlltoAll 运算符。不支持混合维度。

参数：

+   **a2a_sequence_embs_tensor**（*Tensor*） - 输入嵌入。

+   **forward_recat_tensor**（*Tensor*） - 用于前向的 recat 张量。

+   **backward_recat_tensor**（*Tensor*） - 用于反向的 recat 张量。

+   **lengths_after_sparse_data_all2all**（*Tensor*） - AlltoAll 后稀疏特征的长度。

+   **input_splits**（*List**[**int**]*） - 输入拆分。

+   **output_splits**（*List**[**int**]*） - 输出拆分。

+   **variable_batch_size**（*bool*） - 是否启用可变批量大小。

+   **group**（*Optional**[**dist.ProcessGroup**]*） - 要处理的进程组。如果为 None，则将使用默认进程组。

+   **codecs**（*Optional***[*QuantizedCommCodecs**]*） - 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待()以获取结果张量。

返回类型：

Awaitable[List[Tensor]]

警告

alltoall_sequence 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.alltoallv(inputs: List[Tensor], out_split: Optional[List[int]] = None, per_rank_split_lengths: Optional[List[int]] = None, group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[List[Tensor]]
```

对一组输入嵌入执行 alltoallv 操作。每个进程将列表分发给组中的所有进程。

参数：

+   **inputs**（*List**[**Tensor**]*） - 要分发的张量列表，每个 rank 一个。列表中的张量通常具有不同的长度。

+   **out_split**（*Optional**[**List**[**int**]**]*） - 输出拆分大小（或 dim_sum_per_rank），如果未指定，我们将使用 per_rank_split_lengths 来构建一个输出拆分，假设所有嵌入具有相同的维度。

+   **per_rank_split_lengths**（*Optional**[**List**[**int**]**]*） - 每个 rank 的拆分长度。如果未指定，则必须指定 out_split。

+   **组**（*可选**[**dist.ProcessGroup**]*）- 要操作的进程组。如果为 None，则将使用默认进程组。

+   **编解码器**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待(wait())以获取结果张量列表。

返回类型：

Awaitable[List[Tensor]]

警告

alltoallv 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.get_gradient_division() → bool
```

```py
torchrec.distributed.comm_ops.reduce_scatter_base_pooled(input: Tensor, group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

将一个扁平的池化嵌入张量减少然后分散到组中的所有进程。输入张量的大小为 output_tensor_size * world_size。

参数：

+   **输入**（*张量*）- 要分散的扁平张量。

+   **组**（*可选**[**dist.ProcessGroup**]*）- 要操作的进程组。如果为 None，则将使用默认进程组。

+   **编解码器**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待(wait())以获取结果张量。

返回类型：

Awaitable[Tensor]

警告

reduce_scatter_base_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.reduce_scatter_pooled(inputs: List[Tensor], group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

对一个分成世界大小数量的块的池化嵌入张量执行 reduce-scatter 操作。减少操作的结果被分散到组中的所有进程。

参数：

+   **输入**（*List**[**Tensor**]*）- 要分散的张量列表，每个排名一个。

+   **组**（*可选**[**dist.ProcessGroup**]*）- 要操作的进程组。如果为 None，则将使用默认进程组。

+   **编解码器**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待(wait())以获取结果张量。

返回类型：

Awaitable[Tensor]

警告

reduce_scatter_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.reduce_scatter_v_per_feature_pooled(input: Tensor, batch_size_per_rank_per_feature: List[List[int]], embedding_dims: List[int], group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

对一个 1 维池化嵌入张量执行 reduce-scatter-v 操作，每个特征的批处理大小不同，分成世界大小数量的块。减少操作的结果根据输入拆分分散到组中的所有进程。

参数：

+   **输入**（*张量*）- 要分散的张量，每个排名一个。

+   **batch_size_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 用于确定输入拆分的每个特征的每个排名的批处理大小。

+   **嵌入维度**（*List**[**int**]*）- 用于确定输入拆分的每个特征的嵌入维度。

+   **组**（*可选**[**dist.ProcessGroup**]*）- 要操作的进程组。如果为 None，则将使用默认进程组。

+   **编解码器**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

返回：

异步工作句柄（Awaitable），稍后可以等待(wait())以获取结果张量。

返回类型：

Awaitable[Tensor]

警告

reduce_scatter_v_per_feature_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.reduce_scatter_v_pooled(input: Tensor, input_splits: List[int], group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```

对一个分成世界大小数量的块的池化嵌入张量执行 reduce-scatter-v 操作。减少操作的结果根据输入拆分分散到组中的所有进程。

参数：

+   **输入**（*张量*）- 要分散的张量。

+   **input_splits**（*List**[**int**]*）- 输入拆分。

+   **组**（*可选**[**dist.ProcessGroup**]*）- 要操作的进程组。如果为 None，则将使用默认进程组。

返回：

异步工作句柄（Awaitable），稍后可以等待(wait())以获取结果张量。

返回类型：

Awaitable[Tensor]

警告

reduce_scatter_v_pooled 是实验性的，可能会发生变化。

```py
torchrec.distributed.comm_ops.set_gradient_division(val: bool) → None
```

```py
torchrec.distributed.comm_ops.variable_batch_alltoall_pooled(a2a_pooled_embs_tensor: Tensor, batch_size_per_rank_per_feature: List[List[int]], batch_size_per_feature_pre_a2a: List[int], emb_dim_per_rank_per_feature: List[List[int]], group: Optional[ProcessGroup] = None, codecs: Optional[QuantizedCommCodecs] = None) → Awaitable[Tensor]
```  ## torchrec.distributed.dist_data

```py
class torchrec.distributed.dist_data.EmbeddingsAllToOne(device: device, world_size: int, cat_dim: int)
```

基类：`Module`

将每个设备上的汇总/序列嵌入张量合并为单个张量。

参数：

+   **device**（*torch.device*）- 将分配缓冲区的设备。

+   **world_size**（*int*）- 拓扑中的设备数量。

+   **cat_dim**（*int*）- 您希望在哪个维度上进行连接。对于汇总嵌入，它是 1；对于序列嵌入，它是 0。

```py
forward(tensors: List[Tensor]) → Tensor
```

对汇总/序列嵌入张量执行 AlltoOne 操作。

参数：

**tensors**（*List**[**torch.Tensor**]*）- 嵌入张量列表。

返回：

合并嵌入的等待。

返回类型：

Awaitable[torch.Tensor]

```py
set_device(device_str: str) → None
```

```py
training: bool
```

```py
class torchrec.distributed.dist_data.EmbeddingsAllToOneReduce(device: device, world_size: int)
```

基类：`Module`

将每个设备上的汇总嵌入张量合并为单个张量。

参数：

+   **device**（*torch.device*）- 将分配缓冲区的设备。

+   **world_size**（*int*）- 拓扑中的设备数量。

```py
forward(tensors: List[Tensor]) → Tensor
```

对汇总嵌入张量执行 Reduce 的 AlltoOne 操作。

参数：

**tensors**（*List**[**torch.Tensor**]*）- 嵌入张量列表。

返回：

减少嵌入的等待。

返回类型：

Awaitable[torch.Tensor]

```py
set_device(device_str: str) → None
```

```py
training: bool
```

```py
class torchrec.distributed.dist_data.KJTAllToAll(pg: ProcessGroup, splits: List[int], stagger: int = 1)
```

基类：`Module`

将 KeyedJaggedTensor 根据拆分重新分配到 ProcessGroup。

实现利用 torch.distributed 的 AlltoAll 集合。

输入提供了分发所需的张量和输入拆分。 KJTAllToAllSplitsAwaitable 中的第一个集合调用将传输输出拆分（以为张量分配正确的空间）和每个等级的批量大小。 KJTAllToAllTensorsAwaitable 中的后续集合调用将异步传输实际张量。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **splits**（*List**[**int**]*）- 长度为 pg.size()的列表，指示要发送到每个 pg.rank()的特征数量。假定 KeyedJaggedTensor 按目标等级排序。对所有等级都是相同的。

+   **stagger**（*int*）- 用于应用于 recat 张量的间隔值，请参见 _get_recat 函数以获取更多详细信息。

示例：

```py
keys=['A','B','C']
splits=[2,1]
kjtA2A = KJTAllToAll(pg, splits)
awaitable = kjtA2A(rank0_input)

# where:
# rank0_input is KeyedJaggedTensor holding

#         0           1           2
# 'A'    [A.V0]       None        [A.V1, A.V2]
# 'B'    None         [B.V0]      [B.V1]
# 'C'    [C.V0]       [C.V1]      None

# rank1_input is KeyedJaggedTensor holding

#         0           1           2
# 'A'     [A.V3]      [A.V4]      None
# 'B'     None        [B.V2]      [B.V3, B.V4]
# 'C'     [C.V2]      [C.V3]      None

rank0_output = awaitable.wait()

# where:
# rank0_output is KeyedJaggedTensor holding

#         0           1           2           3           4           5
# 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
# 'B'     None        [B.V0]    [B.V1]        None        [B.V2]      [B.V3, B.V4]

# rank1_output is KeyedJaggedTensor holding
#         0           1           2           3           4           5
# 'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None 
```

```py
forward(input: KeyedJaggedTensor) → Awaitable[KJTAllToAllTensorsAwaitable]
```

将输入发送到相关的 ProcessGroup 等级。

第一个等待将获取所提供张量的输出拆分并发出张量 AlltoAll。第二个等待将获取张量。

参数：

**input**（*KeyedJaggedTensor*）- 要分发的值的 KeyedJaggedTensor。

返回：

KJTAllToAllTensorsAwaitable 的等待。

返回类型：

Awaitable[KJTAllToAllTensorsAwaitable]

```py
training: bool
```

```py
class torchrec.distributed.dist_data.KJTAllToAllSplitsAwaitable(pg: ProcessGroup, input: KeyedJaggedTensor, splits: List[int], labels: List[str], tensor_splits: List[List[int]], input_tensors: List[Tensor], keys: List[str], device: device, stagger: int)
```

基类：`Awaitable`[`KJTAllToAllTensorsAwaitable`]

KJT 张量拆分 AlltoAll 的等待。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **input**（*KeyedJaggedTensor*）- 输入 KJT。

+   **splits**（*List**[**int**]*）- 长度为 pg.size()的列表，指示要发送到每个 pg.rank()的特征数量。假定 KeyedJaggedTensor 按目标等级排序。对所有等级都是相同的。

+   **tensor_splits** (*Dict**[**str**,* *List**[**int**]**]*) – 输入 KJT 提供的张量拆分。

+   **input_tensors** (*List**[**torch.Tensor**]*) – 根据 splits 重新分配的提供的 KJT 张量（即长度，值）。

+   **keys** (*List**[**str**]*) – AlltoAll 后的 KJT 键。

+   **device** (*torch.device*) – 将分配缓冲区的设备。

+   **stagger** (*int*) – 应用于 recat 张量的 stagger 值。

```py
class torchrec.distributed.dist_data.KJTAllToAllTensorsAwaitable(pg: ProcessGroup, input: KeyedJaggedTensor, splits: List[int], input_splits: List[List[int]], output_splits: List[List[int]], input_tensors: List[Tensor], labels: List[str], keys: List[str], device: device, stagger: int, stride_per_rank: Optional[List[int]])
```

基类：`Awaitable`[`KeyedJaggedTensor`]

KJT 张量 AlltoAll 的 Awaitable。

参数：

+   **pg** (*dist.ProcessGroup*) – 用于 AlltoAll 通信的 ProcessGroup。

+   **input** (*KeyedJaggedTensor*) – 输入 KJT。

+   **splits** (*List**[**int**]*) – 长度为 pg.size()的列表，指示要发送到每个 pg.rank()的特征数量。假定 KeyedJaggedTensor 按目标排名排序。对所有排名都是相同的。

+   **input_splits** (*List**[**List**[**int**]**]*) – 每个张量在 AlltoAll 中将获得的值的数量）。

+   **output_splits** (*List**[**List**[**int**]**]*) – 每个张量在 AlltoAll 中输出的每个排名的值的数量。

+   **input_tensors** (*List**[**torch.Tensor**]*) – 根据 splits 重新分配的提供的 KJT 张量（即长度，值）。

+   **labels** (*List**[**str**]*) – 每个提供的张量的标签。

+   **keys** (*List**[**str**]*) – AlltoAll 后的 KJT 键。

+   **device** (*torch.device*) – 将分配缓冲区的设备。

+   **stagger** (*int*) – 应用于 recat 张量的 stagger 值。

+   **stride_per_rank** (*Optional**[**List**[**int**]**]*) – 在非可变批次每个特征情况下的每个排名的步幅。

```py
class torchrec.distributed.dist_data.KJTOneToAll(splits: List[int], world_size: int, device: Optional[device] = None)
```

基类：`Module`

将 KeyedJaggedTensor 重新分配到所有设备。

实现利用 OnetoAll 函数，基本上是将特征 P2P 复制到设备上。

参数：

+   **splits** (*List**[**int**]*) – 将 KeyJaggedTensor 特征拆分为复制之前的长度。

+   **world_size** (*int*) – 拓扑中的设备数量。

+   **device** (*torch.device*) – 将分配 KJT 的设备。

```py
forward(kjt: KeyedJaggedTensor) → KJTList
```

首先拆分特征，然后将切片发送到相应的设备。

参数：

**kjt** (*KeyedJaggedTensor*) – 输入特征。

返回：

KeyedJaggedTensor 拆分的 Awaitable。

返回类型：

AwaitableList[[KeyedJaggedTensor]]

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAllGather(pg: ProcessGroup, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

包装用于汇总嵌入通信的全聚合通信原语的模块类。

提供一个具有布局[batch_size，dimension]的本地输入张量，我们希望从所有排名中收集输入张量到一个扁平化的输出张量中。

该类返回汇总嵌入张量的异步 Awaitable 句柄。全聚合仅适用于 NCCL 后端。

参数：

+   **pg** (*dist.ProcessGroup*) – 发生全聚合通信的进程组。

+   **codecs** (*Optional***[*QuantizedCommCodecs**]*) – 量化通信编解码器。

示例：

```py
init_distributed(rank=rank, size=2, backend="nccl")
pg = dist.new_group(backend="nccl")
input = torch.randn(2, 2)
m = PooledEmbeddingsAllGather(pg)
output = m(input)
tensor = output.wait() 
```

```py
forward(local_emb: Tensor) → PooledEmbeddingsAwaitable
```

对汇总嵌入张量执行 reduce scatter 操作。

参数：

**local_emb** (*torch.Tensor*) – 形状为[num_buckets x batch_size, dimension]的张量。

返回：

汇总张量的 Awaitable，形状为[batch_size, dimension]。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAllToAll(pg: ProcessGroup, dim_sum_per_rank: List[int], device: Optional[device] = None, callbacks: Optional[List[Callable[[Tensor], Tensor]]] = None, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

根据 dim_sum_per_rank 将批次分片并收集张量的键与 ProcessGroup。

实现利用 alltoall_pooled 操作。

参数：

+   **pg** (*dist.ProcessGroup*) – 用于 AlltoAll 通信的 ProcessGroup。

+   **dim_sum_per_rank** (*List**[**int**]*) – 每个秩中嵌入的特征数量（维度之和）。

+   **device** (*Optional**[**torch.device**]*) – 将分配缓冲区的设备。

+   **callbacks** (*Optional**[**List**[**Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]**]*) – 回调函数。

+   **codecs** (*Optional***[*QuantizedCommCodecs**]*) – 量化通信编解码器。

示例：

```py
dim_sum_per_rank = [2, 1]
a2a = PooledEmbeddingsAllToAll(pg, dim_sum_per_rank, device)

t0 = torch.rand((6, 2))
t1 = torch.rand((6, 1))
rank0_output = a2a(t0).wait()
rank1_output = a2a(t1).wait()
print(rank0_output.size())
    # torch.Size([3, 3])
print(rank1_output.size())
    # torch.Size([3, 3]) 
```

```py
property callbacks: List[Callable[[Tensor], Tensor]]
```

```py
forward(local_embs: Tensor, batch_size_per_rank: Optional[List[int]] = None) → PooledEmbeddingsAwaitable
```

对池化嵌入张量执行 AlltoAll 池化操作。

参数：

+   **local_embs** (*torch.Tensor*) – 要分发的值的张量。

+   **batch_size_per_rank** (*Optional**[**List**[**int**]**]*) – 每个秩的批次大小，以支持可变批次大小。

返回：

池化嵌入的 awaitable。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAwaitable(tensor_awaitable: Awaitable[Tensor])
```

基类：`Awaitable`[`Tensor`]

在集体操作后的池化嵌入的 awaitable。

参数：

**tensor_awaitable** (*Awaitable**[**torch.Tensor**]*) – 集体后来自组内所有进程的张量的连接张量的 awaitable。

```py
property callbacks: List[Callable[[Tensor], Tensor]]
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsReduceScatter(pg: ProcessGroup, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

包装池化嵌入通信的 reduce-scatter 通信原语的模块类，以行和 twrw 分片的方式。

对于池化嵌入，我们有一个本地模型并行输出张量，布局为[num_buckets x batch_size, dimension]。我们需要在批次之间的 num_buckets 维度上求和。我们根据 input_splits 沿第一维度将张量分成不均匀的块（不同桶的张量切片），将它们减少到输出张量并将结果分散到相应的秩。

该类返回池化嵌入张量的异步 Awaitable 句柄。reduce-scatter-v 操作仅适用于 NCCL 后端。

参数：

+   **pg** (*dist.ProcessGroup*) – reduce-scatter 通信发生的进程组。

+   **codecs** – 量化通信编解码器。

```py
forward(local_embs: Tensor, input_splits: Optional[List[int]] = None) → PooledEmbeddingsAwaitable
```

对池化嵌入张量执行 reduce scatter 操作。

参数：

+   **local_embs** (*torch.Tensor*) – 形状为[num_buckets * batch_size, dimension]的张量。

+   **input_splits** (*Optional**[**List**[**int**]**]*) – 本地嵌入维度 0 的拆分列表。

返回：

tensor 的池化嵌入的 awaitable，形状为[batch_size, dimension]。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.SeqEmbeddingsAllToOne(device: device, world_size: int)
```

基类：`Module`

将每个设备上的池化/序列嵌入张量合并为单个张量。

参数：

+   **device** (*torch.device*) – 将分配缓冲区的设备

+   **world_size** (*int*) – 拓扑中的设备数量。

+   **cat_dim** (*int*) – 您希望在其上连接的维度。对于池化嵌入，它是 1；对于序列嵌入，它是 0。

```py
forward(tensors: List[Tensor]) → List[Tensor]
```

对池化嵌入张量执行 AlltoOne 操作。

参数：

**tensors** (*List**[**torch.Tensor**]*) – 池化嵌入张量的列表。

返回：

合并池化嵌入的 awaitable。

返回类型：

Awaitable[torch.Tensor]

```py
set_device(device_str: str) → None
```

```py
training: bool
```

```py
class torchrec.distributed.dist_data.SequenceEmbeddingsAllToAll(pg: ProcessGroup, features_per_rank: List[int], device: Optional[device] = None, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

根据拆分将序列嵌入重新分配到 ProcessGroup。

参数：

+   **pg**（*dist.ProcessGroup*）- AlltoAll 通信发生在其中的进程组。

+   **features_per_rank**（*List**[**int**]*）- 每个 rank 的特征数量列表。

+   **device**（*可选**[**torch.device**]*）- 在其中分配缓冲区的设备。

+   **codecs**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

示例：

```py
init_distributed(rank=rank, size=2, backend="nccl")
pg = dist.new_group(backend="nccl")
features_per_rank = [4, 4]
m = SequenceEmbeddingsAllToAll(pg, features_per_rank)
local_embs = torch.rand((6, 2))
sharding_ctx: SequenceShardingContext
output = m(
    local_embs=local_embs,
    lengths=sharding_ctx.lengths_after_input_dist,
    input_splits=sharding_ctx.input_splits,
    output_splits=sharding_ctx.output_splits,
    unbucketize_permute_tensor=None,
)
tensor = output.wait() 
```

```py
forward(local_embs: Tensor, lengths: Tensor, input_splits: List[int], output_splits: List[int], unbucketize_permute_tensor: Optional[Tensor] = None, batch_size_per_rank: Optional[List[int]] = None, sparse_features_recat: Optional[Tensor] = None) → SequenceEmbeddingsAwaitable
```

在序列嵌入张量上执行 AlltoAll 操作。

参数：

+   **local_embs**（*torch.Tensor*）- 输入嵌入张量。

+   **lengths**（*torch.Tensor*）- AlltoAll 后稀疏特征的长度。

+   **input_splits**（*List**[**int**]*）- AlltoAll 的输入拆分。

+   **output_splits**（*List**[**int**]*）- AlltoAll 的输出拆分。

+   **unbucketize_permute_tensor**（*可选**[**torch.Tensor**]*）- 存储 KJT 桶化的排列顺序（仅适用于逐行分片）。

+   **batch_size_per_rank** -（可选[List[int]]）：每个 rank 的批次大小。

+   **sparse_features_recat**（*可选**[**torch.Tensor**]*）- 用于稀疏特征输入分布的 recat 张量。如果使用可变批次大小，则必须提供。

返回：

序列嵌入的可等待对象。

返回类型：

SequenceEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.SequenceEmbeddingsAwaitable(tensor_awaitable: Awaitable[Tensor], unbucketize_permute_tensor: Optional[Tensor], embedding_dim: int)
```

基类：`Awaitable`[`Tensor`]

集体操作后的序列嵌入的可等待对象。

参数：

+   **tensor_awaitable**（*Awaitable**[**torch.Tensor**]*）- 集体操作后来自组内所有进程的连接张量的可等待对象。

+   **unbucketize_permute_tensor**（*可选**[**torch.Tensor**]*）- 存储 KJT 桶化的排列顺序（仅适用于逐行分片）。

+   **embedding_dim**（*int*）- 嵌入维度。

```py
class torchrec.distributed.dist_data.SplitsAllToAllAwaitable(input_tensors: List[Tensor], pg: ProcessGroup)
```

基类：`Awaitable`[`List`[`List`[`int`]]]

拆分 AlltoAll 的可等待对象。

参数：

+   **input_tensors**（*List**[**torch.Tensor**]*）- 用于重新分配的拆分张量。

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

```py
class torchrec.distributed.dist_data.VariableBatchPooledEmbeddingsAllToAll(pg: ProcessGroup, emb_dim_per_rank_per_feature: List[List[int]], device: Optional[device] = None, callbacks: Optional[List[Callable[[Tensor], Tensor]]] = None, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

根据 dim_sum_per_rank 对张量进行分片并收集 ProcessGroup 的键。

实现利用 variable_batch_alltoall_pooled 操作。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **emb_dim_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 每个 rank 每个特征的嵌入维度。

+   **设备**（*可选**[**torch.device**]*）- 在其中分配缓冲区的设备。

+   **callbacks**（*可选**[**List**[**Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]**]*）- 回调函数。

+   **codecs**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

示例：

```py
kjt_split = [1, 2]
emb_dim_per_rank_per_feature = [[2], [3, 3]]
a2a = VariableBatchPooledEmbeddingsAllToAll(
    pg, emb_dim_per_rank_per_feature, device
)

t0 = torch.rand(6) # 2 * (2 + 1)
t1 = torch.rand(24) # 3 * (1 + 3) + 3 * (2 + 2)
#        r0_batch_size   r1_batch_size
#  f_0:              2               1
-----------------------------------------
#  f_1:              1               2
#  f_2:              3               2
r0_batch_size_per_rank_per_feature = [[2], [1]]
r1_batch_size_per_rank_per_feature = [[1, 3], [2, 2]]
r0_batch_size_per_feature_pre_a2a = [2, 1, 3]
r1_batch_size_per_feature_pre_a2a = [1, 2, 2]

rank0_output = a2a(
    t0, r0_batch_size_per_rank_per_feature, r0_batch_size_per_feature_pre_a2a
).wait()
rank1_output = a2a(
    t1, r1_batch_size_per_rank_per_feature, r1_batch_size_per_feature_pre_a2a
).wait()

# input splits:
#   r0: [2*2, 1*2]
#   r1: [1*3 + 3*3, 2*3 + 2*3]

# output splits:
#   r0: [2*2, 1*3 + 3*3]
#   r1: [1*2, 2*3 + 2*3]

print(rank0_output.size())
    # torch.Size([16])
    # 2*2 + 1*3 + 3*3
print(rank1_output.size())
    # torch.Size([14])
    # 1*2 + 2*3 + 2*3 
```

```py
property callbacks: List[Callable[[Tensor], Tensor]]
```

```py
forward(local_embs: Tensor, batch_size_per_rank_per_feature: List[List[int]], batch_size_per_feature_pre_a2a: List[int]) → PooledEmbeddingsAwaitable
```

在池化嵌入张量上执行具有可变特征批次大小的 AlltoAll 池化操作。

参数：

+   **local_embs**（*torch.Tensor*）- 要分发的值张量。

+   **batch_size_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 每个 rank 每个特征的批次大小，a2a 后。用于获取输入拆分。

+   **batch_size_per_feature_pre_a2a**（*List**[**int**]*）- 分散之前的本地批次大小，用于获取输出拆分。按 rank_0 特征，rank_1 特征排序，...

返回：

池化嵌入的可等待对象。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.VariableBatchPooledEmbeddingsReduceScatter(pg: ProcessGroup, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

包装了用于 rw 和 twrw 分片中可变批次池化嵌入通信的 reduce-scatter 通信原语的模块类。

对于每个特征池化嵌入的可变批处理，我们有一个局部模型并行输出张量，其 1d 布局为每个特征每个排名的批处理大小总和乘以相应的嵌入维度 [batch_size_r0_f0 * emb_dim_f0 + …)]. 我们根据 batch_size_per_rank_per_feature 和相应的 embedding_dims 将张量分割成不均匀的块，并将它们减少到输出张量中，然后将结果分散到相应的排名。

该类返回用于池化嵌入张量的异步可等待句柄。reduce-scatter-v 操作仅适用于 NCCL 后端。

参数：

+   **pg** (*dist.ProcessGroup*) – reduce-scatter 通信发生的进程组。

+   **codecs** – 量化通信编解码器。

```py
forward(local_embs: Tensor, batch_size_per_rank_per_feature: List[List[int]], embedding_dims: List[int]) → PooledEmbeddingsAwaitable
```

在池化嵌入张量上执行 reduce scatter 操作。

参数：

+   **local_embs** (*torch.Tensor*) – 形状为 [num_buckets * batch_size, dimension] 的张量。

+   **batch_size_per_rank_per_feature** (*List**[**List**[**int**]**]*) – 用于确定输入拆分的每个特征每个排名的批处理大小。

+   **embedding_dims** (*List**[**int**]*) – 用于确定输入拆分的每个特征的嵌入维度。

返回：

形状为 [batch_size, dimension] 的张量的池化嵌入的可等待。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```  ## torchrec.distributed.embedding

```py
class torchrec.distributed.embedding.EmbeddingCollectionAwaitable(*args, **kwargs)
```

基类：`LazyAwaitable``Dict`[`str`, [`JaggedTensor`]]

```py
class torchrec.distributed.embedding.EmbeddingCollectionContext(sharding_contexts: List[torchrec.distributed.sharding.sequence_sharding.SequenceShardingContext] = <factory>, input_features: List[torchrec.sparse.jagged_tensor.KeyedJaggedTensor] = <factory>, reverse_indices: List[torch.Tensor] = <factory>)
```

基类：`Multistreamable`

```py
input_features: List[KeyedJaggedTensor]
```

```py
record_stream(stream: Stream) → None
```

参见 [`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
reverse_indices: List[Tensor]
```

```py
sharding_contexts: List[SequenceShardingContext]
```

```py
class torchrec.distributed.embedding.EmbeddingCollectionSharder(fused_params: Optional[Dict[str, Any]] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None, use_index_dedup: bool = False)
```

基类：`BaseEmbeddingSharder`[`EmbeddingCollection`]

```py
property module_type: Type[EmbeddingCollection]
```

```py
shard(module: EmbeddingCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedEmbeddingCollection
```

执行实际的分片。它将根据相应的 ParameterSharding 指定的位置在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module** (*M*) – 要分片的模块。

+   **params** (*EmbeddingModuleShardingPlan*) – 完全限定的参数名称字典（模块路径 + 参数名称，用‘.’分隔）到其分片规范。

+   **env** (*ShardingEnv*) – 具有进程组的分片环境。

+   **device** (*torch.device*) – 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
shardable_parameters(module: EmbeddingCollection) → Dict[str, Parameter]
```

可以分片的参数列表。

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。请参阅 ShardingType 以获取众所周知的示例。

```py
class torchrec.distributed.embedding.ShardedEmbeddingCollection(module: EmbeddingCollection, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None, use_index_dedup: bool = False)
```

基类：`ShardedEmbeddingModule`[`KJTList`, `List`[`Tensor`], `Dict``str`, [`JaggedTensor`], `EmbeddingCollectionContext`], `FusedOptimizerModule`

ShardedEmbeddingCollection 的实现。这是公共 API 的一部分，允许手动数据分发流水线。

```py
compute(ctx: EmbeddingCollectionContext, dist_input: KJTList) → List[Tensor]
```

```py
compute_and_output_dist(ctx: EmbeddingCollectionContext, input: KJTList) → LazyAwaitable[Dict[str, JaggedTensor]]
```

在存在多个输出分布的情况下，重写此方法并在相应的计算完成后立即启动输出分布是有意义的。

```py
create_context() → EmbeddingCollectionContext
```

```py
property fused_optimizer: KeyedOptimizer
```

```py
input_dist(ctx: EmbeddingCollectionContext, features: KeyedJaggedTensor) → Awaitable[Awaitable[KJTList]]
```

```py
output_dist(ctx: EmbeddingCollectionContext, output: List[Tensor]) → LazyAwaitable[Dict[str, JaggedTensor]]
```

```py
reset_parameters() → None
```

```py
training: bool
```

```py
torchrec.distributed.embedding.create_embedding_sharding(sharding_type: str, sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None) → EmbeddingSharding[SequenceShardingContext, KeyedJaggedTensor, Tensor, Tensor]
```

```py
torchrec.distributed.embedding.create_sharding_infos_by_sharding(module: EmbeddingCollectionInterface, table_name_to_parameter_sharding: Dict[str, ParameterSharding], fused_params: Optional[Dict[str, Any]]) → Dict[str, List[EmbeddingShardingInfo]]
```

```py
torchrec.distributed.embedding.get_ec_index_dedup() → bool
```

```py
torchrec.distributed.embedding.set_ec_index_dedup(val: bool) → None
```  ## torchrec.distributed.embedding_lookup

```py
class torchrec.distributed.embedding_lookup.CommOpGradientScaling(*args, **kwargs)
```

基类：`Function`

```py
static backward(ctx: FunctionCtx, grad_output: Tensor) → Tuple[Tensor, None]
```

为不同 iating 操作定义一个用于反向模式自动微分的公式。

所有子类都必须重写此函数。（定义此函数等效于定义`vjp`函数。）

它必须接受上下文`ctx`作为第一个参数，然后是与`forward()`返回的输出一样多（对于正向函数的非张量输出将传递 None），并且应返回与`forward()`的输入一样多的张量。每个参数都是相对于给定输出的梯度，每个返回值都应该是相对于相应输入的梯度。如果输入不是张量或是不需要梯度的张量，则可以将 None 作为该输入的梯度传递。

上下文可用于检索在正向传递期间保存的张量。它还具有属性`ctx.needs_input_grad`，表示每个输入是否需要梯度的布尔值元组。例如，如果`forward()`的第一个输入需要计算相对于输出的梯度，则`backward()`将具有`ctx.needs_input_grad[0] = True`。

```py
static forward(ctx: FunctionCtx, input_tensor: Tensor, scale_gradient_factor: int) → Tensor
```

定义自定义 autograd 函数的正向传递。

所有子类都必须重写此函数。有两种定义正向传递的方法：

用法 1（合并正向传递和 ctx）：

```py
@staticmethod
def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    pass 
```

+   它必须接受上下文 ctx 作为第一个参数，然后是任意数量的参数（张量或其他类型）。

+   有关更多详细信息，请参见 combining-forward-context

用法 2（分开正向传递和 ctx）：

```py
@staticmethod
def forward(*args: Any, **kwargs: Any) -> Any:
    pass

@staticmethod
def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
    pass 
```

+   正向不再接受 ctx 参数。

+   相反，您还必须重写`torch.autograd.Function.setup_context()`静态方法来处理设置`ctx`对象。`output`是正向传递的输出，`inputs`是正向传递的输入的元组。

+   有关更多详细信息，请参见 extending-autograd

上下文可用于存储在反向传递期间可以检索的任意数据。不应直接在 ctx 上存储张量（尽管出于向后兼容性目的目前未强制执行）。相反，如果打算在`backward`（等效于`vjp`）中使用张量，则应使用`ctx.save_for_backward()`保存张量，如果打算在`jvp`中使用张量，则应使用`ctx.save_for_forward()`保存张量。

```py
class torchrec.distributed.embedding_lookup.GroupedEmbeddingsLookup(grouped_configs: List[GroupedEmbeddingConfig], pg: Optional[ProcessGroup] = None, device: Optional[device] = None)
```

基类：`BaseEmbeddingLookup`[`KeyedJaggedTensor`, `Tensor`]

查找序列嵌入的模块（即嵌入）

```py
flush() → None
```

```py
forward(sparse_features: KeyedJaggedTensor) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

尽管前向传播的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个函数，因为前者负责运行注册的钩子，而后者则会默默地忽略它们。

```py
load_state_dict(state_dict: OrderedDict[str, Union[torch.Tensor, ShardedTensor]], strict: bool = True) → _IncompatibleKeys
```

从`state_dict`中复制参数和缓冲区到此模块及其后代。

如果`strict`为`True`，则`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict** (*dict*) – 包含参数和持久缓冲区的字典。

+   **strict** (*bool**,* *optional*) – 是否严格执行`state_dict`中的键与此模块的`state_dict()`函数返回的键完全匹配。默认值：`True`

+   **assign** (*bool**,* *optional*) – 是否将状态字典中的项目分配给模块中对应的键，而不是将它们原地复制到模块的当前参数和缓冲区中。当为`False`时，保留当前模块中张量的属性，而为`True`时，保留状态字典中张量的属性。默认值：`False`

返回：

+   **missing_keys** 是一个包含缺失键的字符串列表

+   **unexpected_keys** 是一个包含意外键的字符串列表

返回类型：

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果参数或缓冲区注册为`None`，并且其对应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个模块缓冲区的迭代器，产生缓冲区的名称以及缓冲区本身。

参数：

+   **prefix** (*str*) – 要添加到所有缓冲区名称前面的前缀。

+   **recurse** (*bool**,* *optional*) – 如果为 True，则产生此模块和所有子模块的缓冲区。否则，只产生直接属于此模块的缓冲区。默认为 True。

+   **remove_duplicate** (*bool**,* *optional*) – 是否删除结果中的重复缓冲区。默认为 True。

产生：

*(str, torch.Tensor)* – 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个模块参数的迭代器，产生参数的名称以及参数本身。

参数：

+   **prefix** (*str*) – 要添加到所有参数名称前面的前缀。

+   **recurse** (*bool*) – 如果为 True，则产生此模块和所有子模块的参数。否则，只产生直接属于此模块的参数。

+   **remove_duplicate**（*bool**,* *可选*）- 是否删除结果中的重复参数。默认为 True。

产出：

*(str, Parameter)*- 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
named_parameters_by_table() → Iterator[Tuple[str, TableBatchedEmbeddingSlice]]
```

类似于 named_parameters()，但会产出包含在 TableBatchedEmbeddingSlice 中的 table_name 和 embedding_weights。对于具有多个分片的单个表（即 CW），这些会合并成一个表/权重。用于可组合性。

```py
prefetch(sparse_features: KeyedJaggedTensor, forward_stream: Optional[Stream] = None) → None
```

```py
purge() → None
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含模块整体状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键对应参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**,* *可选*）- 如果提供，模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix**（*str**,* *可选*）- 添加到参数和缓冲区名称以组成 state_dict 中键的前缀。默认值：`''`。

+   **keep_vars**（*bool**,* *可选*）- 默认情况下，状态字典中返回的`Tensor`会从自动求导中分离。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
class torchrec.distributed.embedding_lookup.GroupedPooledEmbeddingsLookup(grouped_configs: List[GroupedEmbeddingConfig], device: Optional[device] = None, pg: Optional[ProcessGroup] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None, scale_weight_gradients: bool = True)
```

基类：`BaseEmbeddingLookup`[`KeyedJaggedTensor`, `Tensor`]

Pooled embeddings 的查找模块（即 EmbeddingBags）

```py
flush() → None
```

```py
forward(sparse_features: KeyedJaggedTensor) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此处调用，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
load_state_dict(state_dict: OrderedDict[str, Union[ShardedTensor, torch.Tensor]], strict: bool = True) → _IncompatibleKeys
```

从`state_dict`中复制参数和缓冲区到此模块及其后代。

如果`strict`为`True`，那么`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict**（*dict*）- 包含参数和持久缓冲区的字典。

+   **strict**（*bool**,* *可选*）- 是否严格执行`state_dict`中的键与此模块的`state_dict()`函数返回的键匹配。默认值：`True`

+   **assign**（*布尔值**，*可选*）-是否将状态字典中的项目分配给模块中对应的键，而不是将它们原地复制到模块的当前参数和缓冲区中。当为`False`时，保留当前模块中张量的属性，而为`True`时，保留状态字典中张量的属性。默认：`False`

返回：

+   **missing_keys**是一个包含缺失键的字符串列表

+   **unexpected_keys**是一个包含意外键的字符串列表

返回类型:

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果参数或缓冲区注册为`None`，并且其对应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个迭代器，遍历模块缓冲区，同时产生缓冲区的名称和缓冲区本身。

参数：

+   **前缀**（*字符串*）-要添加到所有缓冲区名称前面的前缀。

+   **递归**（*布尔值**，*可选*）-如果为 True，则产生此模块及所有子模块的缓冲区。否则，仅产生此模块的直接成员缓冲区。默认为 True。

+   **remove_duplicate**（*布尔值**，*可选*）-是否在结果中删除重复的缓冲区。默认为 True。

产量：

*（字符串，torch.Tensor）* - 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个迭代器，遍历模块参数，同时产生参数的名称和参数本身。

参数：

+   **前缀**（*字符串*）-要添加到所有参数名称前面的前缀。

+   **递归**（*布尔值*）-如果为 True，则产生此模块及所有子模块的参数。否则，仅产生此模块的直接成员参数。

+   **remove_duplicate**（*布尔值**，*可选*）-是否在结果中删除重复的参数。默认为 True。

产量：

*（字符串，参数）* - 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
named_parameters_by_table() → Iterator[Tuple[str, TableBatchedEmbeddingSlice]]
```

类似于 named_parameters()，但产生包含在 TableBatchedEmbeddingSlice 中的 table_name 和 embedding_weights。对于具有多个分片的单个表（即 CW），这些被合并为一个表/权重。用于可组合性。

```py
prefetch(sparse_features: KeyedJaggedTensor, forward_stream: Optional[Stream] = None) → None
```

```py
purge() → None
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含对模块整体状态的引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*字典**，*可选*）-如果提供了，则模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认：`None`。

+   **前缀**（*字符串**，*可选*）-添加到参数和缓冲区名称以组成 state_dict 中键的前缀。默认：`''`。

+   **keep_vars**（*布尔值**，*可选*）-默认情况下，状态字典中返回的`Tensor`会从自动求导中分离。如果设置为`True`，则不会执行分离。默认：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
class torchrec.distributed.embedding_lookup.InferGroupedEmbeddingsLookup(grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]], world_size: int, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None)
```

基类：`InferGroupedLookupMixin`, `BaseEmbeddingLookup`[`KJTList`, `List`[`Tensor`]], `TBEToRegisterMixIn`

```py
get_tbes_to_register() → Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]
```

```py
training: bool
```

```py
class torchrec.distributed.embedding_lookup.InferGroupedLookupMixin
```

基类：`ABC`

```py
forward(sparse_features: KJTList) → List[Tensor]
```

```py
load_state_dict(state_dict: OrderedDict[str, torch.Tensor], strict: bool = True) → _IncompatibleKeys
```

```py
named_buffers(prefix: str = '', recurse: bool = True) → Iterator[Tuple[str, Tensor]]
```

```py
named_parameters(prefix: str = '', recurse: bool = True) → Iterator[Tuple[str, Parameter]]
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

```py
class torchrec.distributed.embedding_lookup.InferGroupedPooledEmbeddingsLookup(grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]], world_size: int, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None)
```

基类：`InferGroupedLookupMixin`, `BaseEmbeddingLookup`[`KJTList`, `List`[`Tensor`]], `TBEToRegisterMixIn`

```py
get_tbes_to_register() → Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]
```

```py
training: bool
```

```py
class torchrec.distributed.embedding_lookup.MetaInferGroupedEmbeddingsLookup(grouped_configs: List[GroupedEmbeddingConfig], device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None)
```

基类：`BaseEmbeddingLookup`[`KeyedJaggedTensor`, `Tensor`], `TBEToRegisterMixIn`

元嵌入查找模块用于推断，因为推断查找引用了所有 GPU 工作器上的多个 TBE 操作。推断分组嵌入查找模块包含在 GPU 工作器上分配的元模块。

```py
flush() → None
```

```py
forward(sparse_features: KeyedJaggedTensor) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
get_tbes_to_register() → Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]
```

```py
load_state_dict(state_dict: OrderedDict[str, Union[ShardedTensor, torch.Tensor]], strict: bool = True) → _IncompatibleKeys
```

将参数和缓冲区从`state_dict`复制到此模块及其后代中。

如果`strict`为`True`，则`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict** (*dict*) – 包含参数和持久缓冲区的字典。

+   **strict** (*bool**,* *optional*) – 是否严格执行`state_dict`中的键与此模块的`state_dict()`函数返回的键匹配。默认值：`True`

+   **assign** (*bool**,* *optional*) – 是否将状态字典中的项目分配给模块中对应的键，而不是将它们原地复制到模块的当前参数和缓冲区中。当`False`时，保留当前模块中张量的属性，而当`True`时，保留状态字典中张量的属性。默认值：`False`

返回：

+   **missing_keys**是一个包含缺少键的 str 列表

+   **unexpected_keys**是一个包含意外键的 str 列表

返回类型：

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果参数或缓冲区注册为`None`，并且其对应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个迭代器，遍历模块缓冲区，同时返回缓冲区的名称和缓冲区本身。

参数：

+   **prefix** (*str*) – 添加到所有缓冲区名称前面的前缀。

+   **recurse** (*bool**,* *optional*) – 如果为 True，则产生此模块和所有子模块的缓冲区。否则，仅产生此模块的直接成员。默认为 True。

+   **remove_duplicate** (*bool**,* *optional*) – 是否删除结果中的重复缓冲区。默认为 True。

产出：

*(str, torch.Tensor)* – 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个迭代器，遍历模块参数，同时返回参数的名称和参数本身。

参数：

+   **prefix** (*str*) – 添加到所有参数名称前面的前缀。

+   **recurse** (*bool*) – 如果为 True，则产生此模块和所有子模块的参数。否则，仅产生此模块的直接成员的参数。

+   **remove_duplicate** (*bool**,* *optional*) – 是否删除结果中的重复参数。默认为 True。

产出：

*(str, Parameter)* – 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
purge() → None
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含模块整体状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前，`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination** (*dict**,* *optional*) – 如果提供，则模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix** (*str**,* *optional*) – 添加到 state_dict 中键的参数和缓冲区名称的前缀。默认值：`''`。

+   **keep_vars** (*bool**,* *optional*) – 默认情况下，状态字典中返回的`Tensor`是从自动求导中分离的。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
class torchrec.distributed.embedding_lookup.MetaInferGroupedPooledEmbeddingsLookup(grouped_configs: List[GroupedEmbeddingConfig], device: Optional[device] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None, fused_params: Optional[Dict[str, Any]] = None)
```

基类：`BaseEmbeddingLookup`[`KeyedJaggedTensor`, `Tensor`], `TBEToRegisterMixIn`

元嵌入袋查找模块用于推理，因为推理查找引用了所有 GPU 工作器上的多个 TBE 操作。推理分组嵌入袋查找模块包含在 GPU 工作器上分配的元模块。

```py
flush() → None
```

```py
forward(sparse_features: KeyedJaggedTensor) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数中定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
get_tbes_to_register() → Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]
```

```py
load_state_dict(state_dict: OrderedDict[str, Union[ShardedTensor, torch.Tensor]], strict: bool = True) → _IncompatibleKeys
```

将参数和缓冲区从`state_dict`复制到此模块及其后代。

如果`strict`为`True`，则`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict** (*dict*) – 包含参数和持久缓冲区的字典。

+   **strict** (*布尔值**,* *可选*) – 是否严格执行`state_dict`中的键必须与此模块的`state_dict()`函数返回的键完全匹配。默认值：`True`

+   **assign** (*布尔值**,* *可选*) – 是否将状态字典中的项目分配给模块中对应的键，而不是将它们原地复制到模块的当前参数和缓冲区中。当为`False`时，保留当前模块张量的属性，而为`True`时，保留状态字典中张量的属性。默认值：`False`

返回：

+   **missing_keys**是一个包含缺失键的 str 列表

+   **unexpected_keys**是一个包含意外键的 str 列表

返回类型：

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果参数或缓冲区注册为`None`，并且其对应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个模块缓冲区的迭代器，产出缓冲区的名称以及缓冲区本身。

参数：

+   **prefix** (*str*) – 要添加到所有缓冲区名称前面的前缀。

+   **recurse** (*布尔值**,* *可选*) – 如果为 True，则产出此模块及所有子模块的缓冲区。否则，仅产出此模块的直接成员。默认为 True。

+   **remove_duplicate** (*布尔值**,* *可选*) – 是否在结果中删除重复的缓冲区。默认为 True。

产出：

*(str, torch.Tensor)* – 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个模块参数的迭代器，产出参数的名称以及参数本身。

参数：

+   **prefix** (*str*) – 要添加到所有参数名称前面的前缀。

+   **recurse** (*布尔值*) – 如果为 True，则产出此模块及所有子模块的参数。否则，仅产出此模块的直接成员。

+   **remove_duplicate** (*布尔值**,* *可选*) – 是否在结果中删除重复的参数。默认为 True。

产出：

*(str, Parameter)* – 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
purge() → None
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含模块整体状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是对应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前，`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数，但这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**，*可选*）- 如果提供，模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix**（*str**，*可选*）- 添加到参数和缓冲区名称以组成 state_dict 中键的前缀。默认值：`''`。

+   **keep_vars**（*bool**，*可选*）- 默认情况下，state dict 中返回的`Tensor`会从自动求导中分离。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
torchrec.distributed.embedding_lookup.embeddings_cat_empty_rank_handle(embeddings: List[Tensor], dummy_embs_tensor: Tensor, dim: int = 0) → Tensor
```

```py
torchrec.distributed.embedding_lookup.fx_wrap_tensor_view2d(x: Tensor, dim0: int, dim1: int) → Tensor
```  ## torchrec.distributed.embedding_sharding

```py
class torchrec.distributed.embedding_sharding.BaseEmbeddingDist(*args, **kwargs)
```

基类：`ABC`，`Module`，`Generic`[`C`，`T`，`W`]

将 EmbeddingLookup 的输出从模型并行转换为数据并行。

```py
abstract forward(local_embs: T, sharding_ctx: Optional[C] = None) → Union[Awaitable[W], W]
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

虽然前向传播的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个函数，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.embedding_sharding.BaseSparseFeaturesDist(*args, **kwargs)
```

基类：`ABC`，`Module`，`Generic`[`F`]

将输入从数据并行转换为模型并行。

```py
abstract forward(sparse_features: KeyedJaggedTensor) → Union[Awaitable[Awaitable[F]], F]
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

虽然前向传播的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个函数，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.embedding_sharding.EmbeddingSharding(qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ABC`，`Generic`[`C`，`F`，`T`，`W`]，`FeatureShardingMixIn`

用于为 EmbeddingBagCollection 实现不同的分片类型，例如 table_wise。

```py
abstract create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[F]
```

```py
abstract create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup[F, T]
```

```py
abstract create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[C, T, W]
```

```py
abstract embedding_dims() → List[int]
```

```py
abstract embedding_names() → List[str]
```

```py
abstract embedding_names_per_rank() → List[List[str]]
```

```py
abstract embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
embedding_tables() → List[ShardedEmbeddingTable]
```

```py
property qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]]
```

```py
uncombined_embedding_dims() → List[int]
```

```py
uncombined_embedding_names() → List[str]
```

```py
class torchrec.distributed.embedding_sharding.EmbeddingShardingContext(batch_size_per_rank: List[int] = <factory>, batch_size_per_rank_per_feature: List[List[int]] = <factory>, batch_size_per_feature_pre_a2a: List[int] = <factory>, variable_batch_per_feature: bool = False)
```

基类：`Multistreamable`

```py
batch_size_per_feature_pre_a2a: List[int]
```

```py
batch_size_per_rank: List[int]
```

```py
batch_size_per_rank_per_feature: List[List[int]]
```

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
variable_batch_per_feature: bool = False
```

```py
class torchrec.distributed.embedding_sharding.EmbeddingShardingInfo(embedding_config: torchrec.modules.embedding_configs.EmbeddingTableConfig, param_sharding: torchrec.distributed.types.ParameterSharding, param: torch.Tensor, fused_params: Union[Dict[str, Any], NoneType] = None)
```

基类：`object`

```py
embedding_config: EmbeddingTableConfig
```

```py
fused_params: Optional[Dict[str, Any]] = None
```

```py
param: Tensor
```

```py
param_sharding: ParameterSharding
```

```py
class torchrec.distributed.embedding_sharding.FusedKJTListSplitsAwaitable(requests: List[KJTListSplitsAwaitable[C]], contexts: List[C], pg: Optional[ProcessGroup])
```

基类：`Awaitable``List`[[`KJTListAwaitable`]]

```py
class torchrec.distributed.embedding_sharding.KJTListAwaitable(awaitables: List[Awaitable[KeyedJaggedTensor]], ctx: C)
```

基类：`Awaitable`[`KJTList`]

可等待的 KJTList。

参数：

+   **awaitables**（*List***[*Awaitable***[*KeyedJaggedTensor**]**]*)- 稀疏特征的可等待列表。

+   **ctx**（*C*）- 用于保存从 KJT 到嵌入 AlltoAll 的批量大小信息的分片上下文。

```py
class torchrec.distributed.embedding_sharding.KJTListSplitsAwaitable(awaitables: List[Awaitable[Awaitable[KeyedJaggedTensor]]], ctx: C)
```

基类：`Awaitable`[`Awaitable`[`KJTList`]]，`Generic`[`C`]

可等待的可等待的 KJTList。

参数：

+   **awaitables** (*List***[*Awaitable***[*Awaitable***[*KeyedJaggedTensor**]**]**]*) – 调用具有稀疏特征的 KJTAllToAll 的前向结果以重新分配。

+   **ctx** (*C*) – 保存从输入分布到嵌入 AlltoAll 的元数据的分片上下文。

```py
class torchrec.distributed.embedding_sharding.KJTSplitsAllToAllMeta(pg: torch.distributed.distributed_c10d.ProcessGroup, _input: torchrec.sparse.jagged_tensor.KeyedJaggedTensor, splits: List[int], splits_tensors: List[torch.Tensor], input_splits: List[List[int]], input_tensors: List[torch.Tensor], labels: List[str], keys: List[str], device: torch.device, stagger: int, splits_cumsum: List[int])
```

基类：`object`

```py
device: device
```

```py
input_splits: List[List[int]]
```

```py
input_tensors: List[Tensor]
```

```py
keys: List[str]
```

```py
labels: List[str]
```

```py
pg: ProcessGroup
```

```py
splits: List[int]
```

```py
splits_cumsum: List[int]
```

```py
splits_tensors: List[Tensor]
```

```py
stagger: int
```

```py
class torchrec.distributed.embedding_sharding.ListOfKJTListAwaitable(awaitables: List[Awaitable[KJTList]])
```

基类：`Awaitable`[`ListOfKJTList`]

此模块处理推断的表格级分片输入特征分布。

参数：

**awaitables** (*List***[*Awaitable***[*KJTList**]**]*) – KJTList 的 Awaitable 列表。

```py
class torchrec.distributed.embedding_sharding.ListOfKJTListSplitsAwaitable(awaitables: List[Awaitable[Awaitable[KJTList]]])
```

基类：`Awaitable`[`Awaitable`[`ListOfKJTList`]]

Awaitable 的 Awaitable 的 ListOfKJTList。

参数：

**awaitables** (*List***[*Awaitable***[*Awaitable***[*KJTList**]**]**]*) – 稀疏特征列表的 Awaitable 的 Awaitable 列表。

```py
torchrec.distributed.embedding_sharding.bucketize_kjt_before_all2all(kjt: KeyedJaggedTensor, num_buckets: int, block_sizes: Tensor, output_permute: bool = False, bucketize_pos: bool = False, block_bucketize_row_pos: Optional[List[Tensor]] = None) → Tuple[KeyedJaggedTensor, Optional[Tensor]]
```

将 KeyedJaggedTensor 中的值分桶为 num_buckets 个桶，长度根据桶化结果重新调整。

注意：此函数应仅用于在调用 KJTAllToAll 之前进行逐行分片。

参数：

+   **num_buckets** (*int*) – 将值分桶的桶数。

+   **block_sizes** – (torch.Tensor): 键控维度的桶大小。

+   **output_permute** (*bool*) – 输出未分桶值到分桶值的内存位置映射或不输出。

+   **bucketize_pos** (*bool*) – 输出桶化值的更改位置或不输出。

+   **block_bucketize_row_pos** (*Optional**[**List**[**torch.Tensor**]**]*) – 每个特征的分片大小的偏移量。

返回：

桶化的 KeyedJaggedTensor 和未桶化值到桶化值的可选置换映射。

返回类型：

Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]

```py
torchrec.distributed.embedding_sharding.group_tables(tables_per_rank: List[List[ShardedEmbeddingTable]]) → List[List[GroupedEmbeddingConfig]]
```

按照 DataType、PoolingType 和 EmbeddingComputeKernel 对表进行分组。

参数：

**tables_per_rank** (*List****List**[*[*ShardedEmbeddingTable**]**]*) – 每个秩的一致加权的分片嵌入表列表。

返回：

每个秩的特征的 GroupedEmbeddingConfig 列表。

返回类型：

ListList[[GroupedEmbeddingConfig]]  ## torchrec.distributed.embedding_types

```py
class torchrec.distributed.embedding_types.BaseEmbeddingLookup(*args, **kwargs)
```

基类：`ABC`，`Module`，`Generic`[`F`，`T`]

由不同的嵌入实现实现的接口：例如，依赖于 nn.EmbeddingBag 或表批处理的接口等。

```py
abstract forward(sparse_features: F) → T
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者会负责运行注册的钩子，而后者会默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.embedding_types.BaseEmbeddingSharder(fused_params: Optional[Dict[str, Any]] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ModuleSharder`[`M`]

```py
compute_kernels(sharding_type: str, compute_device_type: str) → List[str]
```

给定分片类型和计算设备的支持计算内核列表。

```py
property fused_params: Optional[Dict[str, Any]]
```

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。查看 ShardingType 以获取常见示例。

```py
storage_usage(tensor: Tensor, compute_device_type: str, compute_kernel: str) → Dict[str, int]
```

给定计算设备和计算内核，列出系统资源及相应的使用情况

```py
class torchrec.distributed.embedding_types.BaseGroupedFeatureProcessor(*args, **kwargs)
```

基类：`Module`

分组特征处理器的抽象基类

```py
abstract forward(features: KeyedJaggedTensor) → KeyedJaggedTensor
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者会负责运行注册的钩子，而后者会默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.embedding_types.BaseQuantEmbeddingSharder(fused_params: Optional[Dict[str, Any]] = None, shardable_params: Optional[List[str]] = None)
```

基类：`ModuleSharder`[`M`]

```py
compute_kernels(sharding_type: str, compute_device_type: str) → List[str]
```

给定分片类型和计算设备的支持计算内核列表。

```py
property fused_params: Optional[Dict[str, Any]]
```

```py
shardable_parameters(module: M) → Dict[str, Parameter]
```

可以进行分片的参数列表。

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。查看 ShardingType 以获取常见示例。

```py
storage_usage(tensor: Tensor, compute_device_type: str, compute_kernel: str) → Dict[str, int]
```

给定计算设备和计算内核，列出系统资源及相应的使用情况

```py
class torchrec.distributed.embedding_types.EmbeddingAttributes(compute_kernel: torchrec.distributed.embedding_types.EmbeddingComputeKernel = <EmbeddingComputeKernel.DENSE: 'dense'>)
```

基类：`object`

```py
compute_kernel: EmbeddingComputeKernel = 'dense'
```

```py
class torchrec.distributed.embedding_types.EmbeddingComputeKernel(value)
```

基类：`Enum`

一个枚举。

```py
DENSE = 'dense'
```

```py
FUSED = 'fused'
```

```py
FUSED_UVM = 'fused_uvm'
```

```py
FUSED_UVM_CACHING = 'fused_uvm_caching'
```

```py
QUANT = 'quant'
```

```py
QUANT_UVM = 'quant_uvm'
```

```py
QUANT_UVM_CACHING = 'quant_uvm_caching'
```

```py
class torchrec.distributed.embedding_types.FeatureShardingMixIn
```

基类：`object`

特征分片接口，提供分片感知特征元数据。

```py
feature_names() → List[str]
```

```py
feature_names_per_rank() → List[List[str]]
```

```py
features_per_rank() → List[int]
```

```py
class torchrec.distributed.embedding_types.GroupedEmbeddingConfig(data_type: torchrec.types.DataType, pooling: torchrec.modules.embedding_configs.PoolingType, is_weighted: bool, has_feature_processor: bool, compute_kernel: torchrec.distributed.embedding_types.EmbeddingComputeKernel, embedding_tables: List[torchrec.distributed.embedding_types.ShardedEmbeddingTable], fused_params: Union[Dict[str, Any], NoneType] = None)
```

基类：`object`

```py
compute_kernel: EmbeddingComputeKernel
```

```py
data_type: DataType
```

```py
dim_sum() → int
```

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
embedding_tables: List[ShardedEmbeddingTable]
```

```py
feature_hash_sizes() → List[int]
```

```py
feature_names() → List[str]
```

```py
fused_params: Optional[Dict[str, Any]] = None
```

```py
has_feature_processor: bool
```

```py
is_weighted: bool
```

```py
num_features() → int
```

```py
pooling: PoolingType
```

```py
table_names() → List[str]
```

```py
class torchrec.distributed.embedding_types.KJTList(features: List[KeyedJaggedTensor])
```

基类：`Multistreamable`

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
class torchrec.distributed.embedding_types.ListOfKJTList(features: List[KJTList])
```

基类：`Multistreamable`

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
class torchrec.distributed.embedding_types.ModuleShardingMixIn
```

基类：`object`

访问分片模块的分片方案的接口。

```py
property shardings: Dict[str, FeatureShardingMixIn]
```

```py
class torchrec.distributed.embedding_types.OptimType(value)
```

基类：`Enum`

一个枚举。

```py
ADAGRAD = 'ADAGRAD'
```

```py
ADAM = 'ADAM'
```

```py
ADAMW = 'ADAMW'
```

```py
LAMB = 'LAMB'
```

```py
LARS_SGD = 'LARS_SGD'
```

```py
LION = 'LION'
```

```py
PARTIAL_ROWWISE_ADAM = 'PARTIAL_ROWWISE_ADAM'
```

```py
PARTIAL_ROWWISE_LAMB = 'PARTIAL_ROWWISE_LAMB'
```

```py
ROWWISE_ADAGRAD = 'ROWWISE_ADAGRAD'
```

```py
SGD = 'SGD'
```

```py
SHAMPOO = 'SHAMPOO'
```

```py
SHAMPOO_V2 = 'SHAMPOO_V2'
```

```py
class torchrec.distributed.embedding_types.ShardedConfig(local_rows: int = 0, local_cols: int = 0)
```

基类：`object`

```py
local_cols: int = 0
```

```py
local_rows: int = 0
```

```py
class torchrec.distributed.embedding_types.ShardedEmbeddingModule(qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ShardedModule`[`CompIn`, `DistOut`, `Out`, `ShrdCtx`], `ModuleShardingMixIn`

所有模型并行嵌入模块都实现了这个接口。输入和输出是数据并行的。

参数::

qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : CommOp 名称到 QuantizedCommCodecs 的映射

```py
extra_repr() → str
```

漂亮地打印模块的查找模块、输入分布和输出分布的表示

```py
prefetch(dist_input: KJTList, forward_stream: Optional[Stream] = None) → None
```

为每个查找模块预取输入特征。

```py
training: bool
```

```py
class torchrec.distributed.embedding_types.ShardedEmbeddingTable(num_embeddings: int, embedding_dim: int, name: str = '', data_type: torchrec.types.DataType = <DataType.FP32: 'FP32'>, feature_names: List[str] = <factory>, weight_init_max: Union[float, NoneType] = None, weight_init_min: Union[float, NoneType] = None, pruning_indices_remapping: Union[torch.Tensor, NoneType] = None, init_fn: Union[Callable[[torch.Tensor], Union[torch.Tensor, NoneType]], NoneType] = None, need_pos: bool = False, pooling: torchrec.modules.embedding_configs.PoolingType = <PoolingType.SUM: 'SUM'>, is_weighted: bool = False, has_feature_processor: bool = False, embedding_names: List[str] = <factory>, compute_kernel: torchrec.distributed.embedding_types.EmbeddingComputeKernel = <EmbeddingComputeKernel.DENSE: 'dense'>, local_rows: int = 0, local_cols: int = 0, local_metadata: Union[torch.distributed._shard.metadata.ShardMetadata, NoneType] = None, global_metadata: Union[torch.distributed._shard.sharded_tensor.metadata.ShardedTensorMetadata, NoneType] = None, fused_params: Union[Dict[str, Any], NoneType] = None)
```

基类：`ShardedMetaConfig`, `EmbeddingAttributes`, `EmbeddingTableConfig`

```py
fused_params: Optional[Dict[str, Any]] = None
```

```py
class torchrec.distributed.embedding_types.ShardedMetaConfig(local_rows: int = 0, local_cols: int = 0, local_metadata: Union[torch.distributed._shard.metadata.ShardMetadata, NoneType] = None, global_metadata: Union[torch.distributed._shard.sharded_tensor.metadata.ShardedTensorMetadata, NoneType] = None)
```

基类：`ShardedConfig`

```py
global_metadata: Optional[ShardedTensorMetadata] = None
```

```py
local_metadata: Optional[ShardMetadata] = None
```

```py
torchrec.distributed.embedding_types.compute_kernel_to_embedding_location(compute_kernel: EmbeddingComputeKernel) → EmbeddingLocation
```  ## torchrec.distributed.embeddingbag

```py
class torchrec.distributed.embeddingbag.EmbeddingAwaitable(*args, **kwargs)
```

基类：`LazyAwaitable`[`Tensor`]

```py
class torchrec.distributed.embeddingbag.EmbeddingBagCollectionAwaitable(*args, **kwargs)
```

基类：`LazyAwaitable`[`KeyedTensor`]

```py
class torchrec.distributed.embeddingbag.EmbeddingBagCollectionContext(sharding_contexts: List[Union[torchrec.distributed.embedding_sharding.EmbeddingShardingContext, NoneType]] = <factory>, inverse_indices: Union[Tuple[List[str], torch.Tensor], NoneType] = None, variable_batch_per_feature: bool = False)
```

基类：`Multistreamable`

```py
inverse_indices: Optional[Tuple[List[str], Tensor]] = None
```

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
sharding_contexts: List[Optional[EmbeddingShardingContext]]
```

```py
variable_batch_per_feature: bool = False
```

```py
class torchrec.distributed.embeddingbag.EmbeddingBagCollectionSharder(fused_params: Optional[Dict[str, Any]] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingSharder`[`EmbeddingBagCollection`]

此实现使用非融合的 EmbeddingBagCollection

```py
property module_type: Type[EmbeddingBagCollection]
```

```py
shard(module: EmbeddingBagCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedEmbeddingBagCollection
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module**（*M*） - 要分片的模块。

+   **params**（*EmbeddingModuleShardingPlan*） - 完全限定的参数名称（模块路径+参数名称，用'.'分隔）到其分片规范的字典。

+   **env**（*ShardingEnv*） - 具有进程组的分片环境。

+   **device**（*torch.device*） - 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
shardable_parameters(module: EmbeddingBagCollection) → Dict[str, Parameter]
```

可以分片的参数列表。

```py
class torchrec.distributed.embeddingbag.EmbeddingBagSharder(fused_params: Optional[Dict[str, Any]] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingSharder`[`EmbeddingBag`]

此实现使用非融合的 nn.EmbeddingBag

```py
property module_type: Type[EmbeddingBag]
```

```py
shard(module: EmbeddingBag, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedEmbeddingBag
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module**（*M*） - 要分片的模块。

+   **params**（*EmbeddingModuleShardingPlan*） - 完全限定的参数名称（模块路径+参数名称，用'.'分隔）到其分片规范的字典。

+   **env**（*ShardingEnv*） - 具有进程组的分片环境。

+   **device**（*torch.device*） - 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
shardable_parameters(module: EmbeddingBag) → Dict[str, Parameter]
```

可以分片的参数列表。

```py
class torchrec.distributed.embeddingbag.ShardedEmbeddingBag(module: EmbeddingBag, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None)
```

基类：`ShardedEmbeddingModule`[`KeyedJaggedTensor`, `Tensor`, `Tensor`, `NullShardedModuleContext`], `FusedOptimizerModule`

nn.EmbeddingBag 的分片实现。这是公共 API 的一部分，允许手动数据分布流水线。

```py
compute(ctx: NullShardedModuleContext, dist_input: KeyedJaggedTensor) → Tensor
```

```py
create_context() → NullShardedModuleContext
```

```py
property fused_optimizer: KeyedOptimizer
```

```py
input_dist(ctx: NullShardedModuleContext, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None) → Awaitable[Awaitable[KeyedJaggedTensor]]
```

```py
load_state_dict(state_dict: OrderedDict[str, torch.Tensor], strict: bool = True) → _IncompatibleKeys
```

将参数和缓冲区从`state_dict`复制到此模块及其后代。

如果`strict`为`True`，则`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict**（*dict*） - 包含参数和持久缓冲区的字典。

+   **strict**（*bool**，*可选*）- 是否严格执行`state_dict`中的键与此模块的`state_dict()`函数返回的键匹配。默认值：`True`

+   **assign**（*bool**，*可选*）- 是否将状态字典中的项目分配给模块中对应的键，而不是将它们原地复制到模块的当前参数和缓冲区中。当为`False`时，保留当前模块中张量的属性，而为`True`时，保留状态字典中张量的属性。默认值：`False`

返回：

+   **missing_keys**是包含缺少键的 str 列表

+   **unexpected_keys**是包含意外键的 str 列表

返回类型：

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果将参数或缓冲区注册为`None`，并且其相应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个迭代器，遍历模块缓冲区，产生缓冲区的名称以及缓冲区本身。

参数：

+   **prefix**（*str*）- 要添加到所有缓冲区名称前面的前缀。

+   **recurse**（*bool**，*可选*）- 如果为 True，则会产生此模块及所有子模块的缓冲区。否则，只会产生此模块的直接成员缓冲区。默认为 True。

+   **remove_duplicate**（*bool**，*可选*）- 是否删除结果中的重复缓冲区。默认为 True。

产生：

*(str, torch.Tensor)* - 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_modules(memo: Optional[Set[Module]] = None, prefix: str = '', remove_duplicate: bool = True) → Iterator[Tuple[str, Module]]
```

返回一个迭代器，遍历网络中的所有模块，产生模块的名称以及模块本身。

参数：

+   **memo** - 一个备忘录，用于存储已添加到结果中的模块集合

+   **prefix** - 将添加到模块名称前面的前缀

+   **remove_duplicate** - 是否删除结果中的重复模块实例

产生：

*(str, Module)* - 名称和模块的元组

注意

重复模块只返回一次。在以下示例中，`l`只会返回一次。

示例：

```py
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
...     print(idx, '->', m)

0 -> ('', Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True)) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个迭代器，遍历模块参数，产生参数的名称以及参数本身。

参数：

+   **prefix**（*str*）- 要添加到所有参数名称前面的前缀。

+   **recurse**（*bool*）- 如果为 True，则会产生此模块及所有子模块的参数。否则，只会产生此模块的直接成员参数。

+   **remove_duplicate**（*bool**，*可选*）- 是否删除结果中的重复参数。默认为 True。

产生：

*(str, Parameter)* - 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
output_dist(ctx: NullShardedModuleContext, output: Tensor) → LazyAwaitable[Tensor]
```

```py
sharded_parameter_names(prefix: str = '') → Iterator[str]
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含模块整体状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**，*可选*）- 如果提供，则模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix** (*str**,* *optional*) – 用于组成状态字典中键的参数和缓冲区名称的前缀。默认值：`''`。

+   **keep_vars** (*bool**,* *optional*) – 默认情况下，状态字典中返回的`Tensor`会从自动求导中分离出来。如果设置为`True`，则不会执行分离操作。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
class torchrec.distributed.embeddingbag.ShardedEmbeddingBagCollection(module: EmbeddingBagCollectionInterface, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ShardedEmbeddingModule`[`KJTList`, `List`[`Tensor`], `KeyedTensor`, `EmbeddingBagCollectionContext`], `FusedOptimizerModule`

EmbeddingBagCollection 的分片实现。这是公共 API 的一部分，允许手动数据分布流水线化。

```py
compute(ctx: EmbeddingBagCollectionContext, dist_input: KJTList) → List[Tensor]
```

```py
compute_and_output_dist(ctx: EmbeddingBagCollectionContext, input: KJTList) → LazyAwaitable[KeyedTensor]
```

在存在多个输出分布的情况下，重写此方法并在相应的计算完成后立即初始化输出分布是有意义的。

```py
create_context() → EmbeddingBagCollectionContext
```

```py
property fused_optimizer: KeyedOptimizer
```

```py
input_dist(ctx: EmbeddingBagCollectionContext, features: KeyedJaggedTensor) → Awaitable[Awaitable[KJTList]]
```

```py
output_dist(ctx: EmbeddingBagCollectionContext, output: List[Tensor]) → LazyAwaitable[KeyedTensor]
```

```py
reset_parameters() → None
```

```py
training: bool
```

```py
class torchrec.distributed.embeddingbag.VariableBatchEmbeddingBagCollectionAwaitable(*args, **kwargs)
```

基类：`LazyAwaitable`[`KeyedTensor`]

```py
torchrec.distributed.embeddingbag.construct_output_kt(embeddings: List[Tensor], embedding_names: List[str], embedding_dims: List[int]) → KeyedTensor
```

```py
torchrec.distributed.embeddingbag.create_embedding_bag_sharding(sharding_type: str, sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, permute_embeddings: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None) → EmbeddingSharding[EmbeddingShardingContext, KeyedJaggedTensor, Tensor, Tensor]
```

```py
torchrec.distributed.embeddingbag.create_sharding_infos_by_sharding(module: EmbeddingBagCollectionInterface, table_name_to_parameter_sharding: Dict[str, ParameterSharding], prefix: str, fused_params: Optional[Dict[str, Any]], suffix: Optional[str] = 'weight') → Dict[str, List[EmbeddingShardingInfo]]
```

```py
torchrec.distributed.embeddingbag.replace_placement_with_meta_device(sharding_infos: List[EmbeddingShardingInfo]) → None
```

在某些情况下，放置设备和张量设备可能不匹配，例如将元设备传递给 DMP 并将 cuda 传递给 EmbeddingShardingPlanner。在获取分片规划器后，我们需要使设备保持一致。## torchrec.distributed.grouped_position_weighted

```py
class torchrec.distributed.grouped_position_weighted.GroupedPositionWeightedModule(max_feature_lengths: Dict[str, int], device: Optional[device] = None)
```

基类：`BaseGroupedFeatureProcessor`

```py
forward(features: KeyedJaggedTensor) → KeyedJaggedTensor
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

尽管前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此处调用，因为前者会负责运行注册的钩子，而后者会默默地忽略它们。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个模块缓冲区的迭代器，同时返回缓冲区的名称和缓冲区本身。

参数：

+   **prefix** (*str*) – 添加到所有缓冲区名称前面的前缀。

+   **recurse** (*bool**,* *optional*) – 如果为 True，则返回此模块及所有子模块的缓冲区。否则，只返回此模块的直接成员缓冲区。默认为 True。

+   **remove_duplicate** (*bool**,* *optional*) – 是否在结果中删除重复的缓冲区。默认为 True。

产出：

*(str, torch.Tensor)* – 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个模块参数的迭代器，同时返回参数的名称和参数本身。

参数：

+   **prefix** (*str*) – 添加到所有参数名称前面的前缀。

+   **recurse** (*bool*) – 如果为 True，则返回此模块及所有子模块的参数。否则，只返回此模块的直接成员参数。

+   **remove_duplicate** (*bool**,* *optional*) – 是否在结果中删除重复的参数。默认为 True。

产出：

*(str, Parameter)* – 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含对模块整体状态的引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是对应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

当前`state_dict()`还接受位置参数`destination`、`prefix`和`keep_vars`。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination** (*dict**,* *可选*) – 如果提供，模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix** (*str**,* *可选*) – 添加到参数和缓冲区名称以组成 state_dict 中键的前缀。默认值：`''`。

+   **keep_vars** (*bool**,* *可选*) – 默认情况下，在状态字典中返回的`Tensor`会与自动求导分离。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```  ## torchrec.distributed.model_parallel

```py
class torchrec.distributed.model_parallel.DataParallelWrapper
```

基类：`ABC`

由自定义数据并行包装器实现的接口。

```py
abstract wrap(dmp: DistributedModelParallel, env: ShardingEnv, device: device) → None
```

```py
class torchrec.distributed.model_parallel.DefaultDataParallelWrapper(bucket_cap_mb: int = 25, static_graph: bool = True, find_unused_parameters: bool = False, allreduce_comm_precision: Optional[str] = None)
```

基类：`DataParallelWrapper`

默认数据并行包装器，将数据并行应用于所有未分片的模块。

```py
wrap(dmp: DistributedModelParallel, env: ShardingEnv, device: device) → None
```

```py
class torchrec.distributed.model_parallel.DistributedModelParallel(module: Module, env: Optional[ShardingEnv] = None, device: Optional[device] = None, plan: Optional[ShardingPlan] = None, sharders: Optional[List[ModuleSharder[Module]]] = None, init_data_parallel: bool = True, init_parameters: bool = True, data_parallel_wrapper: Optional[DataParallelWrapper] = None)
```

基类：`Module`，`FusedOptimizerModule`

模型并行的入口点。

参数：

+   **module** (*nn.Module*) – 要包装的模块。

+   **env** (*可选***[*ShardingEnv**]*) – 具有进程组的分片环境。

+   **device** (*可选**[**torch.device**]*) – 计算设备，默认为 cpu。

+   **plan** (*可选***[*ShardingPlan**]*) – 在分片时使用的计划，默认为 EmbeddingShardingPlanner.collective_plan()。

+   **sharders** (*可选****List**[*[*ModuleSharder**[**nn.Module**]**]**]*) – 可用于分片的 ModuleSharders，默认为 EmbeddingBagCollectionSharder()。

+   **init_data_parallel** (*bool*) – 数据并行模块可以是懒惰的，即它们延迟参数初始化直到第一次前向传递。传递 True 以延迟数据并行模块的初始化。进行第一次前向传递，然后调用 DistributedModelParallel.init_data_parallel()。

+   **init_parameters** (*bool*) – 为仍在元设备上的模块初始化参数。

+   **data_parallel_wrapper** (*可选***[*DataParallelWrapper**]*) – 数据并行模块的自定义包装器。

示例：

```py
@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.fill_(1.0)
    elif isinstance(m, EmbeddingBagCollection):
        for param in m.parameters():
            init.kaiming_normal_(param)

m = MyModel(device='meta')
m = DistributedModelParallel(m)
m.apply(init_weights) 
```

```py
bare_named_parameters(prefix: str = '', recurse: bool = True) → Iterator[Tuple[str, Parameter]]
```

```py
copy(device: device) → DistributedModelParallel
```

通过调用每个模块的自定义复制过程递归地将子模块复制到新设备，因为一些模块需要使用原始引用（例如用于推理的 ShardedModule）。

```py
forward(*args, **kwargs) → Any
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

尽管前向传递的方法需要在此函数内定义，但应该在此之后调用`Module`实例，而不是此方法，因为前者会负责运行注册的钩子，而后者会默默地忽略它们。

```py
property fused_optimizer: KeyedOptimizer
```

```py
init_data_parallel() → None
```

查看 init_data_parallel c-tor 参数的用法。可以多次调用此方法。

```py
load_state_dict(state_dict: OrderedDict[str, torch.Tensor], prefix: str = '', strict: bool = True) → _IncompatibleKeys
```

将参数和缓冲区从`state_dict`复制到此模块及其后代中。

如果`strict`为`True`，则`state_dict`的键必须与此模块的`state_dict()`函数返回的键完全匹配。

警告

如果`assign`为`True`，则必须在调用`load_state_dict`之后创建优化器。

参数：

+   **state_dict**（*dict*）- 包含参数和持久缓冲区的字典。

+   **strict**（*bool**，*可选*）- 是否严格执行`state_dict`中的键与此模块的`state_dict()`函数返回的键匹配。默认：`True`

+   **assign**（*bool**，*可选*）- 是否将状态字典中的项目分配给模块中的相应键，而不是将它们原地复制到模块的当前参数和缓冲区中。当`False`时，保留当前模块张量的属性，而当`True`时，保留状态字典中张量的属性。默认：`False`

返回：

+   **missing_keys**是一个包含缺失键的字符串列表

+   **unexpected_keys**是一个包含意外键的字符串列表

返回类型：

带有`missing_keys`和`unexpected_keys`字段的`NamedTuple`

注意

如果参数或缓冲区注册为`None`，并且其对应的键存在于`state_dict`中，`load_state_dict()`将引发`RuntimeError`。

```py
property module: Module
```

直接访问分片模块的属性，该模块不会包含在 DDP、FSDP、DMP 或任何其他并行包装器中。

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]
```

返回一个迭代器，遍历模块缓冲区，产出缓冲区的名称以及缓冲区本身。

参数：

+   **prefix**（*str*）- 要添加到所有缓冲区名称前面的前缀。

+   **recurse**（*bool**，*可选*）- 如果为 True，则产出此模块及所有子模块的缓冲区。否则，仅产出此模块的直接成员缓冲区。默认为 True。

+   **remove_duplicate**（*bool**，*可选*）- 是否在结果中删除重复的缓冲区。默认为 True。

产量：

（str，torch.Tensor）- 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Parameter]]
```

返回一个迭代器，遍历模块参数，产出参数的名称以及参数本身。

参数：

+   **prefix**（*str*）- 要添加到所有参数名称前面的前缀。

+   **recurse**（*bool*）- 如果为 True，则产出此模块及所有子模块的参数。否则，仅产出此模块的直接成员参数。

+   **remove_duplicate**（*bool**，*可选*）- 是否在结果中删除重复的参数。默认为 True。

产量：

（str，Parameter）- 包含名称和参数的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>>     if name in ['bias']:
>>>         print(param.size()) 
```

```py
property plan: ShardingPlan
```

```py
sparse_grad_parameter_names(destination: Optional[List[str]] = None, prefix: str = '') → List[str]
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]
```

返回一个包含模块整体状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前`state_dict()`还接受`destination`、`prefix`和`keep_vars`的位置参数，但这将被弃用，并且将在未来的版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**，*可选*）- 如果提供，则模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix**（*str**，*可选*）- 添加到参数和缓冲区名称以组成 state_dict 中键的前缀。默认值：`''`。

+   **keep_vars**（*bool**，*可选*）- 默认情况下，在状态字典中返回的`Tensor`是从自动求导中分离的。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整体状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool
```

```py
torchrec.distributed.model_parallel.get_module(module: Module) → Module
```

解开 DMP 模块。

不解开数据并行包装器（即 DDP/FSDP），因此可以使用包装器的覆盖实现。

```py
torchrec.distributed.model_parallel.get_unwrapped_module(module: Module) → Module
```

解开由 DMP、DDP 或 FSDP 包装的模块。## torchrec.distributed.quant_embeddingbag

```py
class torchrec.distributed.quant_embeddingbag.QuantEmbeddingBagCollectionSharder(fused_params: Optional[Dict[str, Any]] = None, shardable_params: Optional[List[str]] = None)
```

基类：`BaseQuantEmbeddingSharder`[`EmbeddingBagCollection`]

```py
property module_type: Type[EmbeddingBagCollection]
```

```py
shard(module: EmbeddingBagCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedQuantEmbeddingBagCollection
```

执行实际分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module**（*M*）- 要分片的模块。

+   **params**（*EmbeddingModuleShardingPlan*[Any, Any, Any]

```py
class torchrec.distributed.quant_embeddingbag.QuantFeatureProcessedEmbeddingBagCollectionSharder(fused_params: Optional[Dict[str, Any]] = None, shardable_params: Optional[List[str]] = None)
```

基类：`BaseQuantEmbeddingSharder`[`FeatureProcessedEmbeddingBagCollection`]

```py
compute_kernels(sharding_type: str, compute_device_type: str) → List[str]
```

给定分片类型和计算设备的支持计算内核列表。

```py
property module_type: Type[FeatureProcessedEmbeddingBagCollection]
```

```py
shard(module: FeatureProcessedEmbeddingBagCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedQuantEmbeddingBagCollection
```

执行实际分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module**（*M*）- 要分片的模块。

+   **params**（*EmbeddingModuleShardingPlan*[Any, Any, Any]

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。请参见 ShardingType 以获取知名示例。

```py
class torchrec.distributed.quant_embeddingbag.ShardedQuantEmbeddingBagCollection(module: EmbeddingBagCollectionInterface, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None)
```

基类：`ShardedQuantEmbeddingModuleState`[`ListOfKJTList`, `List`[`List`[`Tensor`]], `KeyedTensor`, `NullShardedModuleContext`]

EmbeddingBagCollection 的 Sharded 实现。这是公共 API 的一部分，允许手动数据分布流水线化。

```py
compute(ctx: NullShardedModuleContext, dist_input: ListOfKJTList) → List[List[Tensor]]
```

```py
compute_and_output_dist(ctx: NullShardedModuleContext, input: ListOfKJTList) → KeyedTensor
```

在存在多个输出分布的情况下，重写此方法并在相应的计算完成后立即启动输出分布是有意义的。

```py
copy(device: device) → Module
```

```py
create_context() → NullShardedModuleContext
```

```py
embedding_bag_configs() → List[EmbeddingBagConfig]
```

```py
forward(*input, **kwargs) → KeyedTensor
```

执行输入 dist、compute 和输出 dist 步骤。

参数：

+   ***input** – 输入。

+   ****kwargs** – 关键字参数。

返回：

来自输出 dist 的可等待对象。

返回类型：

LazyAwaitable[Out]

```py
input_dist(ctx: NullShardedModuleContext, features: KeyedJaggedTensor) → ListOfKJTList
```

```py
output_dist(ctx: NullShardedModuleContext, output: List[List[Tensor]]) → KeyedTensor
```

```py
sharding_type_to_sharding_infos() → Dict[str, List[EmbeddingShardingInfo]]
```

```py
property shardings: Dict[str, FeatureShardingMixIn]
```

```py
tbes_configs() → Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]
```

```py
training: bool
```

```py
class torchrec.distributed.quant_embeddingbag.ShardedQuantFeatureProcessedEmbeddingBagCollection(module: EmbeddingBagCollectionInterface, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, fused_params: Optional[Dict[str, Any]] = None, device: Optional[device] = None, feature_processor: Optional[FeatureProcessorsCollection] = None)
```

基类：`ShardedQuantEmbeddingBagCollection`

```py
apply_feature_processor(kjt_list: KJTList) → KJTList
```

```py
compute(ctx: NullShardedModuleContext, dist_input: ListOfKJTList) → List[List[Tensor]]
```

```py
embedding_bags: nn.ModuleDict
```

```py
tbes: torch.nn.ModuleList
```

```py
training: bool
```

```py
torchrec.distributed.quant_embeddingbag.create_infer_embedding_bag_sharding(sharding_type: str, sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv) → EmbeddingSharding[NullShardingContext, KJTList, List[Tensor], Tensor]
```

```py
torchrec.distributed.quant_embeddingbag.flatten_feature_lengths(features: KeyedJaggedTensor) → KeyedJaggedTensor
```  ## torchrec.distributed.train_pipeline

注意：由于内部打包问题，train_pipeline.py 必须与较旧版本的 TorchRec 兼容。从其他文件导入新模块可能会破坏模型发布流程。

```py
class torchrec.distributed.train_pipeline.ArgInfo(input_attrs: List[str], is_getitems: List[bool], name: Optional[str])
```

基类：`object`

来自节点的参数表示。

```py
input_attrs
```

输入批次的属性，例如 batch.attr1.attr2 将产生 [“attr1”, “attr2”]。

类型：

List[str]

```py
is_getitems
```

batch[attr1].attr2 将产生 [True, False]。

类型：

List[bool]

```py
name
```

用于流水线化 forward() 调用的关键字参数的名称，或者对于位置参数为 None。

类型：

可选的[str]

```py
input_attrs: List[str]
```

```py
is_getitems: List[bool]
```

```py
name: Optional[str]
```

```py
class torchrec.distributed.train_pipeline.BaseForward(name: str, args: List[ArgInfo], module: ShardedModule, context: TrainPipelineContext, stream: Optional[Stream])
```

基类：`object`

```py
property args: List[ArgInfo]
```

```py
property name: str
```

```py
class torchrec.distributed.train_pipeline.FusedKJTListSplitsAwaitable(requests: List[KJTListSplitsAwaitable[C]], contexts: List[C], pg: Optional[ProcessGroup])
```

基类：`Awaitable``List`[[`KJTListAwaitable`]]

```py
class torchrec.distributed.train_pipeline.KJTAllToAllForward(pg: ProcessGroup, splits: List[int], stagger: int = 1)
```

基类：`object`

```py
class torchrec.distributed.train_pipeline.KJTSplitsAllToAllMeta(pg: torch.distributed.distributed_c10d.ProcessGroup, _input: torchrec.sparse.jagged_tensor.KeyedJaggedTensor, splits: List[int], splits_tensors: List[torch.Tensor], input_splits: List[List[int]], input_tensors: List[torch.Tensor], labels: List[str], keys: List[str], device: torch.device, stagger: int)
```

基类：`object`

```py
device: device
```

```py
input_splits: List[List[int]]
```

```py
input_tensors: List[Tensor]
```

```py
keys: List[str]
```

```py
labels: List[str]
```

```py
pg: ProcessGroup
```

```py
splits: List[int]
```

```py
splits_tensors: List[Tensor]
```

```py
stagger: int
```

```py
class torchrec.distributed.train_pipeline.PipelinedForward(name: str, args: List[ArgInfo], module: ShardedModule, context: TrainPipelineContext, stream: Optional[Stream])
```

基类：`BaseForward`

```py
class torchrec.distributed.train_pipeline.PrefetchPipelinedForward(name: str, args: List[ArgInfo], module: ShardedModule, context: PrefetchTrainPipelineContext, prefetch_stream: Optional[Stream])
```

基类：`BaseForward`

```py
class torchrec.distributed.train_pipeline.PrefetchTrainPipelineContext(input_dist_splits_requests: Dict[str, torchrec.distributed.types.Awaitable[Any]] = <factory>, input_dist_tensors_requests: Dict[str, torchrec.distributed.types.Awaitable[Any]] = <factory>, module_contexts: Dict[str, torchrec.streamable.Multistreamable] = <factory>, module_contexts_next_batch: Dict[str, torchrec.streamable.Multistreamable] = <factory>, fused_splits_awaitables: List[Tuple[List[str], torchrec.distributed.train_pipeline.FusedKJTListSplitsAwaitable]] = <factory>, module_input_post_prefetch: Dict[str, torchrec.streamable.Multistreamable] = <factory>, module_contexts_post_prefetch: Dict[str, torchrec.streamable.Multistreamable] = <factory>)
```

基类：`TrainPipelineContext`

```py
module_contexts_post_prefetch: Dict[str, Multistreamable]
```

```py
module_input_post_prefetch: Dict[str, Multistreamable]
```

```py
class torchrec.distributed.train_pipeline.PrefetchTrainPipelineSparseDist(model: Module, optimizer: Optimizer, device: device, execute_all_batches: bool = True, apply_jit: bool = False)
```

基类：`TrainPipelineSparseDist`[`In`, `Out`]

该流水线将设备传输、ShardedModule.input_dist() 和缓存预取与前向和后向操作重叠。这有助于隐藏 all2all 延迟，同时保留训练前向/后向顺序。

阶段 4：前向、后向 - 使用默认的 CUDA 流 阶段 3：预取 - 使用预取 CUDA 流 阶段 2：ShardedModule.input_dist() - 使用数据分布 CUDA 流 阶段 1：数据传输 - 使用 memcpy CUDA 流

ShardedModule.input_dist() 仅针对调用图中的顶层模块执行。要被视为顶层模块，一个模块只能依赖于输入上的 'getattr' 调用。

输入模型必须是符号化可跟踪的，除了 ShardedModule 和 DistributedDataParallel 模块。

参数：

+   **model** (*torch.nn.Module*) – 要进行流水线处理的模型。

+   **optimizer** (*torch.optim.Optimizer*) – 要使用的优化器。

+   **device** (*torch.device*) – 设备，数据传输、稀疏数据分布、预取和前向/后向传递将在该设备上进行。

+   **execute_all_batches** (*bool*) – 在耗尽数据加载器迭代器后执行流水线中剩余的批次。

+   **apply_jit** (*bool*) – 对非流水线化（未分片）模块应用 torch.jit.script。

```py
progress(dataloader_iter: Iterator[In]) → Out
```

```py
class torchrec.distributed.train_pipeline.SplitsAllToAllAwaitable(input_tensors: List[Tensor], pg: ProcessGroup)
```

基类：`Awaitable`[`List`[`List`[`int`]]]

```py
class torchrec.distributed.train_pipeline.Tracer(leaf_modules: Optional[List[str]] = None)
```

基类：`Tracer`

在跟踪期间禁用代理缓冲区。理想情况下，代理缓冲区应该被禁用，但是一些模型目前正在改变缓冲区的值，这会在跟踪期间导致错误。如果这些模型可以重写以避免这种情况，我们很可能可以删除这行。

```py
graph: Graph
```

```py
is_leaf_module(m: Module, module_qualified_name: str) → bool
```

指定给定的 `nn.Module` 是否是“叶”模块的方法。

叶子模块是出现在 IR 中的原子单元，由`call_module`调用引用。默认情况下，PyTorch 标准库命名空间（torch.nn）中的模块是叶子模块。除非通过此参数另有规定，否则将跟踪所有其他模块，并记录其组成操作。

参数：

+   **m** (*Module*) – 正在查询的模块

+   **module_qualified_name** (*str*) – 此模块根目录的路径。例如，如果您有一个模块层次结构，其中子模块`foo`包含子模块`bar`，子模块`bar`包含子模块`baz`，那么该模块将在此处显示为限定名称`foo.bar.baz`。

注意

此 API 的向后兼容性已得到保证。

```py
module_stack: OrderedDict[str, Tuple[str, Any]]
```

```py
node_name_to_scope: Dict[str, Tuple[str, type]]
```

```py
proxy_buffer_attributes: bool = False
```

```py
scope: Scope
```

```py
class torchrec.distributed.train_pipeline.TrainPipeline(*args, **kwds)
```

基类：`ABC`，`Generic`[`In`, `Out`]

```py
abstract progress(dataloader_iter: Iterator[In]) → Out
```

```py
class torchrec.distributed.train_pipeline.TrainPipelineBase(model: Module, optimizer: Optimizer, device: device)
```

基类：`TrainPipeline`[`In`, `Out`]

此类使用两个阶段的管道运行训练迭代，每个阶段作为一个 CUDA 流，即当前（默认）流和 self._memcpy_stream。对于每次迭代，self._memcpy_stream 将输入从主机（CPU）内存移动到 GPU 内存，而默认流则运行前向、后向和优化。

```py
progress(dataloader_iter: Iterator[In]) → Out
```

```py
class torchrec.distributed.train_pipeline.TrainPipelineContext(input_dist_splits_requests: ~typing.Dict[str, ~torchrec.distributed.types.Awaitable[~typing.Any]] = <factory>, input_dist_tensors_requests: ~typing.Dict[str, ~torchrec.distributed.types.Awaitable[~typing.Any]] = <factory>, module_contexts: ~typing.Dict[str, ~torchrec.streamable.Multistreamable] = <factory>, module_contexts_next_batch: ~typing.Dict[str, ~torchrec.streamable.Multistreamable] = <factory>, fused_splits_awaitables: ~typing.List[~typing.Tuple[~typing.List[str], ~torchrec.distributed.train_pipeline.FusedKJTListSplitsAwaitable]] = <factory>)
```

基类：`object`

TrainPipelineSparseDist 实例的上下文信息。

```py
input_dist_splits_requests
```

将输入分布请求存储在分片等待阶段，在启动输入分布之后发生。

类型：

Dictstr, [Awaitable[Any]]

```py
input_dist_tensors_requests
```

存储在张量等待阶段的输入分布请求，发生在对分片等待进行等待()之后。

类型：

Dictstr, [Awaitable[Any]]

```py
module_contexts
```

存储来自输入分布的模块上下文，用于当前批次。

类型：

Dict[str, Multistreamable]

```py
module_contexts_next_batch
```

存储来自输入分布的模块上下文，用于下一批次。

类型：

Dict[str, Multistreamable]

```py
fused_splits_awaitables
```

融合分片输入分布等待和每个等待的相应模块名称的列表。

类型：

List[Tuple[List[str], FusedKJTListSplitsAwaitable]]

```py
fused_splits_awaitables: List[Tuple[List[str], FusedKJTListSplitsAwaitable]]
```

```py
input_dist_splits_requests: Dict[str, Awaitable[Any]]
```

```py
input_dist_tensors_requests: Dict[str, Awaitable[Any]]
```

```py
module_contexts: Dict[str, Multistreamable]
```

```py
module_contexts_next_batch: Dict[str, Multistreamable]
```

```py
class torchrec.distributed.train_pipeline.TrainPipelineSparseDist(model: Module, optimizer: Optimizer, device: device, execute_all_batches: bool = True, apply_jit: bool = False)
```

基类：`TrainPipeline`[`In`, `Out`]

此管道重叠设备传输和 ShardedModule.input_dist()与前向和后向。这有助于隐藏 all2all 延迟，同时保留训练前向/后向顺序。

阶段 3：前向、后向 - 使用默认的 CUDA 流阶段 2：ShardedModule.input_dist() - 使用 data_dist CUDA 流阶段 1：设备传输 - 使用 memcpy CUDA 流

ShardedModule.input_dist()仅针对调用图中的顶级模块执行。要被视为顶级模块，模块只能依赖于对输入的‘getattr’调用。

输入模型必须是符号跟踪的，除了 ShardedModule 和 DistributedDataParallel 模块。

参数：

+   **model** (*torch.nn.Module*) – 要进行流水线处理的模型。

+   **optimizer** (*torch.optim.Optimizer*) – 要使用的优化器。

+   **device** (*torch.device*) – 设备，其中将发生设备传输、稀疏数据分布和前向/后向传递。

+   **execute_all_batches** (*bool*) – 在耗尽数据加载器迭代器后执行管道中剩余的批次。

+   **apply_jit** (*bool*) – 对非流水线（未分片）模块应用 torch.jit.script。

```py
progress(dataloader_iter: Iterator[In]) → Out
```  ## torchrec.distributed.types

```py
class torchrec.distributed.types.Awaitable
```

基类：`ABC`，`Generic`[`W`]

```py
property callbacks: List[Callable[[W], W]]
```

```py
wait() → W
```

```py
class torchrec.distributed.types.CacheParams(algorithm: Union[fbgemm_gpu.split_table_batched_embeddings_ops_common.CacheAlgorithm, NoneType] = None, load_factor: Union[float, NoneType] = None, reserved_memory: Union[float, NoneType] = None, precision: Union[torchrec.types.DataType, NoneType] = None, prefetch_pipeline: Union[bool, NoneType] = None, stats: Union[torchrec.distributed.types.CacheStatistics, NoneType] = None)
```

基类：`object`

```py
algorithm: Optional[CacheAlgorithm] = None
```

```py
load_factor: Optional[float] = None
```

```py
precision: Optional[DataType] = None
```

```py
prefetch_pipeline: Optional[bool] = None
```

```py
reserved_memory: Optional[float] = None
```

```py
stats: Optional[CacheStatistics] = None
```

```py
class torchrec.distributed.types.CacheStatistics
```

基类：`ABC`

```py
abstract property cacheability: float
```

缓存数据集的难度的总结性度量，独立于缓存大小。得分为 0 表示数据集非常适合缓存（例如，访问之间的局部性很高），得分为 1 表示非常难以缓存。

```py
abstract property expected_lookups: float
```

每个训练步骤中预期的缓存查找次数。

这是全局训练批次中预期的不同值的数量。

```py
abstract expected_miss_rate(clf: float) → float
```

给定缓存大小的预期缓存查找未命中率。

当 clf（缓存加载因子）为 0 时，返回 1.0（100% 未命中）。当 clf 为 1.0 时，返回 0（100% 命中）。对于介于这些极端值之间的 clf 值，返回缓存的估计未命中率，例如基于训练数据集的统计属性的知识。

```py
class torchrec.distributed.types.CommOp(value)
```

基类：`Enum`

一个枚举。

```py
POOLED_EMBEDDINGS_ALL_TO_ALL = 'pooled_embeddings_all_to_all'
```

```py
POOLED_EMBEDDINGS_REDUCE_SCATTER = 'pooled_embeddings_reduce_scatter'
```

```py
SEQUENCE_EMBEDDINGS_ALL_TO_ALL = 'sequence_embeddings_all_to_all'
```

```py
class torchrec.distributed.types.ComputeKernel(value)
```

基类：`Enum`

一个枚举。

```py
DEFAULT = 'default'
```

```py
class torchrec.distributed.types.EmbeddingModuleShardingPlan
```

基类：`ModuleShardingPlan`，`Dict``str`, [`ParameterSharding`]

每个参数（通常是一个表）的 ParameterSharding 映射。这描述了 torchrec 模块的分片计划（例如 EmbeddingBagCollection）

```py
class torchrec.distributed.types.GenericMeta
```

基类：`type`

```py
class torchrec.distributed.types.LazyAwaitable(*args, **kwargs)
```

基类：`Awaitable`[`W`]

LazyAwaitable 类型公开了一个 wait() API，具体类型可以控制如何初始化以及等待() 行为应该如何以实现特定的异步操作。

这个基本的 LazyAwaitable 类型是一个“懒惰”的异步类型，这意味着它会尽可能延迟 wait()，请参见下面的 __torch_function__ 中的详细信息。这可以帮助模型自动启用计算和通信重叠，模型作者不需要手动调用 wait()，如果结果被一个 pytorch 函数使用，或者被其他 python 操作使用（注意：需要实现类似 __getattr__ 的相应魔术方法）

一些注意事项：

+   这适用于 Pytorch 函数，但不适用于任何通用方法，如果您想执行任意的 python 操作，您需要实现相应的魔术方法

+   在一个函数有两个或更多个 LazyAwaitable 参数的情况下，懒惰等待机制无法确保完美的计算/通信重叠（即快速等待第一个，但在第二个上等待时间较长）

```py
class torchrec.distributed.types.LazyNoWait(*args, **kwargs)
```

基类：`LazyAwaitable`[`W`]

```py
class torchrec.distributed.types.ModuleSharder(qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ABC`，`Generic`[`M`]

ModuleSharder 是每个支持分片的模块，例如 EmbeddingBagCollection。

参数：

qcomm_codecs_registry（可选[Dict[str, QuantizedCommCodecs]]）：CommOp 名称到 QuantizedCommCodecs 的映射

```py
compute_kernels(sharding_type: str, compute_device_type: str) → List[str]
```

给定分片类型和计算设备的支持的计算内核列表。

```py
abstract property module_type: Type[M]
```

```py
property qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]]
```

```py
abstract classmethod shard(module: M, params: EmbeddingModuleShardingPlan, env: ShardingEnv, device: Optional[device] = None) → ShardedModule[Any, Any, Any, Any]
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module**（*M*）- 要分片的模块。

+   **params**（*EmbeddingModuleShardingPlan*）- 完全限定的参数名称（模块路径 + 参数名称，用‘.’分隔）到其分片规范的字典。

+   **env**（*ShardingEnv*）- 具有进程组的分片环境。

+   **device**（*torch.device*）- 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
shardable_parameters(module: M) → Dict[str, Parameter]
```

可以分片的参数列表。

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。查看 ShardingType 以获取知名示例。

```py
storage_usage(tensor: Tensor, compute_device_type: str, compute_kernel: str) → Dict[str, int]
```

给定计算设备和计算内核的系统资源列表及相应的使用情况。

```py
class torchrec.distributed.types.ModuleShardingPlan
```

基类：`object`

```py
class torchrec.distributed.types.NoOpQuantizedCommCodec(*args, **kwds)
```

基类：`Generic`[`QuantizationContext`]

QuantizedCommCodec 的默认无操作实现

```py
calc_quantized_size(input_len: int, ctx: Optional[QuantizationContext] = None) → int
```

```py
create_context() → Optional[QuantizationContext]
```

```py
decode(input_grad: Tensor, ctx: Optional[QuantizationContext] = None) → Tensor
```

```py
encode(input_tensor: Tensor, ctx: Optional[QuantizationContext] = None) → Tensor
```

```py
quantized_dtype() → dtype
```

```py
class torchrec.distributed.types.NoWait(obj: W)
```

基类：`Awaitable`[`W`]

```py
class torchrec.distributed.types.NullShardedModuleContext
```

基类：`Multistreamable`

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
class torchrec.distributed.types.NullShardingContext
```

基类：`Multistreamable`

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
class torchrec.distributed.types.ParameterSharding(sharding_type: str, compute_kernel: str, ranks: Optional[List[int]] = None, sharding_spec: Optional[ShardingSpec] = None, cache_params: Optional[CacheParams] = None, enforce_hbm: Optional[bool] = None, stochastic_rounding: Optional[bool] = None, bounds_check_mode: Optional[BoundsCheckMode] = None)
```

基类：`object`

> 描述参数的分片。
> 
> sharding_type（str）：此参数的分片方式。有关众所周知的 ShardingType
> 
> 类型。
> 
> compute_kernel（str）：此参数要使用的计算内核。ranks（Optional[List[int]]）：每个分片的等级。sharding_spec（Optional[ShardingSpec]）：每个分片的 ShardMetadata 列表。cache_params（Optional[CacheParams]）：嵌入查找的缓存参数。enforce_hbm（Optional[bool]）：是否使用 HBM。stochastic_rounding（Optional[bool]）：是否使用随机舍入。bounds_check_mode（Optional[BoundsCheckMode]）：边界检查模式。

注意

ShardingType.TABLE_WISE - 放置此嵌入的等级 ShardingType.COLUMN_WISE - 放置嵌入分片的等级，视为单独的表 ShardingType.TABLE_ROW_WISE - 放置此嵌入时的第一个等级 ShardingType.ROW_WISE，ShardingType.DATA_PARALLEL - 未使用

```py
bounds_check_mode: Optional[BoundsCheckMode] = None
```

```py
cache_params: Optional[CacheParams] = None
```

```py
compute_kernel: str
```

```py
enforce_hbm: Optional[bool] = None
```

```py
ranks: Optional[List[int]] = None
```

```py
sharding_spec: Optional[ShardingSpec] = None
```

```py
sharding_type: str
```

```py
stochastic_rounding: Optional[bool] = None
```

```py
class torchrec.distributed.types.ParameterStorage(value)
```

基类：`Enum`

众所周知的物理资源，可用作 ShardingPlanner 的约束。

```py
DDR = 'ddr'
```

```py
HBM = 'hbm'
```

```py
class torchrec.distributed.types.QuantizedCommCodec(*args, **kwds)
```

基类：`Generic`[`QuantizationContext`]

为在集体调用（pooled_all_to_all、reduce_scatter 等）中使用的张量提供量化或应用混合精度的实现。dtype 是从 encode 调用的张量的 dtype。

这假设输入张量的类型为 torch.float32

```py
>>>
 quantized_tensor = quantized_comm_codec.encode(input_tensor)
 quantized_tensor.dtype == quantized_comm_codec.quantized_dtype
 collective_call(output_tensors, input_tensors=tensor)
 output_tensor = decode(output_tensors) 
```

> torch.assert_close(input_tensors, output_tensor)

```py
calc_quantized_size(input_len: int, ctx: Optional[QuantizationContext] = None) → int
```

根据输入张量的长度，返回量化后的张量长度。由 INT8 编解码器使用，其中量化张量具有一些额外参数。对于其他情况，量化张量应与输入具有相同的长度。

```py
create_context() → Optional[QuantizationContext]
```

创建一个可用于在编码器和解码器之间传递基于会话的参数的上下文对象。

```py
decode(input_grad: Tensor, ctx: Optional[QuantizationContext] = None) → Tensor
```

```py
encode(input_tensor: Tensor, ctx: Optional[QuantizationContext] = None) → Tensor
```

```py
property quantized_dtype: dtype
```

结果编码（input_tensor）的张量数据类型

```py
class torchrec.distributed.types.QuantizedCommCodecs(forward: ~torchrec.distributed.types.QuantizedCommCodec = <torchrec.distributed.types.NoOpQuantizedCommCodec object>, backward: ~torchrec.distributed.types.QuantizedCommCodec = <torchrec.distributed.types.NoOpQuantizedCommCodec object>)
```

基类：`object`

用于 comm op（例如 pooled_all_to_all、reduce_scatter、sequence_all_to_all）的前向和后向传递的量化编解码器。

```py
backward: QuantizedCommCodec = <torchrec.distributed.types.NoOpQuantizedCommCodec object>
```

```py
forward: QuantizedCommCodec = <torchrec.distributed.types.NoOpQuantizedCommCodec object>
```

```py
class torchrec.distributed.types.ShardedModule(qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ABC`，`Module`，`Generic`[`CompIn`，`DistOut`，`Out`，`ShrdCtx`]，`ModuleNoCopyMixin`

所有模型并行模块都实现了此接口。输入和输出是数据并行的。

Args::

qcomm_codecs_registry（Optional[Dict[str, QuantizedCommCodecs]]）：CommOp 名称到 QuantizedCommCodecs 的映射

注意

‘input_dist’ / ‘output_dist’负责将输入/输出从数据并行转换为模型并行，反之亦然。

```py
abstract compute(ctx: ShrdCtx, dist_input: CompIn) → DistOut
```

```py
compute_and_output_dist(ctx: ShrdCtx, input: CompIn) → LazyAwaitable[Out]
```

在存在多个输出分布的情况下，重写此方法并在相应的计算完成后立即启动输出分布是有意义的。

```py
abstract create_context() → ShrdCtx
```

```py
forward(*input, **kwargs) → LazyAwaitable[Out]
```

执行输入分布、计算和输出分布步骤。

参数：

+   ***input** - 输入。

+   ****kwargs** - 关键字参数。

返回：

从输出分布中获取输出的可等待对象。

返回类型：

LazyAwaitable[Out]

```py
abstract input_dist(ctx: ShrdCtx, *input, **kwargs) → Awaitable[Awaitable[CompIn]]
```

```py
abstract output_dist(ctx: ShrdCtx, output: DistOut) → LazyAwaitable[Out]
```

```py
property qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]]
```

```py
sharded_parameter_names(prefix: str = '') → Iterator[str]
```

```py
training: bool
```

```py
class torchrec.distributed.types.ShardingEnv(world_size: int, rank: int, pg: Optional[ProcessGroup] = None)
```

基类：`object`

提供了一个对 torch.distributed.ProcessGroup 的抽象，实际上使得 DistributedModelParallel 在推断期间可用。

```py
classmethod from_local(world_size: int, rank: int) → ShardingEnv
```

创建一个基于本地主机的分片环境。

注意

通常在单个主机推断期间使用。

```py
classmethod from_process_group(pg: ProcessGroup) → ShardingEnv
```

创建基于 ProcessGroup 的分片环境。

注意

通常在训练期间使用。

```py
class torchrec.distributed.types.ShardingPlan(plan: Dict[str, ModuleShardingPlan])
```

基类：`object`

分片计划的表示。这使用了较大包装模型的 FQN（即使用 DistributedModelParallel 包装的模型）。当需要 TorchRec 的可组合性时，应使用 EmbeddingModuleShardingPlan。

```py
plan
```

按模块路径为键的字典，按参数名称为键的参数分片规范字典。

类型：

Dictstr，[EmbeddingModuleShardingPlan]

```py
get_plan_for_module(module_path: str) → Optional[ModuleShardingPlan]
```

参数：

**module_path**（*str*）- 

返回：

按参数名称为键的参数分片规范字典。如果给定的 module_path 不存在分片规范，则为 None。

返回类型：

Optional[ModuleShardingPlan]

```py
plan: Dict[str, ModuleShardingPlan]
```

```py
class torchrec.distributed.types.ShardingPlanner
```

基类：`ABC`

计划分片。此计划可以保存并重复使用以确保分片稳定。

```py
abstract collective_plan(module: Module, sharders: List[ModuleSharder[Module]]) → ShardingPlan
```

在 rank 0 上调用 self.plan(…)并广播。

参数：

+   **module** (*nn.Module*) – 计划进行分片的模块。

+   **sharders** (*List***[*ModuleSharder**[**nn.Module**]**]*) – 提供的模块分片器。

返回：

计算的分片计划。

返回类型：

ShardingPlan

```py
abstract plan(module: Module, sharders: List[ModuleSharder[Module]]) → ShardingPlan
```

为提供的模块和给定的分片器制定分片计划。

参数：

+   **module** (*nn.Module*) – 计划进行分片的模块。

+   **sharders** (*List***[*ModuleSharder**[**nn.Module**]**]*) – 提供的模块分片器。

返回：

计算的分片计划。

返回类型：

ShardingPlan

```py
class torchrec.distributed.types.ShardingType(value)
```

基类：`Enum`

已知的分片类型，用于模块间优化。

```py
COLUMN_WISE = 'column_wise'
```

```py
DATA_PARALLEL = 'data_parallel'
```

```py
ROW_WISE = 'row_wise'
```

```py
TABLE_COLUMN_WISE = 'table_column_wise'
```

```py
TABLE_ROW_WISE = 'table_row_wise'
```

```py
TABLE_WISE = 'table_wise'
```

```py
torchrec.distributed.types.get_tensor_size_bytes(t: Tensor) → int
```

```py
torchrec.distributed.types.scope(method)
```  ## torchrec.distributed.utils

```py
class torchrec.distributed.utils.CopyableMixin(*args, **kwargs)
```

基类：`Module`

允许将模块复制到目标设备。

示例：

```py
class MyModule(CopyableMixin):
    ... 
```

参数：

**device** – 要复制到的 torch.device

返回

在新设备上的 nn.Module

```py
copy(device: device) → Module
```

```py
training: bool
```

```py
class torchrec.distributed.utils.PermutePooledEmbeddings(embs_dims: List[int], permute: List[int], device: Optional[device] = None)
```

基类：`object`

```py
torchrec.distributed.utils.add_params_from_parameter_sharding(fused_params: Optional[Dict[str, Any]], parameter_sharding: ParameterSharding) → Dict[str, Any]
```

从参数分片中提取参数，然后将它们添加到融合参数中。

如果参数分片中存在参数，则将覆盖融合参数中的参数。

参数：

+   **fused_params** (*Optional**[**Dict**[**str**,* *Any**]**]*) – 现有的融合参数

+   **parameter_sharding** (*ParameterSharding*) – 要使用的参数分片

返回：

包含从参数分片中添加的参数的融合参数字典。

返回类型：

[Dict[str, Any]]

```py
torchrec.distributed.utils.add_prefix_to_state_dict(state_dict: Dict[str, Any], prefix: str) → None
```

将状态字典中所有键添加前缀，原地操作。

参数：

+   **state_dict** (*Dict**[**str**,* *Any**]*) – 要更新的输入状态字典。

+   **prefix** (*str*) – 要从状态字典键中过滤的名称。

返回：

无。

```py
torchrec.distributed.utils.append_prefix(prefix: str, name: str) → str
```

将提供的前缀附加到提供的名称。

```py
torchrec.distributed.utils.convert_to_fbgemm_types(fused_params: Dict[str, Any]) → Dict[str, Any]
```

```py
torchrec.distributed.utils.copy_to_device(module: Module, current_device: device, to_device: device) → Module
```

```py
torchrec.distributed.utils.filter_state_dict(state_dict: OrderedDict[str, torch.Tensor], name: str) → OrderedDict[str, torch.Tensor]
```

过滤以提供的名称开头的状态字典键。从结果状态字典的键的开头剥离提供的名称。

参数：

+   **state_dict** (*OrderedDict**[**str**,* *torch.Tensor**]*) – 要过滤的输入状态字典。

+   **name** (*str*) – 要从状态字典键中过滤的名称。

返回：

过滤后的状态字典。

返回类型：

OrderedDict[str, torch.Tensor]

```py
torchrec.distributed.utils.get_unsharded_module_names(model: Module) → List[str]
```

检索不包含任何分片子模块的顶层模块的名称。

参数：

**model** (*torch.nn.Module*) – 从中检索未分片模块名称的模型。

返回：

不包含分片子模块的模块名称列表。

返回类型：

List[str]

```py
torchrec.distributed.utils.init_parameters(module: Module, device: device) → None
```

```py
torchrec.distributed.utils.merge_fused_params(fused_params: Optional[Dict[str, Any]] = None, param_fused_params: Optional[Dict[str, Any]] = None) → Dict[str, Any]
```

配置融合参数，包括 cache_precision，如果值未设置。

在 table_level_fused_params 中设置的值优先于全局融合参数

参数：

+   **fused_params** (*Optional**[**Dict**[**str**,* *Any**]**]*) – 原始融合参数

+   **grouped_fused_params** –

返回：

一个非空配置的融合参数字典，用于配置嵌入查找内核

返回类型：

[Dict[str, Any]]

```py
torchrec.distributed.utils.none_throws(optional: Optional[_T], message: str = 'Unexpected `None`') → _T
```

将可选项转换为其值。如果值为 None，则引发 AssertionError

```py
torchrec.distributed.utils.optimizer_type_to_emb_opt_type(optimizer_class: Type[Optimizer]) → Optional[EmbOptimType]
```

```py
class torchrec.distributed.utils.sharded_model_copy(device: Optional[Union[str, int, device]])
```

基类：`object`

允许将 DistributedModelParallel 模块复制到目标设备。

示例：

```py
# Copying model to CPU.

m = DistributedModelParallel(m)
with sharded_model_copy("cpu"):
    m_cpu = copy.deepcopy(m) 
```

## torchrec.distributed.mc_modules

```py
class torchrec.distributed.mc_modules.ManagedCollisionCollectionAwaitable(*args, **kwargs)
```

基类：`LazyAwaitable`[`KeyedJaggedTensor`]

```py
class torchrec.distributed.mc_modules.ManagedCollisionCollectionContext(sharding_contexts: List[torchrec.distributed.sharding.sequence_sharding.SequenceShardingContext] = <factory>, input_features: List[torchrec.sparse.jagged_tensor.KeyedJaggedTensor] = <factory>, reverse_indices: List[torch.Tensor] = <factory>)
```

基类：`EmbeddingCollectionContext`

```py
input_features: List[KeyedJaggedTensor]
```

```py
reverse_indices: List[Tensor]
```

```py
sharding_contexts: List[SequenceShardingContext]
```

```py
class torchrec.distributed.mc_modules.ManagedCollisionCollectionSharder(qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingSharder`[`ManagedCollisionCollection`]

```py
property module_type: Type[ManagedCollisionCollection]
```

```py
shard(module: ManagedCollisionCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, sharding_type_to_sharding: Dict[str, EmbeddingSharding[EmbeddingShardingContext, KeyedJaggedTensor, Tensor, Tensor]], device: Optional[device] = None) → ShardedManagedCollisionCollection
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module** (*M*) – 要分片的模块。

+   **params** (*EmbeddingModuleShardingPlan*) – 完全限定的参数名称字典（模块路径 + 参数名称，用‘.’分隔）及其分片规范。

+   **env** (*ShardingEnv*) – 具有进程组的分片环境。

+   **device** (*torch.device*) – 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
shardable_parameters(module: ManagedCollisionCollection) → Dict[str, Parameter]
```

可分片的参数列表。

```py
sharding_types(compute_device_type: str) → List[str]
```

支持的分片类型列表。查看 ShardingType 以获取常见示例。

```py
class torchrec.distributed.mc_modules.ShardedManagedCollisionCollection(module: ManagedCollisionCollection, table_name_to_parameter_sharding: Dict[str, ParameterSharding], env: ShardingEnv, device: device, sharding_type_to_sharding: Dict[str, EmbeddingSharding[EmbeddingShardingContext, KeyedJaggedTensor, Tensor, Tensor]], qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`ShardedModule`[`KJTList`, `KJTList`, `KeyedJaggedTensor`, `ManagedCollisionCollectionContext`]

```py
compute(ctx: ManagedCollisionCollectionContext, dist_input: KJTList) → KJTList
```

```py
create_context() → ManagedCollisionCollectionContext
```

```py
evict() → Dict[str, Optional[Tensor]]
```

```py
input_dist(ctx: ManagedCollisionCollectionContext, features: KeyedJaggedTensor) → Awaitable[Awaitable[KJTList]]
```

```py
output_dist(ctx: ManagedCollisionCollectionContext, output: KJTList) → LazyAwaitable[KeyedJaggedTensor]
```

```py
sharded_parameter_names(prefix: str = '') → Iterator[str]
```

```py
training: bool
```

```py
torchrec.distributed.mc_modules.create_mc_sharding(sharding_type: str, sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None) → EmbeddingSharding[SequenceShardingContext, KeyedJaggedTensor, Tensor, Tensor]
```

## torchrec.distributed.mc_embeddingbag

```py
class torchrec.distributed.mc_embeddingbag.ManagedCollisionEmbeddingBagCollectionContext(sharding_contexts: List[Union[torchrec.distributed.embedding_sharding.EmbeddingShardingContext, NoneType]] = <factory>, inverse_indices: Union[Tuple[List[str], torch.Tensor], NoneType] = None, variable_batch_per_feature: bool = False, evictions_per_table: Union[Dict[str, Union[torch.Tensor, NoneType]], NoneType] = None, remapped_kjt: Union[torchrec.distributed.embedding_types.KJTList, NoneType] = None)
```

基类：`EmbeddingBagCollectionContext`

```py
evictions_per_table: Optional[Dict[str, Optional[Tensor]]] = None
```

```py
record_stream(stream: Stream) → None
```

查看[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
remapped_kjt: Optional[KJTList] = None
```

```py
class torchrec.distributed.mc_embeddingbag.ManagedCollisionEmbeddingBagCollectionSharder(ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None, mc_sharder: Optional[ManagedCollisionCollectionSharder] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseManagedCollisionEmbeddingCollectionSharder`[`ManagedCollisionEmbeddingBagCollection`]

```py
property module_type: Type[ManagedCollisionEmbeddingBagCollection]
```

```py
shard(module: ManagedCollisionEmbeddingBagCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedManagedCollisionEmbeddingBagCollection
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module** (*M*) – 要分片的模块。

+   **params** (*EmbeddingModuleShardingPlan*) – 完全限定的参数名称字典（模块路径 + 参数名称，用‘.’分隔）及其分片规范。

+   **env** (*ShardingEnv*) – 具有进程组的分片环境。

+   **device** (*torch.device*) – 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
class torchrec.distributed.mc_embeddingbag.ShardedManagedCollisionEmbeddingBagCollection(module: ManagedCollisionEmbeddingBagCollection, table_name_to_parameter_sharding: Dict[str, ParameterSharding], ebc_sharder: EmbeddingBagCollectionSharder, mc_sharder: ManagedCollisionCollectionSharder, env: ShardingEnv, device: device)
```

基类：`BaseShardedManagedCollisionEmbeddingCollection`[`ManagedCollisionEmbeddingBagCollectionContext`]

```py
create_context() → ManagedCollisionEmbeddingBagCollectionContext
```

```py
training: bool
```

## torchrec.distributed.mc_embedding

```py
class torchrec.distributed.mc_embedding.ManagedCollisionEmbeddingCollectionContext(sharding_contexts: List[torchrec.distributed.sharding.sequence_sharding.SequenceShardingContext] = <factory>, input_features: List[torchrec.sparse.jagged_tensor.KeyedJaggedTensor] = <factory>, reverse_indices: List[torch.Tensor] = <factory>, evictions_per_table: Union[Dict[str, Union[torch.Tensor, NoneType]], NoneType] = None, remapped_kjt: Union[torchrec.distributed.embedding_types.KJTList, NoneType] = None)
```

基类：`EmbeddingCollectionContext`

```py
evictions_per_table: Optional[Dict[str, Optional[Tensor]]] = None
```

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
remapped_kjt: Optional[KJTList] = None
```

```py
class torchrec.distributed.mc_embedding.ManagedCollisionEmbeddingCollectionSharder(ec_sharder: Optional[EmbeddingCollectionSharder] = None, mc_sharder: Optional[ManagedCollisionCollectionSharder] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseManagedCollisionEmbeddingCollectionSharder`[`ManagedCollisionEmbeddingCollection`]

```py
property module_type: Type[ManagedCollisionEmbeddingCollection]
```

```py
shard(module: ManagedCollisionEmbeddingCollection, params: Dict[str, ParameterSharding], env: ShardingEnv, device: Optional[device] = None) → ShardedManagedCollisionEmbeddingCollection
```

执行实际的分片。它将根据相应的 ParameterSharding 在请求的位置上分配参数。

默认实现是数据并行复制。

参数：

+   **module** (*M*) – 要分片的模块。

+   **params** (*EmbeddingModuleShardingPlan*) – 完全限定的参数名称字典（模块路径+参数名称，用‘.’分隔）及其分片规范。

+   **env** (*ShardingEnv*) – 具有进程组的分片环境。

+   **device** (*torch.device*) – 计算设备。

返回：

分片模块实现。

返回类型：

ShardedModule[Any, Any, Any]

```py
class torchrec.distributed.mc_embedding.ShardedManagedCollisionEmbeddingCollection(module: ManagedCollisionEmbeddingCollection, table_name_to_parameter_sharding: Dict[str, ParameterSharding], ec_sharder: EmbeddingCollectionSharder, mc_sharder: ManagedCollisionCollectionSharder, env: ShardingEnv, device: device)
```

基类：`BaseShardedManagedCollisionEmbeddingCollection`[`ManagedCollisionEmbeddingCollectionContext`]

```py
create_context() → ManagedCollisionEmbeddingCollectionContext
```

```py
training: bool
```
