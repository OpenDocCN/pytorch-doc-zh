# torchrec.distributed.sharding

> 原文：[`pytorch.org/torchrec/torchrec.distributed.sharding.html`](https://pytorch.org/torchrec/torchrec.distributed.sharding.html)
>
> 译者：[飞龙](https://github.com/wizardforcel)
>
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


## torchrec.distributed.sharding.cw_sharding

```py
class torchrec.distributed.sharding.cw_sharding.BaseCwEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, permute_embeddings: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseTwEmbeddingSharding`[`C`, `F`, `T`, `W`]

列式分片的基类。

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
uncombined_embedding_dims() → List[int]
```

```py
uncombined_embedding_names() → List[str]
```

```py
class torchrec.distributed.sharding.cw_sharding.CwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, permute_embeddings: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseCwEmbeddingSharding`[`EmbeddingShardingContext`, `KeyedJaggedTensor`, `Tensor`, `Tensor`]

按列切分嵌入包，即。给定的嵌入表沿其列进行分区，并放置在指定的秩上。

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KeyedJaggedTensor]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[EmbeddingShardingContext, Tensor, Tensor]
```

```py
class torchrec.distributed.sharding.cw_sharding.InferCwPooledEmbeddingDist(device: device, world_size: int)
```

基类：`BaseEmbeddingDist`[`NullShardingContext`, `List`[`Tensor`], `Tensor`]

```py
forward(local_embs: List[Tensor], sharding_ctx: Optional[NullShardingContext] = None) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

尽管前向传递的方法需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.sharding.cw_sharding.InferCwPooledEmbeddingDistWithPermute(device: device, world_size: int, embedding_dims: List[int], permute: List[int])
```

基类：`BaseEmbeddingDist`[`NullShardingContext`, `List`[`Tensor`], `Tensor`]

```py
forward(local_embs: List[Tensor], sharding_ctx: Optional[NullShardingContext] = None) → Tensor
```

定义每次调用时执行的计算。

应该被所有子类重写。

注意

尽管前向传递的方法需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.sharding.cw_sharding.InferCwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, permute_embeddings: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseCwEmbeddingSharding`[`NullShardingContext`, `KJTList`, `List`[`Tensor`], `Tensor`]

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KJTList]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup[KJTList, List[Tensor]]
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[NullShardingContext, List[Tensor], Tensor]
```  ## torchrec.distributed.dist_data

```py
class torchrec.distributed.dist_data.EmbeddingsAllToOne(device: device, world_size: int, cat_dim: int)
```

基类：`Module`

将每个设备上的池化/序列嵌入张量合并为单个张量。

参数：

+   **device** (*torch.device*) – 将分配缓冲区的设备。

+   **world_size** (*int*) – 拓扑中的设备数量。

+   **cat_dim** (*int*) – 您希望在哪个维度上进行连接。对于池化嵌入，它是 1；对于序列嵌入，它是 0。

```py
forward(tensors: List[Tensor]) → Tensor
```

对池化/序列嵌入张量执行 AlltoOne 操作。

参数：

**tensors** (*List**[**torch.Tensor**]*) – 嵌入张量的列表。

返回：

合并嵌入的可等待对象。

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

将每个设备上的池化嵌入张量合并为单个张量。

参数：

+   **device**（*torch.device*）- 将分配缓冲区的设备。

+   **world_size**（*int*）- 拓扑中的设备数量。

```py
forward(tensors: List[Tensor]) → Tensor
```

使用 Reduce 对汇总嵌入张量执行 AlltoOne 操作。

参数：

**tensors**（*List**[**torch.Tensor**]*）- 嵌入张量的列表。

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

基础：`Module`

根据拆分将 KeyedJaggedTensor 重新分发到 ProcessGroup。

实现利用 torch.distributed 中的 AlltoAll 集体。

输入提供了必要的张量和输入拆分以进行分发。KJTAllToAllSplitsAwaitable 中的第一个集体调用将传输输出拆分（以为张量分配正确的空间）和每个等级的批量大小。KJTAllToAllTensorsAwaitable 中的后续集体调用将异步传输实际张量。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **splits**（*List**[**int**]*）- 长度为 pg.size()的列表，指示要发送到每个 pg.rank()的特征数量。假定 KeyedJaggedTensor 按目标等级排序。对所有等级都是相同的。

+   **stagger**（*int*）- 要应用于 recat 张量的间隔值，详细信息请参见 _get_recat 函数。

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

一个 KJTAllToAllTensorsAwaitable 的等待。

返回类型：

Awaitable[KJTAllToAllTensorsAwaitable]

```py
training: bool
```

```py
class torchrec.distributed.dist_data.KJTAllToAllSplitsAwaitable(pg: ProcessGroup, input: KeyedJaggedTensor, splits: List[int], labels: List[str], tensor_splits: List[List[int]], input_tensors: List[Tensor], keys: List[str], device: device, stagger: int)
```

基础：`Awaitable`[`KJTAllToAllTensorsAwaitable`]

KJT 张量拆分 AlltoAll 的等待。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **input**（*KeyedJaggedTensor*）- 输入 KJT。

+   **splits**（*List**[**int**]*）- 长度为 pg.size()的列表，指示要发送到每个 pg.rank()的特征数量。假定 KeyedJaggedTensor 按目标等级排序。对所有等级都是相同的。

+   **tensor_splits**（*Dict**[**str**,* *List**[**int**]**]*）- 输入 KJT 提供的张量拆分。

+   **input_tensors**（*List**[**torch.Tensor**]*）- 根据拆分提供的 KJT 张量（即长度、值）进行重新分发。

+   **keys**（*List**[**str**]*）- AlltoAll 后的 KJT 键。

+   **device**（*torch.device*）- 将分配缓冲区的设备。

+   **stagger**（*int*）- 要应用于 recat 张量的间隔值。

```py
class torchrec.distributed.dist_data.KJTAllToAllTensorsAwaitable(pg: ProcessGroup, input: KeyedJaggedTensor, splits: List[int], input_splits: List[List[int]], output_splits: List[List[int]], input_tensors: List[Tensor], labels: List[str], keys: List[str], device: device, stagger: int, stride_per_rank: Optional[List[int]])
```

基础：`Awaitable`[`KeyedJaggedTensor`]

KJT 张量 AlltoAll 的等待。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **input**（*KeyedJaggedTensor*）- 输入 KJT。

+   **splits**（*List**[**int**]**）- 每个 pg.rank() 发送多少特征的长度列表。假定 KeyedJaggedTensor 按目标排名排序。对所有排名都相同。

+   **input_splits**（*List**[**List**[**int**]**]*）- 每个 AlltoAll 中张量的输入拆分（每个排名将获得的值数量）。

+   **output_splits**（*List**[**List**[**int**]**]*）- 每个 AlltoAll 中张量的输出拆分（输出中每个排名的值数量）。

+   **input_tensors**（*List**[**torch.Tensor**]*）- 提供的 KJT 张量（即长度、值），根据拆分重新分配。

+   **labels**（*List**[**str**]*）- 每个提供的张量的标签。

+   **keys**（*List**[**str**]*）- AlltoAll 后的 KJT 键。

+   **device**（*torch.device*）- 将分配缓冲区的设备。

+   **stagger**（*int*）- 应用于 recat 张量的间隔值。

+   **stride_per_rank**（*可选**[**List**[**int**]**]*）- 在非变量批次特征情况下每个排名的步幅。

```py
class torchrec.distributed.dist_data.KJTOneToAll(splits: List[int], world_size: int, device: Optional[device] = None)
```

基类：`Module`

将 KeyedJaggedTensor 重新分配到所有设备。

实现利用 OnetoAll 函数，该函数本质上将特征复制到设备。

参数：

+   **splits**（*List**[**int**]**）- 将 KeyJaggedTensor 特征拆分成副本之前的特征长度。

+   **world_size**（*int*）- 拓扑中的设备数量。

+   **device**（*torch.device*）- 将分配 KJTs 的设备。

```py
forward(kjt: KeyedJaggedTensor) → KJTList
```

首先拆分特征，然后将切片发送到相应的设备。

参数：

**kjt**（*KeyedJaggedTensor*）- 输入特征。

返回：

KeyedJaggedTensor 拆分的可等待对象。

返回类型：

AwaitableList[[KeyedJaggedTensor]]

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAllGather(pg: ProcessGroup, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

包装池化嵌入通信的全收集通信原语的模块类。

提供具有 [batch_size, dimension] 布局的本地输入张量，我们希望从所有排名收集输入张量到一个扁平化的输出张量中。

该类返回池化嵌入张量的异步可等待句柄。全收集仅适用于 NCCL 后端。

参数：

+   **pg**（*dist.ProcessGroup*）- 发生全收集通信的进程组。

+   **codecs**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

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

在池化嵌入张量上执行减少散布操作。

参数：

**local_emb**（*torch.Tensor*）- 形状为 [num_buckets x batch_size, dimension] 的张量。

返回：

形状为 [batch_size, dimension] 的张量的池化嵌入的可等待对象。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAllToAll(pg: ProcessGroup, dim_sum_per_rank: List[int], device: Optional[device] = None, callbacks: Optional[List[Callable[[Tensor], Tensor]]] = None, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

根据 dim_sum_per_rank 使用 ProcessGroup 对张量进行分片和收集键。

实现利用 alltoall_pooled 操作。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于全收集通信的 ProcessGroup。

+   **dim_sum_per_rank**（*List**[**int**]*）- 每个排名中嵌入的特征数量（维度之和）。

+   **device**（*可选**[**torch.device**]*）- 将分配缓冲区的设备。

+   **callbacks**（*可选**[**List**[**Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]**]*）- 回调函数。

+   **codecs**（*可选***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

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

在池化嵌入张量上执行全收集操作。

参数：

+   **local_embs**（*torch.Tensor*）- 要分发的值的张量。

+   **batch_size_per_rank**（*可选**[**List**[**int**]**]*）- 每个排名的批量大小，以支持可变批量大小。

返回：

汇总嵌入的可等待。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsAwaitable(tensor_awaitable: Awaitable[Tensor])
```

基类：`Awaitable`[`Tensor`]

集体操作后的汇总嵌入的可等待。

参数：

**tensor_awaitable**（*Awaitable**[**torch.Tensor**]*）- 集体后来自组中所有进程的张量的连接张量的可等待。

```py
property callbacks: List[Callable[[Tensor], Tensor]]
```

```py
class torchrec.distributed.dist_data.PooledEmbeddingsReduceScatter(pg: ProcessGroup, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

包装了用于行级和 twrw 分片中的汇总嵌入通信的 reduce-scatter 通信原语的模块类。

对于汇总嵌入，我们有一个本地模型并行输出张量，布局为[num_buckets x batch_size，维度]。我们需要跨批次对 num_buckets 维度求和。我们根据 input_splits 将张量沿第一维拆分为不均匀的块（不同桶的张量切片），将它们减少到输出张量并将结果分散到相应的排名。

该类返回汇总嵌入张量的异步 Awaitable 句柄。reduce-scatter-v 操作仅适用于 NCCL 后端。

参数：

+   **pg**（*dist.ProcessGroup*）- 减少散列通信发生在其中的进程组。

+   **codecs** - 量化通信编解码器。

```py
forward(local_embs: Tensor, input_splits: Optional[List[int]] = None) → PooledEmbeddingsAwaitable
```

在汇总嵌入张量上执行减少散列操作。

参数：

+   **local_embs**（*torch.Tensor*）- 形状为[num_buckets * batch_size，维度]的张量。

+   **input_splits**（*可选**[**List**[**int**]**]*）- 用于 local_embs 维度 0 的拆分列表。

返回：

张量的形状为[batch_size，维度]的汇总嵌入的可等待。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```

```py
class torchrec.distributed.dist_data.SeqEmbeddingsAllToOne(device: device, world_size: int)
```

基类：`Module`

将每个设备上的汇总/序列嵌入张量合并为单个张量。

参数：

+   **device**（*torch.device*）- 将分配缓冲区的设备

+   **world_size**（*int*）- 拓扑中的设备数量。

+   **cat_dim**（*int*）- 您希望在其上连接的维度。对于汇总嵌入，它是 1；对于序列嵌入，它是 0。

```py
forward(tensors: List[Tensor]) → List[Tensor]
```

在汇总嵌入张量上执行 AlltoOne 操作。

参数：

**tensors**（*List**[**torch.Tensor**]*）- 汇总嵌入张量的列表。

返回：

合并汇总嵌入的可等待。

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

根据分片将序列嵌入重新分配到 ProcessGroup。

参数：

+   **pg**（*dist.ProcessGroup*）- AlltoAll 通信发生在其中的进程组。

+   **features_per_rank**（*List**[**int**]*）- 每个排名的特征数量列表。

+   **device**（*可选**[**torch.device**]*）- 将分配缓冲区的设备。

+   **codecs**（*可选***[*QuantizedCommCodecs**]*) - 量化通信编解码器。

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

+   **input_splits**（*List**[**int**]*）- AlltoAll 的输入分片。

+   **output_splits**（*List**[**int**]*）- AlltoAll 的输出分片。

+   **unbucketize_permute_tensor**（*可选**[**torch.Tensor**]*）- 存储 KJT bucketize 的排列顺序（仅适用于行级分片）。

+   **batch_size_per_rank** - （可选[List[int]]）：每个 rank 的批量大小。

+   **sparse_features_recat**（*Optional**[**torch.Tensor**]*）- 用于稀疏特征输入分布的 recat 张量。如果使用可变批量大小，则必须提供。

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

在集体操作后的序列嵌入之后的可等待对象。

参数：

+   **tensor_awaitable**（*Awaitable**[**torch.Tensor**]*) - 集体操作后来自组内所有进程的连接张量的可等待对象。

+   **unbucketize_permute_tensor**（*Optional**[**torch.Tensor**]*）- 存储 KJT 桶化的排列顺序（仅适用于逐行分片）。

+   **embedding_dim**（*int*）- 嵌入维度。

```py
class torchrec.distributed.dist_data.SplitsAllToAllAwaitable(input_tensors: List[Tensor], pg: ProcessGroup)
```

基类：`Awaitable`[`List`[`List`[`int`]]]

拆分 AlltoAll 的可等待对象。

参数：

+   **input_tensors**（*List**[**torch.Tensor**]*）- 要重新分配的拆分张量。

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

```py
class torchrec.distributed.dist_data.VariableBatchPooledEmbeddingsAllToAll(pg: ProcessGroup, emb_dim_per_rank_per_feature: List[List[int]], device: Optional[device] = None, callbacks: Optional[List[Callable[[Tensor], Tensor]]] = None, codecs: Optional[QuantizedCommCodecs] = None)
```

基类：`Module`

根据 dim_sum_per_rank 对张量的批次进行分片并收集键与 ProcessGroup 一起。

实现利用 variable_batch_alltoall_pooled 操作。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **emb_dim_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 每个特征的每个 rank 的嵌入维度。

+   **device**（*Optional**[**torch.device**]*）- 将分配缓冲区的设备。

+   **callbacks**（*Optional**[**List**[**Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]**]*）- 回调函数。

+   **codecs**（*Optional***[*QuantizedCommCodecs**]*）- 量化通信编解码器。

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

对池化嵌入张量进行具有每个特征可变批量大小的 AlltoAll 池化操作。

参数：

+   **local_embs**（*torch.Tensor*）- 要分发的值的张量。

+   **batch_size_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 每个特征的每个 rank 的批量大小，a2a 后。用于获取输入拆分。

+   **batch_size_per_feature_pre_a2a**（*List**[**int**]*）- 分散之前的本地批量大小，用于获取输出拆分。按 rank_0 特征，rank_1 特征排序，...

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

包装可变批量池化嵌入通信的 reduce-scatter 通信原语的模块类，rw 和 twrw 分片。

对于每个特征池化嵌入的可变批量，我们有一个本地模型并行输出张量，其布局为每个特征的每个 rank 的批量大小总和乘以相应的嵌入维度的 1d 布局[batch_size_r0_f0 * emb_dim_f0 + …)]。我们根据 batch_size_per_rank_per_feature 和相应的 embedding_dims 将张量分割成不均匀的块，并将它们减少到输出张量并将结果分散到相应的 rank。

该类返回用于池化嵌入张量的异步 Awaitable 句柄。reduce-scatter-v 操作仅适用于 NCCL 后端。

参数：

+   **pg**（*dist.ProcessGroup*）- reduce-scatter 通信发生在其中的进程组。

+   **codecs** - 量化通信编解码器。

```py
forward(local_embs: Tensor, batch_size_per_rank_per_feature: List[List[int]], embedding_dims: List[int]) → PooledEmbeddingsAwaitable
```

对池化嵌入张量执行 reduce scatter 操作。

参数：

+   **local_embs** (*torch.Tensor*) - 形状为[num_buckets * batch_size, dimension]的张量。

+   **batch_size_per_rank_per_feature** (*List**[**List**[**int**]**]*) - 每个特征的每个等级的批量大小，用于确定输入拆分。

+   **embedding_dims** (*List**[**int**]*) - 每个特征的嵌入维度，用于确定输入拆分。

返回：

等待池化张量的嵌入，形状为[batch_size, dimension]。

返回类型：

PooledEmbeddingsAwaitable

```py
training: bool
```  ## torchrec.distributed.sharding.dp_sharding

```py
class torchrec.distributed.sharding.dp_sharding.BaseDpEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None)
```

基类：`EmbeddingSharding`[`C`, `F`, `T`, `W`]

基类用于数据并行分片。

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
embedding_names_per_rank() → List[List[str]]
```

```py
embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
embedding_tables() → List[ShardedEmbeddingTable]
```

```py
feature_names() → List[str]
```

```py
class torchrec.distributed.sharding.dp_sharding.DpPooledEmbeddingDist
```

基类：`BaseEmbeddingDist`[`EmbeddingShardingContext`, `Tensor`, `Tensor`]

将池化嵌入分发为数据并行。

```py
forward(local_embs: Tensor, sharding_ctx: Optional[EmbeddingShardingContext] = None) → Awaitable[Tensor]
```

由于池化嵌入已经以数据并行方式分布，因此无操作。

参数：

**local_embs** (*torch.Tensor*) - 输出序列嵌入。

返回：

等待池化嵌入张量。

返回类型：

Awaitable[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.dp_sharding.DpPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None)
```

基类：`BaseDpEmbeddingSharding`[`EmbeddingShardingContext`, `KeyedJaggedTensor`, `Tensor`, `Tensor`]

将嵌入包数据并行分片，没有表分片，即给定的嵌入表在所有等级上都复制。

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KeyedJaggedTensor]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[EmbeddingShardingContext, Tensor, Tensor]
```

```py
class torchrec.distributed.sharding.dp_sharding.DpSparseFeaturesDist
```

基类：`BaseSparseFeaturesDist`[`KeyedJaggedTensor`]

将稀疏特征（输入）分发为数据并行。

```py
forward(sparse_features: KeyedJaggedTensor) → Awaitable[Awaitable[KeyedJaggedTensor]]
```

由于稀疏特征已经以数据并行方式分布，因此无操作。

参数：

**sparse_features** (*SparseFeatures*) - 输入稀疏特征。

返回：

等待稀疏特征的等待。

返回类型：

Awaitable[Awaitable[SparseFeatures]]

```py
training: bool
```  ## torchrec.distributed.sharding.rw_sharding

```py
class torchrec.distributed.sharding.rw_sharding.BaseRwEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, need_pos: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`EmbeddingSharding`[`C`, `F`, `T`, `W`]

基类用于按行分片。

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
embedding_names_per_rank() → List[List[str]]
```

```py
embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
embedding_tables() → List[ShardedEmbeddingTable]
```

```py
feature_names() → List[str]
```

```py
class torchrec.distributed.sharding.rw_sharding.InferRwPooledEmbeddingDist(device: device, world_size: int)
```

基类：`BaseEmbeddingDist`[`NullShardingContext`, `List`[`Tensor`], `Tensor`]

以 AlltoOne 操作以 RW 方式重新分配汇集的嵌入张量。

参数：

+   **device** (*torch.device*) – 将要通信的张量所在的设备。

+   **world_size** (*int*) – 拓扑中的设备数量。

```py
forward(local_embs: List[Tensor], sharding_ctx: Optional[NullShardingContext] = None) → Tensor
```

在序列嵌入张量上执行 AlltoOne 操作。

参数：

**local_embs** (*torch.Tensor*) – 要分发的值的张量。

返回：

序列嵌入的 awaitable。

返回类型：

Awaitable[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.rw_sharding.InferRwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, need_pos: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseRwEmbeddingSharding`[`NullShardingContext`, `KJTList`, `List`[`Tensor`], `Tensor`]

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KJTList]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup[KJTList, List[Tensor]]
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[NullShardingContext, List[Tensor], Tensor]
```

```py
class torchrec.distributed.sharding.rw_sharding.InferRwSparseFeaturesDist(world_size: int, num_features: int, feature_hash_sizes: List[int], device: Optional[device] = None, is_sequence: bool = False, has_feature_processor: bool = False, need_pos: bool = False, embedding_shard_metadata: Optional[List[List[int]]] = None)
```

基类：`BaseSparseFeaturesDist`[`KJTList`]

```py
forward(sparse_features: KeyedJaggedTensor) → KJTList
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

尽管前向传递的步骤需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此处调用，因为前者会负责运行注册的钩子，而后者会默默忽略它们。

```py
training: bool
```

```py
class torchrec.distributed.sharding.rw_sharding.RwPooledEmbeddingDist(pg: ProcessGroup, embedding_dims: List[int], qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingDist`[`EmbeddingShardingContext`, `Tensor`, `Tensor`]

以 RW 方式执行 reduce-scatter 操作重新分配汇集的嵌入张量。

参数：

**pg** (*dist.ProcessGroup*) – 用于 reduce-scatter 通信的 ProcessGroup。

```py
forward(local_embs: Tensor, sharding_ctx: Optional[EmbeddingShardingContext] = None) → Awaitable[Tensor]
```

在汇集的嵌入张量上执行 reduce-scatter 池化操作。

参数：

+   **local_embs** (*torch.Tensor*) – 要分发的汇集的嵌入张量。

+   **sharding_ctx** (*Optional***[*EmbeddingShardingContext**]*) – 来自 KJTAllToAll 操作的共享上下文。

返回：

汇集的嵌入张量的 awaitable。

返回类型：

Awaitable[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.rw_sharding.RwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, need_pos: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseRwEmbeddingSharding`[`EmbeddingShardingContext`, `KeyedJaggedTensor`, `Tensor`, `Tensor`]

按行分片嵌入包，即。给定的嵌入表按行均匀分布，表切片放置在所有秩上。

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KeyedJaggedTensor]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[EmbeddingShardingContext, Tensor, Tensor]
```

```py
class torchrec.distributed.sharding.rw_sharding.RwSparseFeaturesDist(pg: ProcessGroup, num_features: int, feature_hash_sizes: List[int], device: Optional[device] = None, is_sequence: bool = False, has_feature_processor: bool = False, need_pos: bool = False)
```

基类：`BaseSparseFeaturesDist`[`KeyedJaggedTensor`]

以 RW 方式对稀疏特征进行分桶，然后通过 AlltoAll 集体操作重新分配。

参数：

+   **pg** (*dist.ProcessGroup*) - 用于 AlltoAll 通信的 ProcessGroup。

+   **intra_pg** (*dist.ProcessGroup*) - 单个主机组内用于 AlltoAll 通信的 ProcessGroup。

+   **num_features** (*int*) - 特征总数。

+   **feature_hash_sizes** (*List**[**int**]*) - 特征的哈希大小。

+   **device** (*Optional**[**torch.device**]*) - 将分配缓冲区的设备。

+   **is_sequence** (*bool*) - 如果这是用于序列嵌入。

+   **has_feature_processor** (*bool*) - 特征处理器的存在（即位置加权特征）。

```py
forward(sparse_features: KeyedJaggedTensor) → Awaitable[Awaitable[KeyedJaggedTensor]]
```

将稀疏特征值分桶为拓扑中设备数量的桶，然后执行 AlltoAll 操作。

参数：

**sparse_features** (*KeyedJaggedTensor*) - 要分桶和重新分配的稀疏特征。

返回：

可等待的可等待的 KeyedJaggedTensor。

返回类型：

Awaitable[Awaitable[KeyedJaggedTensor]]

```py
training: bool
```

```py
torchrec.distributed.sharding.rw_sharding.get_block_sizes_runtime_device(block_sizes: List[int], runtime_device: device, tensor_cache: Dict[str, Tuple[Tensor, List[Tensor]]], embedding_shard_metadata: Optional[List[List[int]]] = None, dtype: dtype = torch.int32) → Tuple[Tensor, List[Tensor]]
```

```py
torchrec.distributed.sharding.rw_sharding.get_embedding_shard_metadata(grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]]) → Tuple[List[List[int]], bool]
```  ## torchrec.distributed.sharding.tw_sharding

```py
class torchrec.distributed.sharding.tw_sharding.BaseTwEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`EmbeddingSharding`[`C`, `F`, `T`, `W`]

表格智能分片的基类。

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
embedding_names_per_rank() → List[List[str]]
```

```py
embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
embedding_tables() → List[ShardedEmbeddingTable]
```

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
class torchrec.distributed.sharding.tw_sharding.InferTwEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseTwEmbeddingSharding`[`NullShardingContext`, `KJTList`, `List`[`Tensor`], `Tensor`]

为推断分片嵌入包表格

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KJTList]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup[KJTList, List[Tensor]]
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[NullShardingContext, List[Tensor], Tensor]
```

```py
class torchrec.distributed.sharding.tw_sharding.InferTwPooledEmbeddingDist(device: device, world_size: int)
```

基类：`BaseEmbeddingDist`[`NullShardingContext`, `List`[`Tensor`], `Tensor`]

合并每个设备的汇总嵌入张量以进行推断。

参数：

+   **device** (*Optional**[**torch.device**]*) - 将分配缓冲区的设备。

+   **world_size** (*int*) - 拓扑中设备的数量。

```py
forward(local_embs: List[Tensor], sharding_ctx: Optional[NullShardingContext] = None) → Tensor
```

对汇总嵌入张量执行 AlltoOne 操作。

参数：

**local_embs** (*List**[**torch.Tensor**]*) - 具有 len(local_embs) == world_size 的汇总嵌入张量。

返回：

可等待的合并汇总嵌入张量。

返回类型：

Awaitable[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.tw_sharding.InferTwSparseFeaturesDist(features_per_rank: List[int], world_size: int, device: Optional[device] = None)
```

基类：`BaseSparseFeaturesDist`[`KJTList`]

将稀疏特征重新分配到所有设备进行推断。

参数：

+   **features_per_rank** (*List**[**int**]*) - 发送到每个等级的特征数。

+   **world_size** (*int*) - 拓扑中设备的数量。

+   **fused_params** (*Dict**[**str**,* *Any**]*) - 模型的融合参数。

```py
forward(sparse_features: KeyedJaggedTensor) → KJTList
```

对稀疏特征执行 OnetoAll 操作。

参数：

**sparse_features**（*KeyedJaggedTensor*）- 要重新分配的稀疏特征。

返回：

可等待的 KeyedJaggedTensor 的可等待。

返回类型：

Awaitable[Awaitable[KeyedJaggedTensor]]

```py
training: bool
```

```py
class torchrec.distributed.sharding.tw_sharding.TwPooledEmbeddingDist(pg: ProcessGroup, dim_sum_per_rank: List[int], emb_dim_per_rank_per_feature: List[List[int]], device: Optional[device] = None, callbacks: Optional[List[Callable[[Tensor], Tensor]]] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingDist`[`EmbeddingShardingContext`, `Tensor`, `Tensor`]

使用 AlltoAll 集体操作重新分配池化的嵌入张量，以进行表格划分。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **dim_sum_per_rank**（*List**[**int**]*）- 每个 rank 中嵌入的特征数量（维度之和）。

+   **emb_dim_per_rank_per_feature**（*List**[**List**[**int**]**]*）- 每个特征的每个 rank 的嵌入维度，用于每个特征的可变批处理。

+   **device**（*Optional**[**torch.device**]*）- 将分配缓冲区的设备。

+   **callbacks**（*Optional**[**List**[**Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]**]*）-

+   **qcomm_codecs_registry**（*Optional****Dict**[**str**,* [*QuantizedCommCodecs**]**]*）-

```py
forward(local_embs: Tensor, sharding_ctx: Optional[EmbeddingShardingContext] = None) → Awaitable[Tensor]
```

对池化的嵌入张量执行 AlltoAll 操作。

参数：

+   **local_embs**（*torch.Tensor*）- 要分发的值的张量。

+   **sharding_ctx**（*Optional***[*EmbeddingShardingContext**]*）- 来自 KJTAllToAll 操作的共享上下文。

返回：

池化的嵌入的可等待。

返回类型：

Awaitable[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.tw_sharding.TwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseTwEmbeddingSharding`[`EmbeddingShardingContext`, `KeyedJaggedTensor`, `Tensor`, `Tensor`]

按表格划分嵌入包，即。给定的嵌入表完全放置在选定的 rank 上。

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KeyedJaggedTensor]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[EmbeddingShardingContext, Tensor, Tensor]
```

```py
class torchrec.distributed.sharding.tw_sharding.TwSparseFeaturesDist(pg: ProcessGroup, features_per_rank: List[int])
```

基类：`BaseSparseFeaturesDist`[`KeyedJaggedTensor`]

使用 AlltoAll 集体操作重新分配稀疏特征，以进行表格划分。

参数：

+   **pg**（*dist.ProcessGroup*）- 用于 AlltoAll 通信的 ProcessGroup。

+   **features_per_rank**（*List**[**int**]*）- 发送到每个 rank 的特征数量。

```py
forward(sparse_features: KeyedJaggedTensor) → Awaitable[Awaitable[KeyedJaggedTensor]]
```

对稀疏特征执行 AlltoAll 操作。

参数：

**sparse_features**（*KeyedJaggedTensor*）- 要重新分配的稀疏特征。

返回：

可等待的 KeyedJaggedTensor 的可等待。

返回类型：

可等待对象[可等待对象[KeyedJaggedTensor]]

```py
training: bool
```  ## torchrec.distributed.sharding.twcw_sharding

```py
class torchrec.distributed.sharding.twcw_sharding.TwCwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, permute_embeddings: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`CwPooledEmbeddingSharding`

按表格方式分片嵌入包，即给定的嵌入表按列进行分区，并将表切片放置在主机组内的所有秩上。  ## torchrec.distributed.sharding.twrw_sharding

```py
class torchrec.distributed.sharding.twrw_sharding.BaseTwRwEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, need_pos: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`EmbeddingSharding`[`C`, `F`, `T`, `W`]

表格智能行智能分片的基类。

```py
embedding_dims() → List[int]
```

```py
embedding_names() → List[str]
```

```py
embedding_names_per_rank() → List[List[str]]
```

```py
embedding_shard_metadata() → List[Optional[ShardMetadata]]
```

```py
feature_names() → List[str]
```

```py
class torchrec.distributed.sharding.twrw_sharding.TwRwPooledEmbeddingDist(rank: int, cross_pg: ProcessGroup, intra_pg: ProcessGroup, dim_sum_per_node: List[int], emb_dim_per_node_per_feature: List[List[int]], device: Optional[device] = None, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseEmbeddingDist`[`EmbeddingShardingContext`, `Tensor`, `Tensor`]

通过在主机级别逐行执行 reduce-scatter 操作，然后在全局级别逐表执行全对全操作，以 TWRW 方式重新分配池化嵌入张量。

参数：

+   **cross_pg** (*dist.ProcessGroup*) – 用于全对全通信的全局级 ProcessGroup。

+   **intra_pg** (*dist.ProcessGroup*) – 用于 reduce-scatter 通信的主机级 ProcessGroup。

+   **dim_sum_per_node** (*列表**[**int**]*) – 每个主机的嵌入特征的数量（维度之和）。

+   **emb_dim_per_node_per_feature** (*列表**[**列表**[**int**]**]*) –

+   **device** (*可选**[**torch.device**]*) – 将分配缓冲区的设备。

+   **qcomm_codecs_registry** (*可选****Dict**[**str**,* [*QuantizedCommCodecs**]**]*) –

```py
forward(local_embs: Tensor, sharding_ctx: Optional[EmbeddingShardingContext] = None) → Awaitable[Tensor]
```

对池化嵌入张量执行 reduce-scatter 池化操作，然后进行全对全池化操作。

参数：

**local_embs** (*torch.Tensor*) – 要分发的池化嵌入张量。

返回：

池化嵌入张量的可等待对象。

返回类型：

可等待对象[torch.Tensor]

```py
training: bool
```

```py
class torchrec.distributed.sharding.twrw_sharding.TwRwPooledEmbeddingSharding(sharding_infos: List[EmbeddingShardingInfo], env: ShardingEnv, device: Optional[device] = None, need_pos: bool = False, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None)
```

基类：`BaseTwRwEmbeddingSharding`[`EmbeddingShardingContext`, `KeyedJaggedTensor`, `Tensor`, `Tensor`]

按表格方式分片嵌入包，然后按行方式分片。

```py
create_input_dist(device: Optional[device] = None) → BaseSparseFeaturesDist[KeyedJaggedTensor]
```

```py
create_lookup(device: Optional[device] = None, fused_params: Optional[Dict[str, Any]] = None, feature_processor: Optional[BaseGroupedFeatureProcessor] = None) → BaseEmbeddingLookup
```

```py
create_output_dist(device: Optional[device] = None) → BaseEmbeddingDist[EmbeddingShardingContext, Tensor, Tensor]
```

```py
class torchrec.distributed.sharding.twrw_sharding.TwRwSparseFeaturesDist(pg: ProcessGroup, local_size: int, features_per_rank: List[int], feature_hash_sizes: List[int], device: Optional[device] = None, has_feature_processor: bool = False, need_pos: bool = False)
```

基类：`BaseSparseFeaturesDist`[`KeyedJaggedTensor`]

以 TWRW 方式对稀疏特征进行分桶，然后通过全对全集体操作重新分配。

参数：

+   **pg** (*dist.ProcessGroup*) – 用于全对全通信的 ProcessGroup。

+   **intra_pg** (*dist.ProcessGroup*) – 单个主机组内用于 AlltoAll 通信的 ProcessGroup。

+   **id_list_features_per_rank** (*List**[**int**]*) – 发送到每个排名的 id 列表特征的数量。

+   **id_score_list_features_per_rank** (*List**[**int**]*) – 发送到每个排名的 id 分数列表特征的数量。

+   **id_list_feature_hash_sizes** (*List**[**int**]*) – id 列表特征的哈希大小。

+   **id_score_list_feature_hash_sizes** (*List**[**int**]*) – id 分数列表特征的哈希大小。

+   **device** (*Optional**[**torch.device**]*) – 将分配缓冲区的设备。

+   **has_feature_processor** (*bool*) – 特征处理器的存在（即位置加权特征）。

示例：

```py
3 features
2 hosts with 2 devices each

Bucketize each feature into 2 buckets
Staggered shuffle with feature splits [2, 1]
AlltoAll operation

NOTE: result of staggered shuffle and AlltoAll operation look the same after
reordering in AlltoAll

Result:
    host 0 device 0:
        feature 0 bucket 0
        feature 1 bucket 0

    host 0 device 1:
        feature 0 bucket 1
        feature 1 bucket 1

    host 1 device 0:
        feature 2 bucket 0

    host 1 device 1:
        feature 2 bucket 1 
```

```py
forward(sparse_features: KeyedJaggedTensor) → Awaitable[Awaitable[KeyedJaggedTensor]]
```

将稀疏特征值分桶为本地世界大小的桶数，对稀疏特征执行交错洗牌，然后执行 AlltoAll 操作。

参数：

**sparse_features** (*KeyedJaggedTensor*) – 要进行分桶和重新分配的稀疏特征。

返回：

KeyedJaggedTensor 的可等待对象。

返回类型：

Awaitable[KeyedJaggedTensor]

```py
training: bool
```
