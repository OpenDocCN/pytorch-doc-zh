# torchrec.sparse

> 原文：[`pytorch.org/torchrec/torchrec.sparse.html`](https://pytorch.org/torchrec/torchrec.sparse.html)
>
> 译者：[飞龙](https://github.com/wizardforcel)
>
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


Torchrec Jagged Tensors

它有 3 个类：JaggedTensor，KeyedJaggedTensor，KeyedTensor。

JaggedTensor

它表示一个（可选加权）不规则张量。JaggedTensor 是一个具有不规则维度的张量，其切片可能具有不同的长度。请参考 KeyedJaggedTensor docstring 以获取完整示例和更多信息。

KeyedJaggedTensor

KeyedJaggedTensor 具有额外的“Key”信息。以第一维为键，最后一维为不规则。请参考 KeyedJaggedTensor docstring 以获取完整示例和更多信息。

KeyedTensor

KeyedTensor 保存了一个可以通过键访问的密集张量列表，每个张量可以通过键访问。键维度可以是可变长度（每个键的长度）。常见用例包括存储不同维度的池化嵌入。请参考 KeyedTensor docstring 以获取完整示例和更多信息。

## torchrec.sparse.jagged_tensor

```py
class torchrec.sparse.jagged_tensor.ComputeJTDictToKJT(*args, **kwargs)
```

基类：`Module`

将 JaggedTensors 的字典转换为 KeyedJaggedTensor。参数：

示例：传入 jt_dict

> {
> 
> “Feature0”：JaggedTensor([[V0,V1],None,V2]），“Feature1”：JaggedTensor([V3,V4,[V5,V6,V7]]），
> 
> }

返回：带有内容的 kjt：# 0 1 2 <– dim_1 # “Feature0” [V0,V1] None [V2] # “Feature1” [V3] [V4] [V5,V6,V7] # ^ # dim_0

```py
forward(jt_dict: Dict[str, JaggedTensor]) → KeyedJaggedTensor
```

参数：

**jt_dict** – 一个 JaggedTensor 的字典

返回：

KeyedJaggedTensor

```py
training: bool
```

```py
class torchrec.sparse.jagged_tensor.ComputeKJTToJTDict(*args, **kwargs)
```

基类：`Module`

将 KeyedJaggedTensor 转换为 JaggedTensors 的字典。

参数：

示例：

# 0 1 2 <– dim_1 # “Feature0” [V0,V1] None [V2] # “Feature1” [V3] [V4] [V5,V6,V7] # ^ # dim_0

将返回

{

“Feature0”：JaggedTensor([[V0,V1],None,V2]），“Feature1”：JaggedTensor([V3,V4,[V5,V6,V7]]），

}

```py
forward(keyed_jagged_tensor: KeyedJaggedTensor) → Dict[str, JaggedTensor]
```

将 KeyedJaggedTensor 转换为 JaggedTensors 的字典。

参数：

**keyed_jagged_tensor** (*KeyedJaggedTensor*) – 要转换的张量

返回：

Dict[str, JaggedTensor]

```py
training: bool
```

```py
class torchrec.sparse.jagged_tensor.JaggedTensor(*args, **kwargs)
```

基类：`Pipelineable`

表示一个（可选加权）不规则张量。

JaggedTensor 是一个具有*不规则维度*的张量，其切片可能具有不同的长度。请参考 KeyedJaggedTensor 以获取完整示例。

实现可以进行 torch.jit.script。

注意

我们不会进行输入验证，因为这很昂贵，您应该始终传入有效的长度、偏移等。

参数：

+   **values**（*torch.Tensor*）– 密集表示中的值张量。

+   **weights**（*可选**[**torch.Tensor**]）– 如果值有权重。形状与值相同的张量。

+   **lengths**（*可选**[**torch.Tensor**]）– 切片，表示为长度。

+   **offsets**（*可选**[**torch.Tensor**]）– 切片，表示为累积偏移。

```py
static empty(is_weighted: bool = False, device: Optional[device] = None, values_dtype: Optional[dtype] = None, weights_dtype: Optional[dtype] = None, lengths_dtype: dtype = torch.int32) → JaggedTensor
```

```py
static from_dense(values: List[Tensor], weights: Optional[List[Tensor]] = None) → JaggedTensor
```

从形状为(B, N,)的密集值/权重构建 JaggedTensor。

请注意，长度和偏移仍然是形状为(B,)的。

参数：

+   **values**（*List**[**torch.Tensor**]）– 用于密集表示的张量列表

+   **weights**（*可选**[**List**[**torch.Tensor**]**]）– 如果值有权重，形状与值相同的张量。

返回：

从 2D 密集张量创建的 JaggedTensor。

返回类型：

JaggedTensor

示例：

```py
values = [
    torch.Tensor([1.0]),
    torch.Tensor(),
    torch.Tensor([7.0, 8.0]),
    torch.Tensor([10.0, 11.0, 12.0]),
]
weights = [
    torch.Tensor([1.0]),
    torch.Tensor(),
    torch.Tensor([7.0, 8.0]),
    torch.Tensor([10.0, 11.0, 12.0]),
]
j1 = JaggedTensor.from_dense(
    values=values,
    weights=weights,
)

# j1 = [[1.0], [], [7.0], [8.0], [10.0, 11.0, 12.0]] 
```

```py
static from_dense_lengths(values: Tensor, lengths: Tensor, weights: Optional[Tensor] = None) → JaggedTensor
```

从形状为(B, N,)的密集值/权重构建 JaggedTensor。

请注意，长度仍然是形状为(B,)的。

```py
lengths() → Tensor
```

```py
lengths_or_none() → Optional[Tensor]
```

```py
offsets() → Tensor
```

```py
offsets_or_none() → Optional[Tensor]
```

```py
record_stream(stream: Stream) → None
```

参见[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
to(device: device, non_blocking: bool = False) → JaggedTensor
```

请注意，根据[`pytorch.org/docs/stable/generated/torch.Tensor.to.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)，to 可能返回 self 或 self 的副本。因此，请记住要使用赋值运算符 to，例如，in = in.to(new_device)。

```py
to_dense() → List[Tensor]
```

构建 JT 值的密集表示。

返回：

张量列表。

返回类型：

列表[torch.Tensor]

示例：

```py
values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
jt = JaggedTensor(values=values, offsets=offsets)

values_list = jt.to_dense()

# values_list = [
#     torch.tensor([1.0, 2.0]),
#     torch.tensor([]),
#     torch.tensor([3.0]),
#     torch.tensor([4.0]),
#     torch.tensor([5.0]),
#     torch.tensor([6.0, 7.0, 8.0]),
# ] 
```

```py
to_dense_weights() → Optional[List[Tensor]]
```

构造 JT 的权重的稠密表示。

返回：

张量列表，如果没有权重则为 None。

返回类型：

可选[列表[torch.Tensor]]

示例：

```py
values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

weights_list = jt.to_dense_weights()

# weights_list = [
#     torch.tensor([0.1, 0.2]),
#     torch.tensor([]),
#     torch.tensor([0.3]),
#     torch.tensor([0.4]),
#     torch.tensor([0.5]),
#     torch.tensor([0.6, 0.7, 0.8]),
# ] 
```

```py
to_padded_dense(desired_length: Optional[int] = None, padding_value: float = 0.0) → Tensor
```

从 JT 的值构造一个形状为(B, N,)的 2D 稠密张量。

请注意，B 是 self.lengths()的长度，N 是最长的特征长度或 desired_length。

如果 desired_length > length，则将使用 padding_value 进行填充，否则将选择 desired_length 处的最后一个值。

参数：

+   **desired_length** (*整数*) - 张量的长度。

+   **填充值** (*浮点数*) - 如果需要填充的填充值。

返回：

2d 稠密张量。

返回类型：

torch.Tensor

示例：

```py
values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
jt = JaggedTensor(values=values, offsets=offsets)

dt = jt.to_padded_dense(
    desired_length=2,
    padding_value=10.0,
)

# dt = [
#     [1.0, 2.0],
#     [10.0, 10.0],
#     [3.0, 10.0],
#     [4.0, 10.0],
#     [5.0, 10.0],
#     [6.0, 7.0],
# ] 
```

```py
to_padded_dense_weights(desired_length: Optional[int] = None, padding_value: float = 0.0) → Optional[Tensor]
```

从 JT 的权重构造一个形状为(B, N,)的 2D 稠密张量。

请注意，B 是 self.lengths()的长度，N 是最长的特征长度或 desired_length。

如果 desired_length > length，则将使用 padding_value 进行填充，否则将选择 desired_length 处的最后一个值。

参数：

+   **desired_length** (*整数*) - 张量的长度。

+   **padding_value** (*浮点数*) - 如果需要填充的填充值。

返回：

2d 稠密张量，如果没有权重则为 None。

返回类型：

可选[torch.Tensor]

示例：

```py
values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

d_wt = jt.to_padded_dense_weights(
    desired_length=2,
    padding_value=1.0,
)

# d_wt = [
#     [0.1, 0.2],
#     [1.0, 1.0],
#     [0.3, 1.0],
#     [0.4, 1.0],
#     [0.5, 1.0],
#     [0.6, 0.7],
# ] 
```

```py
values() → Tensor
```

```py
weights() → Tensor
```

```py
weights_or_none() → Optional[Tensor]
```

```py
class torchrec.sparse.jagged_tensor.JaggedTensorMeta(name, bases, namespace, **kwargs)
```

基类：`ABCMeta`，`ProxyableClassMeta`

```py
class torchrec.sparse.jagged_tensor.KeyedJaggedTensor(*args, **kwargs)
```

基类：`Pipelineable`

表示一个（可选加权）键控不规则张量。

KeyedJaggedTensor 是一个具有*jagged dimension*的张量，其切片可能具有不同的长度。在第一维上键入，在最后一维上不规则。

实现为 torch.jit.script-able。

参数：

+   **keys** (*列表**[**字符串**]*) - Jagged Tensor 的键。

+   **values** (*torch.Tensor*) - 稠密表示中的值张量。

+   **weights** (*可选**[**torch.Tensor**]*) - 如果值有权重。形状与值相同的张量。

+   **lengths** (*可选**[**torch.Tensor**]*) - 不规则切片，表示为长度。

+   **offsets** (*可选**[**torch.Tensor**]*) - 不规则切片，表示为累积偏移量。

+   **stride** (*可选**[**整数**]*) - 每批的示例数。

+   **stride_per_key_per_rank** (*可选**[**列表**[**列表**[**整数**]**]**]*) - 每个等级的每个键的批次大小（示例数），外部列表表示键，内部列表表示值。内部列表中的每个值表示批次中来自其索引在分布上下文中的等级的数量。

+   **length_per_key** (*可选**[**列表**[**整数**]**]*) - 每个键的起始长度。

+   **offset_per_key** (*可选**[**列表**[**整数**]**]*) - 每个键和最终偏移的起始偏移量。

+   **index_per_key** (*可选**[**字典**[**字符串**,* *整数**]**]*) - 每个键的索引。

+   **jt_dict** (*可选****字典**[**字符串**,* [*JaggedTensor**]**]*) -

+   **inverse_indices** (*可选**[**元组**[**列表**[**字符串**]**,* *torch.Tensor**]**]*) - 用于为每个键的可变步幅扩展去重嵌入输出的逆索引。

示例：

```py
#              0       1        2  <-- dim_1
# "Feature0"   [V0,V1] None    [V2]
# "Feature1"   [V3]    [V4]    [V5,V6,V7]
#   ^
#  dim_0

dim_0: keyed dimension (ie. `Feature0`, `Feature1`)
dim_1: optional second dimension (ie. batch size)
dim_2: The jagged dimension which has slice lengths between 0-3 in the above example

# We represent this data with following inputs:

values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset 
```

```py
static concat(kjt_list: List[KeyedJaggedTensor]) → KeyedJaggedTensor
```

```py
device() → device
```

```py
static dist_init(keys: List[str], tensors: List[Tensor], variable_stride_per_key: bool, num_workers: int, recat: Optional[Tensor], stride_per_rank: Optional[List[int]], stagger: int = 1) → KeyedJaggedTensor
```

```py
dist_labels() → List[str]
```

```py
dist_splits(key_splits: List[int]) → List[List[int]]
```

```py
dist_tensors() → List[Tensor]
```

```py
static empty(is_weighted: bool = False, device: Optional[device] = None, values_dtype: Optional[dtype] = None, weights_dtype: Optional[dtype] = None, lengths_dtype: dtype = torch.int32) → KeyedJaggedTensor
```

```py
static empty_like(kjt: KeyedJaggedTensor) → KeyedJaggedTensor
```

```py
flatten_lengths() → KeyedJaggedTensor
```

```py
static from_jt_dict(jt_dict: Dict[str, JaggedTensor]) → KeyedJaggedTensor
```

从 Dict[str, JaggedTensor]构造一个 KeyedJaggedTensor，但是如果 JaggedTensors 都具有相同的“隐式”batch_size 维度，则此函数将仅起作用。

基本上，我们可以将 JaggedTensors 可视化为格式为[batch_size x variable_feature_dim]的 2-D 张量。如果有一些批次没有特征值，输入的 JaggedTensor 可以不包含任何值。

但 KeyedJaggedTensor（默认情况下）通常会填充“None”，以便存储在 KeyedJaggedTensor 中的所有 JaggedTensors 具有相同的 batch_size 维度。也就是说，在这种情况下，JaggedTensor 输入没有自动为空批次填充，此函数将出错/无法工作。

考虑以下 KeyedJaggedTensor 的可视化：# 0 1 2 <– dim_1 # “Feature0” [V0,V1] None [V2] # “Feature1” [V3] [V4] [V5,V6,V7] # ^ # dim_0

请注意，这个 KeyedJaggedTensor 的输入看起来像：

values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7] # V == 任何张量数据类型 weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7] # W == 任何张量数据类型 lengths: torch.Tensor = [2, 0, 1, 1, 1, 3] # 表示 Jagged 切片的偏移量: torch.Tensor = [0, 2, 2, 3, 4, 5, 8] # 每个 Jagged 切片的偏移量从 0 开始 keys: List[str] = [“Feature0”, “Feature1”] # 对应于 dim_0 索引的每个值 index_per_key: Dict[str, int] = {“Feature0”: 0, “Feature1”: 1} # 每个键的索引 offset_per_key: List[int] = [0, 3, 8] # 每个键的起始偏移和最终偏移

现在，如果输入 jt_dict = {

# “Feature0” [V0,V1] [V2] # “Feature1” [V3] [V4] [V5,V6,V7]

}并且每个 JaggedTensor 中都省略了“None”，那么此函数将失败，因为我们无法正确地填充“None”，因为它在技术上不知道在 JaggedTensor 中正确的批次/位置进行填充。

基本上，此函数推断的长度张量将是[2, 1, 1, 1, 3]，表示可变的 batch_size dim_1 违反了现有的假设/前提，即 KeyedJaggedTensor 应该具有固定的 batch_size 维度。

```py
static from_lengths_sync(keys: List[str], values: Tensor, lengths: Tensor, weights: Optional[Tensor] = None, stride: Optional[int] = None, stride_per_key_per_rank: Optional[List[List[int]]] = None, inverse_indices: Optional[Tuple[List[str], Tensor]] = None) → KeyedJaggedTensor
```

```py
static from_offsets_sync(keys: List[str], values: Tensor, offsets: Tensor, weights: Optional[Tensor] = None, stride: Optional[int] = None, stride_per_key_per_rank: Optional[List[List[int]]] = None, inverse_indices: Optional[Tuple[List[str], Tensor]] = None) → KeyedJaggedTensor
```

```py
inverse_indices() → Tuple[List[str], Tensor]
```

```py
inverse_indices_or_none() → Optional[Tuple[List[str], Tensor]]
```

```py
keys() → List[str]
```

```py
length_per_key() → List[int]
```

```py
length_per_key_or_none() → Optional[List[int]]
```

```py
lengths() → Tensor
```

```py
lengths_offset_per_key() → List[int]
```

```py
lengths_or_none() → Optional[Tensor]
```

```py
offset_per_key() → List[int]
```

```py
offset_per_key_or_none() → Optional[List[int]]
```

```py
offsets() → Tensor
```

```py
offsets_or_none() → Optional[Tensor]
```

```py
permute(indices: List[int], indices_tensor: Optional[Tensor] = None) → KeyedJaggedTensor
```

```py
pin_memory() → KeyedJaggedTensor
```

```py
record_stream(stream: Stream) → None
```

请参阅[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
split(segments: List[int]) → List[KeyedJaggedTensor]
```

```py
stride() → int
```

```py
stride_per_key() → List[int]
```

```py
stride_per_key_per_rank() → List[List[int]]
```

```py
sync() → KeyedJaggedTensor
```

```py
to(device: device, non_blocking: bool = False, dtype: Optional[dtype] = None) → KeyedJaggedTensor
```

请注意，根据[`pytorch.org/docs/stable/generated/torch.Tensor.to.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)，to 可能返回 self 或 self 的副本。因此，请记住使用赋值运算符 to，例如，in = in.to(new_device)。

```py
to_dict() → Dict[str, JaggedTensor]
```

```py
unsync() → KeyedJaggedTensor
```

```py
values() → Tensor
```

```py
variable_stride_per_key() → bool
```

```py
weights() → Tensor
```

```py
weights_or_none() → Optional[Tensor]
```

```py
class torchrec.sparse.jagged_tensor.KeyedTensor(*args, **kwargs)
```

基类：`Pipelineable`

KeyedTensor 保存了一个稠密张量的连接列表，每个张量可以通过一个键访问。

键维度可以是可变长度（length_per_key）。常见用例包括存储不同维度的池化嵌入。

实现为 torch.jit.script-able。

参数：

+   **keys**（*List**[**str**]*） - 键列表。

+   **length_per_key**（*List**[**int**]*） - 沿键维度的每个键的长度。

+   **values**（*torch.Tensor*） - 稠密张量，通常沿键维度连接。

+   **key_dim**（*int*） - 键维度，从零开始索引，默认为 1（通常 B 是 0 维）。

示例：

```py
# kt is KeyedTensor holding

#                         0           1           2
#     "Embedding A"    [1,1]       [1,1]        [1,1]
#     "Embedding B"    [2,1,2]     [2,1,2]      [2,1,2]
#     "Embedding C"    [3,1,2,3]   [3,1,2,3]    [3,1,2,3]

tensor_list = [
    torch.tensor([[1,1]] * 3),
    torch.tensor([[2,1,2]] * 3),
    torch.tensor([[3,1,2,3]] * 3),
]

keys = ["Embedding A", "Embedding B", "Embedding C"]

kt = KeyedTensor.from_tensor_list(keys, tensor_list)

kt.values()
    # tensor(
    #     [
    #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
    #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
    #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
    #     ]
    # )

kt["Embedding B"]
    # tensor([[2, 1, 2], [2, 1, 2], [2, 1, 2]]) 
```

```py
static from_tensor_list(keys: List[str], tensors: List[Tensor], key_dim: int = 1, cat_dim: int = 1) → KeyedTensor
```

```py
key_dim() → int
```

```py
keys() → List[str]
```

```py
length_per_key() → List[int]
```

```py
offset_per_key() → List[int]
```

```py
record_stream(stream: Stream) → None
```

请参阅[`pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)

```py
static regroup(keyed_tensors: List[KeyedTensor], groups: List[List[str]]) → List[Tensor]
```

```py
static regroup_as_dict(keyed_tensors: List[KeyedTensor], groups: List[List[str]], keys: List[str]) → Dict[str, Tensor]
```

```py
to(device: device, non_blocking: bool = False) → KeyedTensor
```

请注意，根据[`pytorch.org/docs/stable/generated/torch.Tensor.to.html`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)，to 可能返回 self 或 self 的副本。因此，请记住使用赋值运算符 to，例如，in = in.to(new_device)。

```py
to_dict() → Dict[str, Tensor]
```

```py
values() → Tensor
```

```py
torchrec.sparse.jagged_tensor.jt_is_equal(jt_1: JaggedTensor, jt_2: JaggedTensor) → bool
```

此函数通过比较它们的内部表示来检查两个 JaggedTensors 是否相等。比较是通过比较内部表示的值来完成的。对于可选字段，将 None 值视为相等。

参数：

+   **jt_1**（*JaggedTensor* → bool
```

此函数通过比较它们的内部表示来检查两个 KeyedJaggedTensors 是否相等。比较是通过比较内部表示的值来完成的。对于可选字段，将 None 值视为相等。我们通过确保它们具有相同长度并且相应的键是相同顺序和相同值来比较键。

参数：

+   **kjt_1**（[*KeyedJaggedTensor*](#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor"） - 第一个 KeyedJaggedTensor

+   **kjt_2**（[*KeyedJaggedTensor*](#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor"） - 第二个 KeyedJaggedTensor

返回：

如果两个 KeyedJaggedTensors 具有相同的值，则为 True

返回类型：

bool  ## 模块内容[]（＃module-0“此标题的永久链接”）

Torchrec 不规则张量

它有 3 个类：JaggedTensor，KeyedJaggedTensor，KeyedTensor。

JaggedTensor

它表示一个（可选加权）不规则张量。 JaggedTensor 是一个具有不规则维度的张量，其切片可能具有不同的长度。 请参考 KeyedJaggedTensor 文档字符串以获取完整示例和更多信息。

KeyedJaggedTensor

KeyedJaggedTensor 具有额外的“Key”信息。 键在第一个维度上，最后一个维度上是不规则的。 请参考 KeyedJaggedTensor 文档字符串以获取完整示例和更多信息。

KeyedTensor

KeyedTensor 保存了一个连接的密集张量列表，每个张量可以通过键访问。 键的维度可以是可变长度（每个键的长度）。 常见用例包括存储不同维度的池化嵌入。 请参考 KeyedTensor 文档字符串以获取完整示例和更多信息。
