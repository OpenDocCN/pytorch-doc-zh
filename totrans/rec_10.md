# torchrec.modules

> 原文：[https://pytorch.org/torchrec/torchrec.modules.html](https://pytorch.org/torchrec/torchrec.modules.html)

Torchrec常见模块

torchrec模块包含各种模块的集合。

这些模块包括：

+   nn.Embedding和nn.EmbeddingBag的扩展，分别称为EmbeddingBagCollection和EmbeddingCollection。

+   已建立的模块，如[DeepFM](https://arxiv.org/pdf/1703.04247.pdf)和[CrossNet](https://arxiv.org/abs/1708.05123)。

+   常见的模块模式，如MLP和SwishLayerNorm。

+   TorchRec的自定义模块，如PositionWeightedModule和LazyModuleExtensionMixin。

+   EmbeddingTower和EmbeddingTowerCollection，逻辑上的“塔”嵌入传递给提供的交互模块。

## torchrec.modules.activation[](#module-torchrec.modules.activation "Permalink to this heading")

激活模块

```py
class torchrec.modules.activation.SwishLayerNorm(input_dims: Union[int, List[int], Size], device: Optional[device] = None)¶
```

基类：`Module`

应用带有层归一化的Swish函数：Y = X * Sigmoid(LayerNorm(X))。

参数：

+   **input_dims** (*Union**[**int**,* *List**[**int**]**,* *torch.Size**]*) – 要进行归一化的维度。如果输入张量的形状为[batch_size, d1, d2, d3]，设置input_dim=[d2, d3]将在最后两个维度上进行层归一化。

+   **device** (*Optional**[**torch.device**]*) – 默认计算设备。

示例：

```py
sln = SwishLayerNorm(100) 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input** (*torch.Tensor*) – 输入张量。

返回：

一个输出张量。

返回类型：

torch.Tensor

```py
training: bool¶
```  ## torchrec.modules.crossnet[](#module-torchrec.modules.crossnet "Permalink to this heading")

CrossNet API

```py
class torchrec.modules.crossnet.CrossNet(in_features: int, num_layers: int)¶
```

基类：`Module`

[交叉网络](https://arxiv.org/abs/1708.05123)：

Cross Net是对形状为\((*, N)\)的张量进行一系列“交叉”操作，使其形状相同，有效地创建\(N\)个可学习的多项式函数。

在这个模块中，交叉操作是基于一个满秩矩阵（NxN）定义的，这样交叉效应可以覆盖每一层上的所有位。在每一层l上，张量被转换为：

\[x_{l+1} = x_0 * (W_l \cdot x_l + b_l) + x_l\]

其中\(W_l\)是一个方阵\((NxN)\)，\(*)表示逐元素乘法，\(\cdot\)表示矩阵乘法。

参数：

+   **in_features** (*int*) – 输入的维度。

+   **num_layers** (*int*) – 模块中的层数。

示例：

```py
batch_size = 3
num_layers = 2
in_features = 10
input = torch.randn(batch_size, in_features)
dcn = CrossNet(num_layers=num_layers)
output = dcn(input) 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input** (*torch.Tensor*) – 形状为[batch_size, in_features]的张量。

返回：

形状为[batch_size, in_features]的张量。

返回类型：

torch.Tensor

```py
training: bool¶
```

```py
class torchrec.modules.crossnet.LowRankCrossNet(in_features: int, num_layers: int, low_rank: int = 1)¶
```

基类：`Module`

低秩交叉网络是一个高效的交叉网络。它不是在每一层使用满秩交叉矩阵（NxN），而是使用两个核\(W (N x r)\)和\(V (r x N)\)，其中r << N，以简化矩阵乘法。

在每一层l上，张量被转换为：

\[x_{l+1} = x_0 * (W_l \cdot (V_l \cdot x_l) + b_l) + x_l\]

其中\(W_l\)可以是一个向量，\(*)表示逐元素乘法，\(\cdot\)表示矩阵乘法。

注意

秩r应该被聪明地选择。通常，我们期望r < N/2以节省计算；我们应该期望\(r ~= N/4\)以保持完整秩交叉网络的准确性。

参数：

+   **in_features** (*int*) – 输入的维度。

+   **num_layers** (*int*) – 模块中的层数。

+   **low_rank** (*int*) – 交叉矩阵的秩设置（默认为1）。值必须始终 >= 1。

示例：

```py
batch_size = 3
num_layers = 2
in_features = 10
input = torch.randn(batch_size, in_features)
dcn = LowRankCrossNet(num_layers=num_layers, low_rank=3)
output = dcn(input) 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input** (*torch.Tensor*) – 形状为[batch_size, in_features]的张量。

返回：

形状为[batch_size, in_features]的张量。

返回类型：

torch.Tensor

```py
training: bool¶
```

```py
class torchrec.modules.crossnet.LowRankMixtureCrossNet(in_features: int, num_layers: int, num_experts: int = 1, low_rank: int = 1, activation: ~typing.Union[~torch.nn.modules.module.Module, ~typing.Callable[[~torch.Tensor], ~torch.Tensor]] = <built-in method relu of type object>)¶
```

基类：`Module`

低秩混合交叉网络是来自[论文](https://arxiv.org/pdf/2008.13535.pdf)的DCN V2实现：

LowRankMixtureCrossNet将每层的可学习交叉参数定义为一个低秩矩阵\((N*r)\)以及专家混合。与LowRankCrossNet相比，这个模块不依赖于单个专家来学习特征交叉，而是利用这样的\(K\)专家；每个专家在不同子空间中学习特征交互，并通过依赖于输入\(x\)的门控机制自适应地组合学习到的交叉。

在每一层l上，张量被转换为：

\[x_{l+1} = MoE({expert_i : i \in K_{experts}}) + x_l\]

每个\(expert_i\)被定义为：

\[expert_i = x_0 * (U_{li} \cdot g(C_{li} \cdot g(V_{li} \cdot x_l)) + b_l)\]

其中\(U_{li} (N, r)\)，\(C_{li} (r, r)\)和\(V_{li} (r, N)\)是低秩矩阵，\(*)表示逐元素乘法，\(x\)表示矩阵乘法，\(g()\)是非线性激活函数。

当num_expert为1时，门控评估和MOE将被跳过以节省计算。

参数：

+   **in_features**（*int*）- 输入的维度。

+   **num_layers**（*int*）- 模块中的层数。

+   **low_rank**（*int*）- 交叉矩阵的秩设置（默认= 1）。值必须始终>= 1

+   **activation**（*Union**[**torch.nn.Module**,* *Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]*)- 非线性激活函数，用于定义专家。默认为relu。

示例：

```py
batch_size = 3
num_layers = 2
in_features = 10
input = torch.randn(batch_size, in_features)
dcn = LowRankCrossNet(num_layers=num_layers, num_experts=5, low_rank=3)
output = dcn(input) 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input**（*torch.Tensor*）- 具有形状[batch_size，in_features]的张量。

返回：

具有形状[batch_size，in_features]的张量。

返回类型：

torch.Tensor

```py
training: bool¶
```

```py
class torchrec.modules.crossnet.VectorCrossNet(in_features: int, num_layers: int)¶
```

基类：`Module`

向量交叉网络可以被称为[DCN-V1](https://arxiv.org/pdf/1708.05123.pdf)。

它也是一个专门的低秩交叉网络，其中rank=1。在这个版本中，在每一层上，我们只保留一个向量核W（Nx1），而不是保留两个核W和V。我们使用点操作来计算特征的“交叉”效应，从而节省两次矩阵乘法以进一步减少计算成本并减少可学习参数的数量。

在每一层l上，张量被转换为

\[x_{l+1} = x_0 * (W_l . x_l + b_l) + x_l\]

其中\(W_l\)是一个向量，\(*)表示逐元素乘法；\(.\)表示点操作。

参数：

+   **in_features**（*int*）- 输入的维度。

+   **num_layers**（*int*）- 模块中的层数。

示例：

```py
batch_size = 3
num_layers = 2
in_features = 10
input = torch.randn(batch_size, in_features)
dcn = VectorCrossNet(num_layers=num_layers)
output = dcn(input) 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input**（*torch.Tensor*）- 具有形状[batch_size，in_features]的张量。

返回：

具有形状[batch_size，in_features]的张量。

返回类型：

torch.Tensor

```py
training: bool¶
```  ## torchrec.modules.deepfm[](#module-torchrec.modules.deepfm "Permalink to this heading")

深度因子分解机模块

以下模块基于[深度因子分解机（DeepFM）论文](https://arxiv.org/pdf/1703.04247.pdf)

+   类DeepFM实现了DeepFM框架

+   类FactorizationMachine实现了上述论文中提到的FM。

```py
class torchrec.modules.deepfm.DeepFM(dense_module: Module)¶
```

基类：`Module`

这是[DeepFM模块](https://arxiv.org/pdf/1703.04247.pdf)

这个模块不涵盖已发表论文的端到端功能。相反，它仅涵盖了出版物的深度组件。它用于学习高阶特征交互。如果应该学习低阶特征交互，请改用FactorizationMachine模块，它将共享此模块的嵌入输入。

为了支持建模的灵活性，我们将关键组件定制为：

+   与公开论文不同，我们将输入从原始稀疏特征更改为特征的嵌入。这允许在嵌入维度和嵌入数量方面具有灵活性，只要所有嵌入张量具有相同的批量大小。

+   在公开论文的基础上，我们允许用户自定义隐藏层为任何模块，不仅限于MLP。

模块的一般架构如下：

```py
 1 x 10                  output
         /|\
          |                     pass into `dense_module`
          |
        1 x 90
         /|\
          |                     concat
          |
1 x 20, 1 x 30, 1 x 40          list of embeddings 
```

参数：

**dense_module**（*nn.Module*）– DeepFM中可以使用的任何自定义模块（例如MLP）。此模块的in_features必须等于元素计数。例如，如果输入嵌入是[randn(3, 2, 3), randn(3, 4, 5)]，则in_features应为：2*3+4*5。

示例：

```py
import torch
from torchrec.fb.modules.deepfm import DeepFM
from torchrec.fb.modules.mlp import LazyMLP
batch_size = 3
output_dim = 30
# the input embedding are a torch.Tensor of [batch_size, num_embeddings, embedding_dim]
input_embeddings = [
    torch.randn(batch_size, 2, 64),
    torch.randn(batch_size, 2, 32),
]
dense_module = nn.Linear(192, output_dim)
deepfm = DeepFM(dense_module=dense_module)
deep_fm_output = deepfm(embeddings=input_embeddings) 
```

```py
forward(embeddings: List[Tensor]) → Tensor¶
```

参数：

**embeddings**（*List**[**torch.Tensor**]*）–

所有嵌入的列表（例如dense、common_sparse、specialized_sparse、embedding_features、raw_embedding_features）的形状为：

```py
(batch_size, num_embeddings, embedding_dim) 
```

为了方便操作，具有相同嵌入维度的嵌入可以选择堆叠到单个张量中。例如，当我们有1个维度为32的训练嵌入，5个维度为64的本地嵌入和3个维度为16的稠密特征时，我们可以准备嵌入列表为：

```py
tensor(B, 1, 32) (trained_embedding with num_embeddings=1, embedding_dim=32)
tensor(B, 5, 64) (native_embedding with num_embeddings=5, embedding_dim=64)
tensor(B, 3, 16) (dense_features with num_embeddings=3, embedding_dim=32) 
```

注意

所有输入张量的批量大小需要相同。

返回：

带有展平和连接的嵌入的dense_module输出作为输入。

返回类型：

torch.Tensor

```py
training: bool¶
```

```py
class torchrec.modules.deepfm.FactorizationMachine¶
```

继承：`Module`

这是因子分解机模块，在[DeepFM论文](https://arxiv.org/pdf/1703.04247.pdf)中提到：

该模块不涵盖已发表论文的端到端功能。相反，它仅涵盖了出版物的FM部分，并用于学习二阶特征交互。

为了支持建模灵活性，我们将关键组件定制为与公共论文不同：

> 我们将输入从原始稀疏特征更改为特征的嵌入。只要所有嵌入张量具有相同的批量大小，就可以灵活地设置嵌入维度和嵌入数量。

该模块的一般架构如下：

```py
 1 x 10                  output
         /|\
          |                     pass into `dense_module`
          |
        1 x 90
         /|\
          |                     concat
          |
1 x 20, 1 x 30, 1 x 40          list of embeddings 
```

示例：

```py
batch_size = 3
# the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
input_embeddings = [
    torch.randn(batch_size, 2, 64),
    torch.randn(batch_size, 2, 32),
]
fm = FactorizationMachine()
output = fm(embeddings=input_embeddings) 
```

```py
forward(embeddings: List[Tensor]) → Tensor¶
```

参数：

**embeddings**（*List**[**torch.Tensor**]*）–

所有嵌入的列表（例如dense、common_sparse、specialized_sparse、embedding_features、raw_embedding_features）的形状为：

```py
(batch_size, num_embeddings, embedding_dim) 
```

为了方便操作，具有相同嵌入维度的嵌入可以选择堆叠到单个张量中。例如，当我们有1个维度为32的训练嵌入，5个维度为64的本地嵌入和3个维度为16的稠密特征时，我们可以准备嵌入列表为：

```py
tensor(B, 1, 32) (trained_embedding with num_embeddings=1, embedding_dim=32)
tensor(B, 5, 64) (native_embedding with num_embeddings=5, embedding_dim=64)
tensor(B, 3, 16) (dense_features with num_embeddings=3, embedding_dim=32) 
```

注意

所有输入张量的批量大小需要相同。

返回：

带有展平和连接的嵌入的FM输出作为输入。预期为[B, 1]。

返回类型：

torch.Tensor

```py
training: bool¶
```  ## torchrec.modules.embedding_configs[](#module-torchrec.modules.embedding_configs "Permalink to this heading")

```py
class torchrec.modules.embedding_configs.BaseEmbeddingConfig(num_embeddings: int, embedding_dim: int, name: str = '', data_type: torchrec.types.DataType = <DataType.FP32: 'FP32'>, feature_names: List[str] = <factory>, weight_init_max: Union[float, NoneType] = None, weight_init_min: Union[float, NoneType] = None, pruning_indices_remapping: Union[torch.Tensor, NoneType] = None, init_fn: Union[Callable[[torch.Tensor], Union[torch.Tensor, NoneType]], NoneType] = None, need_pos: bool = False)¶
```

继承：`object`

```py
data_type: DataType = 'FP32'¶
```

```py
embedding_dim: int¶
```

```py
feature_names: List[str]¶
```

```py
get_weight_init_max() → float¶
```

```py
get_weight_init_min() → float¶
```

```py
init_fn: Optional[Callable[[Tensor], Optional[Tensor]]] = None¶
```

```py
name: str = ''¶
```

```py
need_pos: bool = False¶
```

```py
num_embeddings: int¶
```

```py
num_features() → int¶
```

```py
pruning_indices_remapping: Optional[Tensor] = None¶
```

```py
weight_init_max: Optional[float] = None¶
```

```py
weight_init_min: Optional[float] = None¶
```

```py
class torchrec.modules.embedding_configs.EmbeddingBagConfig(num_embeddings: int, embedding_dim: int, name: str = '', data_type: torchrec.types.DataType = <DataType.FP32: 'FP32'>, feature_names: List[str] = <factory>, weight_init_max: Union[float, NoneType] = None, weight_init_min: Union[float, NoneType] = None, pruning_indices_remapping: Union[torch.Tensor, NoneType] = None, init_fn: Union[Callable[[torch.Tensor], Union[torch.Tensor, NoneType]], NoneType] = None, need_pos: bool = False, pooling: torchrec.modules.embedding_configs.PoolingType = <PoolingType.SUM: 'SUM'>)¶
```

继承：[`BaseEmbeddingConfig`](#torchrec.modules.embedding_configs.BaseEmbeddingConfig "torchrec.modules.embedding_configs.BaseEmbeddingConfig")

```py
pooling: PoolingType = 'SUM'¶
```

```py
class torchrec.modules.embedding_configs.EmbeddingConfig(num_embeddings: int, embedding_dim: int, name: str = '', data_type: torchrec.types.DataType = <DataType.FP32: 'FP32'>, feature_names: List[str] = <factory>, weight_init_max: Union[float, NoneType] = None, weight_init_min: Union[float, NoneType] = None, pruning_indices_remapping: Union[torch.Tensor, NoneType] = None, init_fn: Union[Callable[[torch.Tensor], Union[torch.Tensor, NoneType]], NoneType] = None, need_pos: bool = False)¶
```

继承：[`BaseEmbeddingConfig`](#torchrec.modules.embedding_configs.BaseEmbeddingConfig "torchrec.modules.embedding_configs.BaseEmbeddingConfig")

```py
embedding_dim: int¶
```

```py
feature_names: List[str]¶
```

```py
num_embeddings: int¶
```

```py
class torchrec.modules.embedding_configs.EmbeddingTableConfig(num_embeddings: int, embedding_dim: int, name: str = '', data_type: torchrec.types.DataType = <DataType.FP32: 'FP32'>, feature_names: List[str] = <factory>, weight_init_max: Union[float, NoneType] = None, weight_init_min: Union[float, NoneType] = None, pruning_indices_remapping: Union[torch.Tensor, NoneType] = None, init_fn: Union[Callable[[torch.Tensor], Union[torch.Tensor, NoneType]], NoneType] = None, need_pos: bool = False, pooling: torchrec.modules.embedding_configs.PoolingType = <PoolingType.SUM: 'SUM'>, is_weighted: bool = False, has_feature_processor: bool = False, embedding_names: List[str] = <factory>)¶
```

继承：[`BaseEmbeddingConfig`](#torchrec.modules.embedding_configs.BaseEmbeddingConfig "torchrec.modules.embedding_configs.BaseEmbeddingConfig")

```py
embedding_names: List[str]¶
```

```py
has_feature_processor: bool = False¶
```

```py
is_weighted: bool = False¶
```

```py
pooling: PoolingType = 'SUM'¶
```

```py
class torchrec.modules.embedding_configs.PoolingType(value)¶
```

继承：`Enum`

一个枚举。

```py
MEAN = 'MEAN'¶
```

```py
NONE = 'NONE'¶
```

```py
SUM = 'SUM'¶
```

```py
class torchrec.modules.embedding_configs.QuantConfig(activation, weight, per_table_weight_dtype)¶
```

继承：`tuple`

```py
activation: PlaceholderObserver¶
```

字段编号0的别名

```py
per_table_weight_dtype: Optional[Dict[str, dtype]]¶
```

字段编号2的别名

```py
weight: PlaceholderObserver¶
```

字段编号1的别名

```py
torchrec.modules.embedding_configs.data_type_to_dtype(data_type: DataType) → dtype¶
```

```py
torchrec.modules.embedding_configs.data_type_to_sparse_type(data_type: DataType) → SparseType¶
```

```py
torchrec.modules.embedding_configs.dtype_to_data_type(dtype: dtype) → DataType¶
```

```py
torchrec.modules.embedding_configs.pooling_type_to_pooling_mode(pooling_type: PoolingType) → PoolingMode¶
```

```py
torchrec.modules.embedding_configs.pooling_type_to_str(pooling_type: PoolingType) → str¶
```  ## torchrec.modules.embedding_modules[](#module-torchrec.modules.embedding_modules "Permalink to this heading")

```py
class torchrec.modules.embedding_modules.EmbeddingBagCollection(tables: List[EmbeddingBagConfig], is_weighted: bool = False, device: Optional[device] = None)¶
```

继承：[`EmbeddingBagCollectionInterface`](#torchrec.modules.embedding_modules.EmbeddingBagCollectionInterface "torchrec.modules.embedding_modules.EmbeddingBagCollectionInterface")

EmbeddingBagCollection表示池化嵌入（EmbeddingBags）的集合。

它以KeyedJaggedTensor形式处理稀疏数据，其值形式为[F X B X L]，其中：

+   F：特征（键）

+   B：批量大小

+   L：稀疏特征的长度（不规则）

并输出形式为[B * (F * D)]的KeyedTensor的值，其中：

+   F：特征（键）

+   D：每个特征（键）的嵌入维度

+   B：批量大小

参数：

+   **tables**（*List**[*[*EmbeddingBagConfig*](#torchrec.modules.embedding_configs.EmbeddingBagConfig "torchrec.modules.embedding_configs.EmbeddingBagConfig")*]*）– 嵌入表的列表。

+   **is_weighted**（*bool*）- 输入KeyedJaggedTensor是否加权。

+   **设备**（*可选**[**torch.device**]*）- 默认计算设备。

示例：

```py
table_0 = EmbeddingBagConfig(
    name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
)
table_1 = EmbeddingBagConfig(
    name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
)

ebc = EmbeddingBagCollection(tables=[table_0, table_1])

#        0       1        2  <-- batch
# "f1"   [0,1] None    [2]
# "f2"   [3]    [4]    [5,6,7]
#  ^
# feature

features = KeyedJaggedTensor(
    keys=["f1", "f2"],
    values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
)

pooled_embeddings = ebc(features)
print(pooled_embeddings.values())
tensor([[-0.8899, -0.1342, -1.9060, -0.0905, -0.2814, -0.9369, -0.7783],
    [ 0.0000,  0.0000,  0.0000,  0.1598,  0.0695,  1.3265, -0.1011],
    [-0.4256, -1.1846, -2.1648, -1.0893,  0.3590, -1.9784, -0.7681]],
    grad_fn=<CatBackward0>)
print(pooled_embeddings.keys())
['f1', 'f2']
print(pooled_embeddings.offset_per_key())
tensor([0, 3, 7]) 
```

```py
property device: device¶
```

```py
embedding_bag_configs() → List[EmbeddingBagConfig]¶
```

```py
forward(features: KeyedJaggedTensor) → KeyedTensor¶
```

参数：

**特征**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor")）- 形式为[F X B X L]的KJT。

返回：

KeyedTensor

```py
is_weighted() → bool¶
```

```py
reset_parameters() → None¶
```

```py
training: bool¶
```

```py
class torchrec.modules.embedding_modules.EmbeddingBagCollectionInterface(*args, **kwargs)¶
```

基类：`ABC`，`Module`

嵌入袋集合的接口。

```py
abstract embedding_bag_configs() → List[EmbeddingBagConfig]¶
```

```py
abstract forward(features: KeyedJaggedTensor) → KeyedTensor¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
abstract is_weighted() → bool¶
```

```py
training: bool¶
```

```py
class torchrec.modules.embedding_modules.EmbeddingCollection(tables: List[EmbeddingConfig], device: Optional[device] = None, need_indices: bool = False)¶
```

基类：[`EmbeddingCollectionInterface`](#torchrec.modules.embedding_modules.EmbeddingCollectionInterface "torchrec.modules.embedding_modules.EmbeddingCollectionInterface")

嵌入集合表示一组非池化嵌入。

它以形式为[F X B X L]的KeyedJaggedTensor处理稀疏数据，其中：

+   F：特征（键）

+   B：批量大小

+   L：稀疏特征的长度（可变）

并输出Dict[特征（键），JaggedTensor]。每个JaggedTensor包含形式为(B * L) X D的值，其中：

+   B：批量大小

+   L：稀疏特征的长度（不规则）

+   D：每个特征（键）的嵌入维度和长度的形式为L

参数：

+   **表格**（*列表**[*[*嵌入配置*](#torchrec.modules.embedding_configs.EmbeddingConfig "torchrec.modules.embedding_configs.EmbeddingConfig")*]*）- 嵌入表格列表。

+   **设备**（*可选**[**torch.device**]*）- 默认计算设备。

+   **need_indices**（*bool*）- 如果我们需要将索引传递给最终查找字典。

示例：

```py
e1_config = EmbeddingConfig(
    name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
)
e2_config = EmbeddingConfig(
    name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
)

ec = EmbeddingCollection(tables=[e1_config, e2_config])

#     0       1        2  <-- batch
# 0   [0,1] None    [2]
# 1   [3]    [4]    [5,6,7]
# ^
# feature

features = KeyedJaggedTensor.from_offsets_sync(
    keys=["f1", "f2"],
    values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
)
feature_embeddings = ec(features)
print(feature_embeddings['f2'].values())
tensor([[-0.2050,  0.5478,  0.6054],
[ 0.7352,  0.3210, -3.0399],
[ 0.1279, -0.1756, -0.4130],
[ 0.7519, -0.4341, -0.0499],
[ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>) 
```

```py
property device: device¶
```

```py
embedding_configs() → List[EmbeddingConfig]¶
```

```py
embedding_dim() → int¶
```

```py
embedding_names_by_table() → List[List[str]]¶
```

```py
forward(features: KeyedJaggedTensor) → Dict[str, JaggedTensor]¶
```

参数：

**特征**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor")）- 形式为[F X B X L]的KJT。

返回：

字典[str, JaggedTensor]

```py
need_indices() → bool¶
```

```py
reset_parameters() → None¶
```

```py
training: bool¶
```

```py
class torchrec.modules.embedding_modules.EmbeddingCollectionInterface(*args, **kwargs)¶
```

基类：`ABC`，`Module`

嵌入集合的接口。

```py
abstract embedding_configs() → List[EmbeddingConfig]¶
```

```py
abstract embedding_dim() → int¶
```

```py
abstract embedding_names_by_table() → List[List[str]]¶
```

```py
abstract forward(features: KeyedJaggedTensor) → Dict[str, JaggedTensor]¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
abstract need_indices() → bool¶
```

```py
training: bool¶
```

```py
torchrec.modules.embedding_modules.get_embedding_names_by_table(tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]) → List[List[str]]¶
```

```py
torchrec.modules.embedding_modules.process_pooled_embeddings(pooled_embeddings: List[Tensor], inverse_indices: Tensor) → Tensor¶
```

```py
torchrec.modules.embedding_modules.reorder_inverse_indices(inverse_indices: Optional[Tuple[List[str], Tensor]], feature_names: List[str]) → Tensor¶
```  ## torchrec.modules.feature_processor[](#module-torchrec.modules.feature_processor "Permalink to this heading")

```py
class torchrec.modules.feature_processor.BaseFeatureProcessor(*args, **kwargs)¶
```

基类：`Module`

特征处理器的抽象基类。

```py
abstract forward(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
training: bool¶
```

```py
class torchrec.modules.feature_processor.BaseGroupedFeatureProcessor(*args, **kwargs)¶
```

基类：`Module`

分组特征处理器的抽象基类

```py
abstract forward(features: KeyedJaggedTensor) → KeyedJaggedTensor¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此之后调用，因为前者负责运行注册的钩子，而后者则默默地忽略它们。

```py
training: bool¶
```

```py
class torchrec.modules.feature_processor.PositionWeightedModule(max_feature_lengths: Dict[str, int], device: Optional[device] = None)¶
```

基类：[`BaseFeatureProcessor`](#torchrec.modules.feature_processor.BaseFeatureProcessor "torchrec.modules.feature_processor.BaseFeatureProcessor")

向id列表特征添加位置权重。

参数：

**max_feature_lengths**（*字典**[**str**,* *int**]*）- 特征名称到最大长度的映射。max_length，也称为截断大小，指定每个样本具有的最大id数量。对于每个特征，其位置权重参数大小为max_length。

```py
forward(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

参数：

**特征**（*字典**[**str**,* [*JaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor "torchrec.sparse.jagged_tensor.JaggedTensor")*]*）- 键到JaggedTensor的字典，表示特征。

返回：

与输入特征相同，权重字段已填充。

返回类型：

Dict[str，[JaggedTensor](torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor "torchrec.sparse.jagged_tensor.JaggedTensor")]

```py
reset_parameters() → None¶
```

```py
training: bool¶
```

```py
class torchrec.modules.feature_processor.PositionWeightedProcessor(max_feature_lengths: Dict[str, int], device: Optional[device] = None)¶
```

基类：[`BaseGroupedFeatureProcessor`](#torchrec.modules.feature_processor.BaseGroupedFeatureProcessor "torchrec.modules.feature_processor.BaseGroupedFeatureProcessor")

PositionWeightedProcessor表示将位置权重应用于KeyedJaggedTensor的处理器。

它可以处理非分片和分片输入以及相应的输出

参数：

+   **max_feature_lengths**（*Dict**[**str**，* *int**]）- feature_lengths的字典，键是feature_name，值是长度。

+   **device**（*Optional**[**torch.device**]）- 默认计算设备。

示例：

```py
keys=["Feature0", "Feature1", "Feature2"]
values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7])
lengths=torch.tensor([2, 0, 1, 1, 1, 3, 2, 3, 0])
features = KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)
pw = FeatureProcessorCollection(
    feature_processor_modules={key: PositionWeightedFeatureProcessor(max_feature_length=100) for key in keys}
)
result = pw(features)
# result is
# KeyedJaggedTensor({
#     "Feature0": {
#         "values": [[0, 1], [], [2]],
#         "weights": [[1.0, 1.0], [], [1.0]]
#     },
#     "Feature1": {
#         "values": [[3], [4], [5, 6, 7]],
#         "weights": [[1.0], [1.0], [1.0, 1.0, 1.0]]
#     },
#     "Feature2": {
#         "values": [[3, 4], [5, 6, 7], []],
#         "weights": [[1.0, 1.0], [1.0, 1.0, 1.0], []]
#     }
# }) 
```

```py
forward(features: KeyedJaggedTensor) → KeyedJaggedTensor¶
```

在非分片或非流水线模型中，输入特征同时包含fp_feature和non_fp_features，输出将过滤掉non_fp特征。在分片流水线模型中，输入特征只能包含所有或所有feature_processed特征，因为输入特征来自ebc的input_dist()，该函数将过滤掉不在ebc中的键。输入大小与输出大小相同

参数：

**features**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor")）- 输入特征

返回：

KeyedJaggedTensor

```py
named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) → Iterator[Tuple[str, Tensor]]¶
```

返回一个模块缓冲区的迭代器，同时生成缓冲区的名称和缓冲区本身。

参数：

+   **prefix**（*str*）- 要添加到所有缓冲区名称前面的前缀。

+   **recurse**（*bool**，*可选*）- 如果为True，则生成此模块和所有子模块的缓冲区。否则，仅生成直接属于此模块的缓冲区。默认为True。

+   **remove_duplicate**（*bool**，*可选*）- 是否在结果中删除重复的缓冲区。默认为True。

产出：

*(str，torch.Tensor)* - 包含名称和缓冲区的元组

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>>     if name in ['running_var']:
>>>         print(buf.size()) 
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]¶
```

返回包含模块整个状态的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

当前`state_dict()`还接受`destination`，`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在将来的版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**，*可选*）- 如果提供，则模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix**（*str**，*可选*）- 添加到state_dict中的参数和缓冲区名称以组成键的前缀。默认值：`''`。

+   **keep_vars**（*bool**，*可选*）- 默认情况下，state dict中返回的`Tensor`会与autograd分离。如果设置为`True`，则不会执行分离。默认值：`False`。

返回：

包含模块整个状态的字典

返回类型：

字典

示例：

```py
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight'] 
```

```py
training: bool¶
```

```py
torchrec.modules.feature_processor.position_weighted_module_update_features(features: Dict[str, JaggedTensor], weighted_features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```  ## torchrec.modules.lazy_extension[](#module-torchrec.modules.lazy_extension "Permalink to this heading")

```py
class torchrec.modules.lazy_extension.LazyModuleExtensionMixin(*args, **kwargs)¶
```

基类：`LazyModuleMixin`

这是LazyModuleMixin的临时扩展，支持将关键字参数传递给惰性模块的前向方法。

长期计划是将此功能上游到LazyModuleMixin。有关详细信息，请参阅[https://github.com/pytorch/pytorch/issues/59923](https://github.com/pytorch/pytorch/issues/59923)。

请参阅TestLazyModuleExtensionMixin，其中包含确保的单元测试：

+   LazyModuleExtensionMixin._infer_parameters与torch.nn.modules.lazy.LazyModuleMixin._infer_parameters具有源代码的一致性，只是前者可以接受关键字参数。

+   LazyModuleExtensionMixin._call_impl的源代码与torch.nn.Module._call_impl具有相同的代码平等性，只是前者可以将关键字参数传递给forward pre hooks。

```py
apply(fn: Callable[[Module], None]) → Module¶
```

将fn递归地应用于每个子模块（由.children()返回），以及self。典型用法包括初始化模型的参数。

注意

在未初始化的懒惰模块上调用apply()将导致错误。用户需要在对懒惰模块调用apply()之前初始化懒惰模块（通过进行虚拟前向传递）。

参数：

**fn**（*torch.nn.Module -> None*） - 要应用于每个子模块的函数。

返回：

self

返回类型：

torch.nn.Module

示例：

```py
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == torch.nn.LazyLinear:
        m.weight.fill_(1.0)
        print(m.weight)

linear = torch.nn.LazyLinear(2)
linear.apply(init_weights)  # this fails, because `linear` (a lazy-module) hasn't been initialized yet

input = torch.randn(2, 10)
linear(input)  # run a dummy forward pass to initialize the lazy-module

linear.apply(init_weights)  # this works now 
```

```py
torchrec.modules.lazy_extension.lazy_apply(module: Module, fn: Callable[[Module], None]) → Module¶
```

将一个函数附加到一个模块，该函数将递归地应用于模块的每个子模块（由.children()返回）以及模块本身，就在第一次前向传递之后（即在所有子模块和参数初始化之后）。

典型用法包括初始化懒惰模块的参数的数值（即从LazyModuleMixin继承的模块）。

注意

lazy_apply()可用于懒惰和非懒惰模块。

参数：

+   **module**（*torch.nn.Module*） - 递归应用fn的模块。

+   **fn**（*Callable**[**[**torch.nn.Module**]**,* *None**]*） - 要附加到模块并稍后应用于模块的每个子模块和模块本身的函数。

返回：

附加了fn的模块。

返回类型：

torch.nn.Module

示例：

```py
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == torch.nn.LazyLinear:
        m.weight.fill_(1.0)
        print(m.weight)

linear = torch.nn.LazyLinear(2)
lazy_apply(linear, init_weights)  # doesn't run `init_weights` immediately
input = torch.randn(2, 10)
linear(input)  # runs `init_weights` only once, right after first forward pass

seq = torch.nn.Sequential(torch.nn.LazyLinear(2), torch.nn.LazyLinear(2))
lazy_apply(seq, init_weights)  # doesn't run `init_weights` immediately
input = torch.randn(2, 10)
seq(input)  # runs `init_weights` only once, right after first forward pass 
```  ## torchrec.modules.mlp[](#module-torchrec.modules.mlp "Permalink to this heading")

```py
class torchrec.modules.mlp.MLP(in_size: int, layer_sizes: ~typing.List[int], bias: bool = True, activation: ~typing.Union[str, ~typing.Callable[[], ~torch.nn.modules.module.Module], ~torch.nn.modules.module.Module, ~typing.Callable[[~torch.Tensor], ~torch.Tensor]] = <built-in method relu of type object>, device: ~typing.Optional[~torch.device] = None, dtype: ~torch.dtype = torch.float32)¶
```

基类：`Module`

按顺序应用一堆感知器模块（即多层感知器）。

参数：

+   **in_size**（*int*） - 输入的in_size。

+   **layer_sizes**（*List**[**int**]*） - 每个感知器模块的out_size。

+   **bias**（*bool*） - 如果设置为False，则该层将不会学习附加偏差。默认值：True。

+   **activation**（*str**,* *Union**[**Callable**[**[**]**,* *torch.nn.Module**]**,* *torch.nn.Module**,* *Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]*） - 要应用于每个感知器模块的线性变换输出的激活函数。如果激活是一个str，我们目前只支持以下字符串，如“relu”，“sigmoid”和“swish_layernorm”。如果激活是一个Callable[[], torch.nn.Module]，则会为每个感知器模块调用activation()一次，以生成该感知器模块的激活模块，并且这些激活模块之间不会共享参数。一个用例是当所有激活模块共享相同的构造函数参数，但不共享实际的模块参数时。默认值：torch.relu。

+   **device**（*Optional**[**torch.device**]*） - 默认计算设备。

示例：

```py
batch_size = 3
in_size = 40
input = torch.randn(batch_size, in_size)

layer_sizes = [16, 8, 4]
mlp_module = MLP(in_size, layer_sizes, bias=True)
output = mlp_module(input)
assert list(output.shape) == [batch_size, layer_sizes[-1]] 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input**（*torch.Tensor*） - 形状为(B, I)的张量，其中I是每个输入样本中的元素数量。

返回：

形状为(B, O)的张量，其中O是最后一个感知器模块的out_size。

返回类型：

torch.Tensor

```py
training: bool¶
```

```py
class torchrec.modules.mlp.Perceptron(in_size: int, out_size: int, bias: bool = True, activation: ~typing.Union[~torch.nn.modules.module.Module, ~typing.Callable[[~torch.Tensor], ~torch.Tensor]] = <built-in method relu of type object>, device: ~typing.Optional[~torch.device] = None, dtype: ~torch.dtype = torch.float32)¶
```

基类：`Module`

应用线性变换和激活。

参数：

+   **in_size**（*int*） - 每个输入样本中的元素数量。

+   **out_size**（*int*） - 每个输出样本中的元素数量。

+   **bias**（*bool*） - 如果设置为`False`，该层将不会学习附加偏差。默认值：`True`。

+   **activation**（*Union**[**torch.nn.Module**,* *Callable**[**[**torch.Tensor**]**,* *torch.Tensor**]**]*） - 要应用于线性变换输出的激活函数。默认值：torch.relu。

+   **device**（*Optional**[**torch.device**]*） - 默认计算设备。

示例：

```py
batch_size = 3
in_size = 40
input = torch.randn(batch_size, in_size)

out_size = 16
perceptron = Perceptron(in_size, out_size, bias=True)

output = perceptron(input)
assert list(output) == [batch_size, out_size] 
```

```py
forward(input: Tensor) → Tensor¶
```

参数：

**input**（*torch.Tensor*） - 形状为(B, I)的张量，其中I是每个输入样本中的元素数量。

返回：

形状为(B, O)的张量，其中O是每个输入样本中的元素数量。

每个输出样本中的通道（即out_size）。

返回类型：

torch.Tensor

```py
training: bool¶
```  ## torchrec.modules.utils[](#module-torchrec.modules.utils "Permalink to this heading")

```py
torchrec.modules.utils.check_module_output_dimension(module: Union[Iterable[Module], Module], in_features: int, out_features: int) → bool¶
```

验证给定模块或模块列表的out_features是否与指定的数字匹配。如果给定模块列表或ModuleList，则递归检查所有子模块。

```py
torchrec.modules.utils.construct_jagged_tensors(embeddings: Tensor, features: KeyedJaggedTensor, embedding_names: List[str], need_indices: bool = False, features_to_permute_indices: Optional[Dict[str, List[int]]] = None, original_features: Optional[KeyedJaggedTensor] = None, reverse_indices: Optional[Tensor] = None) → Dict[str, JaggedTensor]¶
```

```py
torchrec.modules.utils.construct_modulelist_from_single_module(module: Module, sizes: Tuple[int, ...]) → Module¶
```

给定单个模块，通过复制提供的模块并重新初始化线性层来构造大小为sizes的（嵌套的）ModuleList。

```py
torchrec.modules.utils.convert_list_of_modules_to_modulelist(modules: Iterable[Module], sizes: Tuple[int, ...]) → Module¶
```

```py
torchrec.modules.utils.extract_module_or_tensor_callable(module_or_callable: Union[Callable[[], Module], Module, Callable[[Tensor], Tensor]]) → Union[Module, Callable[[Tensor], Tensor]]¶
```

```py
torchrec.modules.utils.get_module_output_dimension(module: Union[Callable[[Tensor], Tensor], Module], in_features: int) → int¶
```

```py
torchrec.modules.utils.init_mlp_weights_xavier_uniform(m: Module) → None¶
```

## torchrec.modules.mc_modules[](#torchrec-modules-mc-modules "此标题的永久链接")

```py
class torchrec.modules.mc_modules.DistanceLFU_EvictionPolicy(decay_exponent: float = 1.0, threshold_filtering_func: Optional[Callable[[Tensor], Tuple[Tensor, Union[float, Tensor]]]] = None)¶
```

基类：[`MCHEvictionPolicy`](#torchrec.modules.mc_modules.MCHEvictionPolicy "torchrec.modules.mc_modules.MCHEvictionPolicy")

```py
coalesce_history_metadata(current_iter: int, history_metadata: Dict[str, Tensor], unique_ids_counts: Tensor, unique_inverse_mapping: Tensor, additional_ids: Optional[Tensor] = None, threshold_mask: Optional[Tensor] = None) → Dict[str, Tensor]¶
```

参数：history_metadata（Dict[str，torch.Tensor]）：历史元数据字典 additional_ids（torch.Tensor）：要用作历史的一部分的额外ids unique_inverse_mapping（torch.Tensor）：从torch.unique生成的逆映射

> 使用torch.cat[history_accumulator, additional_ids]将历史元数据张量索引映射到它们的合并张量索引。

合并元数据历史缓冲区并返回处理后的元数据张量字典。

```py
property metadata_info: List[MCHEvictionPolicyMetadataInfo]¶
```

```py
record_history_metadata(current_iter: int, incoming_ids: Tensor, history_metadata: Dict[str, Tensor]) → None¶
```

参数：current_iter（int）：当前迭代 incoming_ids（torch.Tensor）：传入的ids history_metadata（Dict[str，torch.Tensor]）：历史元数据字典

根据传入的ids计算并记录元数据

对于实现的驱逐策略。

```py
update_metadata_and_generate_eviction_scores(current_iter: int, mch_size: int, coalesced_history_argsort_mapping: Tensor, coalesced_history_sorted_unique_ids_counts: Tensor, coalesced_history_mch_matching_elements_mask: Tensor, coalesced_history_mch_matching_indices: Tensor, mch_metadata: Dict[str, Tensor], coalesced_history_metadata: Dict[str, Tensor]) → Tuple[Tensor, Tensor]¶
```

参数：

返回（被驱逐的索引，选定的新索引）的元组：

被驱逐的索引是要被驱逐的mch映射中的索引，而selected_new_indices是要添加到mch中的合并历史中ids的索引。

```py
class torchrec.modules.mc_modules.LFU_EvictionPolicy(threshold_filtering_func: Optional[Callable[[Tensor], Tuple[Tensor, Union[float, Tensor]]]] = None)¶
```

基类：[`MCHEvictionPolicy`](#torchrec.modules.mc_modules.MCHEvictionPolicy "torchrec.modules.mc_modules.MCHEvictionPolicy")

```py
coalesce_history_metadata(current_iter: int, history_metadata: Dict[str, Tensor], unique_ids_counts: Tensor, unique_inverse_mapping: Tensor, additional_ids: Optional[Tensor] = None, threshold_mask: Optional[Tensor] = None) → Dict[str, Tensor]¶
```

参数：history_metadata（Dict[str，torch.Tensor]）：历史元数据字典 additional_ids（torch.Tensor）：要用作历史的一部分的额外ids unique_inverse_mapping（torch.Tensor）：从torch.unique生成的逆映射

> 使用torch.cat[history_accumulator, additional_ids]将历史元数据张量索引映射到它们的合并张量索引。

合并元数据历史缓冲区并返回处理后的元数据张量字典。

```py
property metadata_info: List[MCHEvictionPolicyMetadataInfo]¶
```

```py
record_history_metadata(current_iter: int, incoming_ids: Tensor, history_metadata: Dict[str, Tensor]) → None¶
```

参数：current_iter（int）：当前迭代 incoming_ids（torch.Tensor）：传入的ids history_metadata（Dict[str，torch.Tensor]）：历史元数据字典

根据传入的ids计算并记录元数据

对于实现的驱逐策略。

```py
update_metadata_and_generate_eviction_scores(current_iter: int, mch_size: int, coalesced_history_argsort_mapping: Tensor, coalesced_history_sorted_unique_ids_counts: Tensor, coalesced_history_mch_matching_elements_mask: Tensor, coalesced_history_mch_matching_indices: Tensor, mch_metadata: Dict[str, Tensor], coalesced_history_metadata: Dict[str, Tensor]) → Tuple[Tensor, Tensor]¶
```

参数：

返回（被驱逐的索引，选定的新索引）的元组：

被驱逐的索引是要被驱逐的mch映射中的索引，而selected_new_indices是要添加到mch中的合并历史中ids的索引。

```py
class torchrec.modules.mc_modules.LRU_EvictionPolicy(decay_exponent: float = 1.0, threshold_filtering_func: Optional[Callable[[Tensor], Tuple[Tensor, Union[float, Tensor]]]] = None)¶
```

基类：[`MCHEvictionPolicy`](#torchrec.modules.mc_modules.MCHEvictionPolicy "torchrec.modules.mc_modules.MCHEvictionPolicy")

```py
coalesce_history_metadata(current_iter: int, history_metadata: Dict[str, Tensor], unique_ids_counts: Tensor, unique_inverse_mapping: Tensor, additional_ids: Optional[Tensor] = None, threshold_mask: Optional[Tensor] = None) → Dict[str, Tensor]¶
```

参数：history_metadata（Dict[str，torch.Tensor]）：历史元数据字典 additional_ids（torch.Tensor）：要用作历史的一部分的额外ids unique_inverse_mapping（torch.Tensor）：从torch.unique生成的逆映射

> 使用torch.cat[history_accumulator, additional_ids]将历史元数据张量索引映射到它们的合并张量索引。

合并元数据历史缓冲区并返回处理后的元数据张量字典。

```py
property metadata_info: List[MCHEvictionPolicyMetadataInfo]¶
```

```py
record_history_metadata(current_iter: int, incoming_ids: Tensor, history_metadata: Dict[str, Tensor]) → None¶
```

参数：current_iter（int）：当前迭代 incoming_ids（torch.Tensor）：传入的ids history_metadata（Dict[str，torch.Tensor]）：历史元数据字典

根据传入的ids计算并记录元数据

对于实现的驱逐策略。

```py
update_metadata_and_generate_eviction_scores(current_iter: int, mch_size: int, coalesced_history_argsort_mapping: Tensor, coalesced_history_sorted_unique_ids_counts: Tensor, coalesced_history_mch_matching_elements_mask: Tensor, coalesced_history_mch_matching_indices: Tensor, mch_metadata: Dict[str, Tensor], coalesced_history_metadata: Dict[str, Tensor]) → Tuple[Tensor, Tensor]¶
```

参数：

返回（被驱逐的索引，选定的新索引）的元组：

被驱逐的索引是要被驱逐的mch映射中的索引，而selected_new_indices是要添加到mch中的合并历史中ids的索引。

```py
class torchrec.modules.mc_modules.MCHEvictionPolicy(metadata_info: List[MCHEvictionPolicyMetadataInfo], threshold_filtering_func: Optional[Callable[[Tensor], Tuple[Tensor, Union[float, Tensor]]]] = None)¶
```

基类：`ABC`

```py
abstract coalesce_history_metadata(current_iter: int, history_metadata: Dict[str, Tensor], unique_ids_counts: Tensor, unique_inverse_mapping: Tensor, additional_ids: Optional[Tensor] = None, threshold_mask: Optional[Tensor] = None) → Dict[str, Tensor]¶
```

参数：history_metadata（Dict[str，torch.Tensor]）：历史元数据字典 additional_ids（torch.Tensor）：要用作历史的一部分的额外ids unique_inverse_mapping（torch.Tensor）：从torch.unique生成的逆映射

> 使用torch.cat[history_accumulator, additional_ids]将历史元数据张量索引映射到它们的合并张量索引。

合并元数据历史缓冲区并返回处理后的元数据张量字典。

```py
abstract property metadata_info: List[MCHEvictionPolicyMetadataInfo]¶
```

```py
abstract record_history_metadata(current_iter: int, incoming_ids: Tensor, history_metadata: Dict[str, Tensor]) → None¶
```

参数：current_iter（int）：当前迭代incoming_ids（torch.Tensor）：传入的ids history_metadata（Dict[str，torch.Tensor]）：历史元数据字典

基于传入的ids计算和记录元数据

用于实现驱逐策略。

```py
abstract update_metadata_and_generate_eviction_scores(current_iter: int, mch_size: int, coalesced_history_argsort_mapping: Tensor, coalesced_history_sorted_unique_ids_counts: Tensor, coalesced_history_mch_matching_elements_mask: Tensor, coalesced_history_mch_matching_indices: Tensor, mch_metadata: Dict[str, Tensor], coalesced_history_metadata: Dict[str, Tensor]) → Tuple[Tensor, Tensor]¶
```

参数：

返回（驱逐的索引，选择的新索引）的元组，其中：

被驱逐的索引是要被驱逐的mch映射中的索引，而选择的新索引是要添加到mch中的合并历史中的id的索引。

```py
class torchrec.modules.mc_modules.MCHEvictionPolicyMetadataInfo(metadata_name, is_mch_metadata, is_history_metadata)¶
```

基础：`tuple`

```py
is_history_metadata: bool¶
```

字段编号2的别名

```py
is_mch_metadata: bool¶
```

字段编号1的别名

```py
metadata_name: str¶
```

字段编号0的别名

```py
class torchrec.modules.mc_modules.MCHManagedCollisionModule(zch_size: int, device: device, eviction_policy: MCHEvictionPolicy, eviction_interval: int, input_hash_size: int = 9223372036854775808, input_hash_func: Optional[Callable[[Tensor, int], Tensor]] = None, mch_size: Optional[int] = None, mch_hash_func: Optional[Callable[[Tensor, int], Tensor]] = None, name: Optional[str] = None, output_global_offset: int = 0)¶
```

基础：[`ManagedCollisionModule`](#torchrec.modules.mc_modules.ManagedCollisionModule "torchrec.modules.mc_modules.ManagedCollisionModule")

ZCH / MCH管理的碰撞模块

参数：

+   **zch_size**（*int*）-输出id的范围，在[output_size_offset，output_size_offset + zch_size - 1]内

+   **device**（*torch.device*）-将执行此模块的设备

+   **eviction_policy**（*驱逐策略*）-要使用的驱逐策略

+   **eviction_interval**（*int*）-触发驱逐策略的间隔

+   **input_hash_size**（*int*）-输入特征id范围，将作为第二个参数传递给input_hash_func

+   **input_hash_func**（*可选**[**Callable**]）-用于为输入特征生成哈希的函数。此函数通常用于在与输入数据相同或更大的范围内驱动均匀分布

+   **mch_size**（*可选**[**int**]）-残余输出的大小（即传统MCH），实验性功能。 Ids在内部移位为output_size_offset + zch_output_range

+   **mch_hash_func**（*可选**[**Callable**]）-用于为残余特征生成哈希的函数。将哈希降至mch_size。

+   **output_global_offset**（*int*）-输出范围的输出id的偏移量，通常仅在分片应用程序中使用。

```py
evict() → Optional[Tensor]¶
```

如果此迭代不应进行驱逐，则返回None。否则，返回要重置的插槽的id。在驱逐时，此模块应为这些插槽重置其状态，并假设下游模块将正确处理此操作。

```py
forward(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

参数：feature（JaggedTensor]）：特征表示：返回：修改后的JT：rtype：Dict[str，JaggedTensor]

```py
input_size() → int¶
```

返回输入的数字范围，用于分片信息

```py
output_size() → int¶
```

返回输出的数字范围，用于验证与下游嵌入查找

```py
preprocess(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

```py
profile(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

```py
rebuild_with_output_id_range(output_id_range: Tuple[int, int], device: Optional[device] = None) → MCHManagedCollisionModule¶
```

用于为RW分片创建本地MC模块，现在是一个hack

```py
remap(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

```py
training: bool¶
```

```py
class torchrec.modules.mc_modules.ManagedCollisionCollection(managed_collision_modules: Dict[str, ManagedCollisionModule], embedding_configs: List[BaseEmbeddingConfig])¶
```

基础：`Module`

ManagedCollisionCollection表示一组受管理的碰撞模块。传递给MCC的输入将由受管理的碰撞模块重新映射

> 并返回。

参数：

+   **managed_collision_modules**（*Dict**[**str**,* [*ManagedCollisionModule*](#torchrec.modules.mc_modules.ManagedCollisionModule "torchrec.modules.mc_modules.ManagedCollisionModule")*]）-受管理的碰撞模块的字典

+   **embedding_confgs**（*List**[*[*BaseEmbeddingConfig*](#torchrec.modules.embedding_configs.BaseEmbeddingConfig "torchrec.modules.embedding_configs.BaseEmbeddingConfig")*]）-每个具有受管理碰撞模块的表的嵌入配置列表

```py
embedding_configs() → List[BaseEmbeddingConfig]¶
```

```py
evict() → Dict[str, Optional[Tensor]]¶
```

```py
forward(features: KeyedJaggedTensor) → KeyedJaggedTensor¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

尽管前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这样做，因为前者负责运行注册的钩子，而后者则会默默地忽略它们。

```py
training: bool¶
```

```py
class torchrec.modules.mc_modules.ManagedCollisionModule(device: device)¶
```

基础：`Module`

ManagedCollisionModule的抽象基类。将输入id映射到范围[0，max_output_id)。

参数：

+   **max_output_id**（*int*）-重新映射的id的最大输出值。

+   **input_hash_size**（*int*）-输入范围的最大值，即[0，input_hash_size]

+   **remapping_range_start_index**（*int*）-重新映射范围的相对起始索引

+   **device**（*torch.device*）-默认计算设备。

示例：

jt = JaggedTensor(…) mcm = ManagedCollisionModule(…) mcm_jt = mcm(fp)

```py
property device: device¶
```

```py
abstract evict() → Optional[Tensor]¶
```

如果本次迭代不应进行驱逐，则返回None。否则，返回要重置的插槽的ID。在驱逐时，此模块应为这些插槽重置其状态，假设下游模块将正确处理此操作。

```py
abstract forward(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
abstract input_size() → int¶
```

返回输入的数值范围，用于分片信息

```py
abstract output_size() → int¶
```

返回输出的数值范围，用于验证与下游嵌入查找的比较

```py
abstract preprocess(features: Dict[str, JaggedTensor]) → Dict[str, JaggedTensor]¶
```

```py
abstract rebuild_with_output_id_range(output_id_range: Tuple[int, int], device: Optional[device] = None) → ManagedCollisionModule¶
```

用于为RW分片创建本地MC模块，目前是一个hack

```py
training: bool¶
```

```py
torchrec.modules.mc_modules.apply_mc_method_to_jt_dict(method: str, features_dict: Dict[str, JaggedTensor], table_to_features: Dict[str, List[str]], managed_collisions: ModuleDict) → Dict[str, JaggedTensor]¶
```

将MC方法应用于JaggedTensors字典，返回具有相同顺序的更新字典

```py
torchrec.modules.mc_modules.average_threshold_filter(id_counts: Tensor) → Tuple[Tensor, Tensor]¶
```

```py
torchrec.modules.mc_modules.dynamic_threshold_filter(id_counts: Tensor, threshold_skew_multiplier: float = 10.0) → Tuple[Tensor, Tensor]¶
```

## torchrec.modules.mc_embedding_modules[](#torchrec-modules-mc-embedding-modules "Permalink to this heading")

```py
class torchrec.modules.mc_embedding_modules.BaseManagedCollisionEmbeddingCollection(embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection], managed_collision_collection: ManagedCollisionCollection, return_remapped_features: bool = False)¶
```

基类：`Module`

BaseManagedCollisionEmbeddingCollection代表一个EC/EBC模块和一组管理的冲突模块。MC-EC/EBC的输入将首先通过管理的冲突模块进行修改，然后传递到嵌入集合中。

参数：

+   **embedding_module** – 用于查找嵌入的EmbeddingCollection

+   **managed_collision_modules** – 管理冲突模块的字典

+   **return_remapped_features** (*bool*) – 是否返回重新映射的输入特征以及嵌入

```py
forward(features: KeyedJaggedTensor) → Tuple[Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]]¶
```

定义每次调用时执行的计算。

应该被所有子类覆盖。

注意

虽然前向传递的配方需要在此函数内定义，但应该在此之后调用`Module`实例，而不是这个，因为前者负责运行注册的钩子，而后者会默默地忽略它们。

```py
training: bool¶
```

```py
class torchrec.modules.mc_embedding_modules.ManagedCollisionEmbeddingBagCollection(embedding_bag_collection: EmbeddingBagCollection, managed_collision_collection: ManagedCollisionCollection, return_remapped_features: bool = False)¶
```

基类：[`BaseManagedCollisionEmbeddingCollection`](#torchrec.modules.mc_embedding_modules.BaseManagedCollisionEmbeddingCollection "torchrec.modules.mc_embedding_modules.BaseManagedCollisionEmbeddingCollection")

ManagedCollisionEmbeddingBagCollection代表一个EmbeddingBagCollection模块和一组管理的冲突模块。MC-EBC的输入将首先通过管理的冲突模块进行修改，然后传递到嵌入袋集合中。

有关输入和输出类型的详细信息，请参见EmbeddingBagCollection

参数：

+   **embedding_module** – 用于查找嵌入的EmbeddingBagCollection

+   **managed_collision_modules** – 管理冲突模块的字典

+   **return_remapped_features** (*bool*) – 是否返回重新映射的输入特征以及嵌入

```py
training: bool¶
```

```py
class torchrec.modules.mc_embedding_modules.ManagedCollisionEmbeddingCollection(embedding_collection: EmbeddingCollection, managed_collision_collection: ManagedCollisionCollection, return_remapped_features: bool = False)¶
```

基类：[`BaseManagedCollisionEmbeddingCollection`](#torchrec.modules.mc_embedding_modules.BaseManagedCollisionEmbeddingCollection "torchrec.modules.mc_embedding_modules.BaseManagedCollisionEmbeddingCollection")

ManagedCollisionEmbeddingCollection代表一个EmbeddingCollection模块和一组管理的冲突模块。MC-EC的输入将首先通过管理的冲突模块进行修改，然后传递到嵌入集合中。

有关输入和输出类型的详细信息，请参见EmbeddingCollection

参数：

+   **embedding_module** – 用于查找嵌入的EmbeddingCollection

+   **managed_collision_modules** – 管理冲突模块的字典

+   **return_remapped_features** (*bool*) – 是否返回重新映射的输入特征以及嵌入

```py
training: bool¶
```

```py
torchrec.modules.mc_embedding_modules.evict(evictions: Dict[str, Optional[Tensor]], ebc: Union[EmbeddingBagCollection, EmbeddingCollection]) → None¶
```
