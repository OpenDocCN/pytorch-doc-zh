# torchrec.models

> 原文：[`pytorch.org/torchrec/torchrec.models.html`](https://pytorch.org/torchrec/torchrec.models.html)

## torchrec.models.deepfm

```py
class torchrec.models.deepfm.DenseArch(in_features: int, hidden_layer_size: int, embedding_dim: int)
```

基类：`Module`

处理 DeepFMNN 模型的密集特征。输出层大小为 EmbeddingBagCollection embeddings 的 embedding_dimension。

参数：

+   **in_features** (*int*) – 密集输入特征的维度。

+   **hidden_layer_size** (*int*) – DenseArch 中隐藏层的大小。

+   **embedding_dim** (*int*) – 与 sparseArch 的 embedding_dimension 相同的大小。

+   **device** (*torch.device*) – 默认计算设备。

示例：

```py
B = 20
D = 3
in_features = 10
dense_arch = DenseArch(
    in_features=in_features, hidden_layer_size=10, embedding_dim=D
)

dense_arch_input = torch.rand((B, in_features))
dense_embedded = dense_arch(dense_arch_input) 
```

```py
forward(features: Tensor) → Tensor
```

参数：

**features** (*torch.Tensor*) – 大小为 B X num_features。

返回：

大小为 B X D 的输出张量。

返回类型：

torch.Tensor

```py
training: bool
```

```py
class torchrec.models.deepfm.FMInteractionArch(fm_in_features: int, sparse_feature_names: List[str], deep_fm_dimension: int)
```

基类：`Module`

处理 SparseArch（sparse_features）和 DenseArch（dense_features）的输出，并根据 DeepFM 论文的外部来源应用一般的 DeepFM 交互：[`arxiv.org/pdf/1703.04247.pdf`](https://arxiv.org/pdf/1703.04247.pdf)

预期的输出维度应为 dense_features，D。

参数：

+   **fm_in_features** (*int*) – DeepFM 中 dense_module 的输入维度。例如，如果输入嵌入是[randn(3, 2, 3), randn(3, 4, 5)]，则 fm_in_features 应为：2 * 3 + 4 * 5。

+   **sparse_feature_names** (*List**[**str**]*) – 长度为 F。

+   **deep_fm_dimension** (*int*) – DeepFM arch 中深度交互（DI）的输出。

示例：

```py
D = 3
B = 10
keys = ["f1", "f2"]
F = len(keys)
fm_inter_arch = FMInteractionArch(sparse_feature_names=keys)
dense_features = torch.rand((B, D))
sparse_features = KeyedTensor(
    keys=keys,
    length_per_key=[D, D],
    values=torch.rand((B, D * F)),
)
cat_fm_output = fm_inter_arch(dense_features, sparse_features) 
```

```py
forward(dense_features: Tensor, sparse_features: KeyedTensor) → Tensor
```

参数：

+   **dense_features** (*torch.Tensor*) – 大小为 B X D 的张量。

+   **sparse_features** (*KeyedJaggedTensor*) – 大小为 F * D X B 的 KJT。

返回：

大小为 B X (D + DI + 1) 的输出张量。

返回类型：

torch.Tensor

```py
training: bool
```

```py
class torchrec.models.deepfm.OverArch(in_features: int)
```

基类：`Module`

最终 Arch - 简单 MLP。输出只是一个目标。

参数：

**in_features** (*int*) – 交互 arch 的输出维度。

示例：

```py
B = 20
over_arch = OverArch()
logits = over_arch(torch.rand((B, 10))) 
```

```py
forward(features: Tensor) → Tensor
```

参数：

**features** (*torch.Tensor*) –

返回：

大小为 B X 1 的输出张量。

返回类型：

torch.Tensor

```py
training: bool
```

```py
class torchrec.models.deepfm.SimpleDeepFMNN(num_dense_features: int, embedding_bag_collection: EmbeddingBagCollection, hidden_layer_size: int, deep_fm_dimension: int)
```

基类：`Module`

具有 DeepFM arch 的基本 recsys 模块。通过学习每个特征的池化嵌入来处理稀疏特征。通过将密集特征投影到相同的嵌入空间中，学习密集特征和稀疏特征之间的关系。通过本文提出的 deep_fm 学习这些密集和稀疏特征之间的交互：[`arxiv.org/pdf/1703.04247.pdf`](https://arxiv.org/pdf/1703.04247.pdf)

该模块假设所有稀疏特征具有相同的嵌入维度（即，每个 EmbeddingBagConfig 使用相同的嵌入维度）

在模型文档中始终使用以下符号：

+   F：稀疏特征的数量

+   D：稀疏特征的 embedding_dimension

+   B：批量大小

+   num_features：密集特征的数量

参数：

+   **num_dense_features** (*int*) – 输入密集特征的数量。

+   **embedding_bag_collection** (*EmbeddingBagCollection*) – 用于定义 SparseArch 的嵌入袋集合。

+   **hidden_layer_size** (*int*) – dense 模块中使用的隐藏层大小。

+   **deep_fm_dimension** (*int*) – deep_fm 的深度交互模块中使用的输出层大小。

示例：

```py
B = 2
D = 8

eb1_config = EmbeddingBagConfig(
    name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
)
eb2_config = EmbeddingBagConfig(
    name="t2",
    embedding_dim=D,
    num_embeddings=100,
    feature_names=["f2"],
)

ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
sparse_nn = SimpleDeepFMNN(
    embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
)

features = torch.rand((B, 100))

#     0       1
# 0   [1,2] [4,5]
# 1   [4,3] [2,9]
# ^
# feature
sparse_features = KeyedJaggedTensor.from_offsets_sync(
    keys=["f1", "f3"],
    values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
    offsets=torch.tensor([0, 2, 4, 6, 8]),
)

logits = sparse_nn(
    dense_features=features,
    sparse_features=sparse_features,
) 
```

```py
forward(dense_features: Tensor, sparse_features: KeyedJaggedTensor) → Tensor
```

参数：

+   **dense_features** (*torch.Tensor*) – 密集特征。

+   **sparse_features** (*KeyedJaggedTensor*) – 稀疏特征。

返回：

大小为 B X 1 的 logits。

返回类型：

torch.Tensor

```py
training: bool
```

```py
class torchrec.models.deepfm.SparseArch(embedding_bag_collection: EmbeddingBagCollection)
```

基类：`Module`

处理 DeepFMNN 模型的稀疏特征。为所有 EmbeddingBag 和每个集合的嵌入特征进行查找。

参数：

**embedding_bag_collection** (*EmbeddingBagCollection*) – 表示一个池化嵌入的集合。

示例：

```py
eb1_config = EmbeddingBagConfig(
    name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
)
eb2_config = EmbeddingBagConfig(
    name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
)

ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

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

sparse_arch(features) 
```

```py
forward(features: KeyedJaggedTensor) → KeyedTensor
```

参数：

**features** (*KeyedJaggedTensor*) –

返回：

一个大小为 F * D X B 的输出 KJT。

返回类型:

KeyedJaggedTensor

```py
training: bool
```

## torchrec.models.dlrm

## 模块内容
