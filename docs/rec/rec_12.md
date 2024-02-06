# torchrec.quant

> [https://pytorch.org/torchrec/torchrec.quant.html](https://pytorch.org/torchrec/torchrec.quant.html)

Torchrec量化

Torchrec为推断提供了EmbeddingBagCollection的量化版本。它依赖于fbgemm量化操作。这减少了模型权重的大小并加快了模型执行速度。

示例

```py
>>> import torch.quantization as quant
>>> import torchrec.quant as trec_quant
>>> import torchrec as trec
>>> qconfig = quant.QConfig(
>>>     activation=quant.PlaceholderObserver,
>>>     weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
>>> )
>>> quantized = quant.quantize_dynamic(
>>>     module,
>>>     qconfig_spec={
>>>         trec.EmbeddingBagCollection: qconfig,
>>>     },
>>>     mapping={
>>>         trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
>>>     },
>>>     inplace=inplace,
>>> ) 
```

## torchrec.quant.embedding_modules[](#module-torchrec.quant.embedding_modules "Permalink to this heading")

```py
class torchrec.quant.embedding_modules.EmbeddingBagCollection(tables: List[EmbeddingBagConfig], is_weighted: bool, device: device, output_dtype: dtype = torch.float32, table_name_to_quantized_weights: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None, register_tbes: bool = False, quant_state_dict_split_scale_bias: bool = False, row_alignment: int = 16)¶
```

基础：[`EmbeddingBagCollectionInterface`](torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollectionInterface "torchrec.modules.embedding_modules.EmbeddingBagCollectionInterface"), `ModuleNoCopyMixin`

EmbeddingBagCollection表示池化嵌入（EmbeddingBags）的集合。这个EmbeddingBagCollection被量化为较低的精度。它依赖于fbgemm量化操作并提供表批处理。

它处理形式为 [F X B X L] 的KeyedJaggedTensor的稀疏数据 F: 特征（键） B: 批量大小 L: 稀疏特征的长度（不规则）

并输出形式为 [B * (F * D)] 的KeyedTensor，其中 F: 特征（键） D: 每个特征（键）的嵌入维度 B: 批量大小

参数：

+   **table_name_to_quantized_weights**（*字典**[**str**,* *元组**[**张量**,* *张量**]**]*）- 表到量化权重的映射

+   **embedding_configs**（*列表**[*[*EmbeddingBagConfig*](torchrec.modules.html#torchrec.modules.embedding_configs.EmbeddingBagConfig "torchrec.modules.embedding_configs.EmbeddingBagConfig")*]*）- 嵌入表的列表

+   **is_weighted** - (布尔值)：输入的KeyedJaggedTensor是否加权

+   **设备** - （可选[torch.device]）：默认计算设备

调用参数：

特征：KeyedJaggedTensor，

返回：

KeyedTensor

示例：

```py
table_0 = EmbeddingBagConfig(
    name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
)
table_1 = EmbeddingBagConfig(
    name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
)
ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

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

ebc.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.PlaceholderObserver.with_args(
        dtype=torch.qint8
    ),
    weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
)

qebc = QuantEmbeddingBagCollection.from_float(ebc)
quantized_embeddings = qebc(features) 
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

**特征**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor"））- 形式为 [F X B X L] 的 KJT。

返回：

KeyedTensor

```py
classmethod from_float(module: EmbeddingBagCollection) → EmbeddingBagCollection¶
```

```py
is_weighted() → bool¶
```

```py
output_dtype() → dtype¶
```

```py
training: bool¶
```

```py
class torchrec.quant.embedding_modules.EmbeddingCollection(tables: List[EmbeddingConfig], device: device, need_indices: bool = False, output_dtype: dtype = torch.float32, table_name_to_quantized_weights: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None, register_tbes: bool = False, quant_state_dict_split_scale_bias: bool = False, row_alignment: int = 16)¶
```

基础：[`EmbeddingCollectionInterface`](torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingCollectionInterface "torchrec.modules.embedding_modules.EmbeddingCollectionInterface"), `ModuleNoCopyMixin`

EmbeddingCollection表示非池化嵌入的集合。

它处理形式为 [F X B X L] 的KeyedJaggedTensor的稀疏数据，其中：

+   F: 特征（键）

+   B: 批量大小

+   L: 稀疏特征的长度（可变）

并输出Dict[特征（键），JaggedTensor]。每个JaggedTensor包含形式为 (B * L) X D 的值，其中：

+   B: 批量大小

+   L: 稀疏特征的长度（不规则）

+   D: 每个特征（键）的嵌入维度和长度的形式为 L

参数：

+   **tables**（*列表**[*[*EmbeddingConfig*](torchrec.modules.html#torchrec.modules.embedding_configs.EmbeddingConfig "torchrec.modules.embedding_configs.EmbeddingConfig")*]*）- 嵌入表的列表。

+   **设备**（*可选**[**torch.device**]*）- 默认计算设备。

+   **need_indices**（*布尔值*）- 如果我们需要将索引传递给最终查找结果字典

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

**特征**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor"））- 形式为 [F X B X L] 的 KJT。

返回：

Dict[str, JaggedTensor]

```py
classmethod from_float(module: EmbeddingCollection) → EmbeddingCollection¶
```

```py
need_indices() → bool¶
```

```py
output_dtype() → dtype¶
```

```py
training: bool¶
```

```py
class torchrec.quant.embedding_modules.FeatureProcessedEmbeddingBagCollection(tables: List[EmbeddingBagConfig], is_weighted: bool, device: device, output_dtype: dtype = torch.float32, table_name_to_quantized_weights: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None, register_tbes: bool = False, quant_state_dict_split_scale_bias: bool = False, row_alignment: int = 16, feature_processor: Optional[FeatureProcessorsCollection] = None)¶
```

基础：[`EmbeddingBagCollection`](#torchrec.quant.embedding_modules.EmbeddingBagCollection "torchrec.quant.embedding_modules.EmbeddingBagCollection")

```py
embedding_bags: nn.ModuleDict¶
```

```py
forward(features: KeyedJaggedTensor) → KeyedTensor¶
```

参数：

**特征**（[*KeyedJaggedTensor*](torchrec.sparse.html#torchrec.sparse.jagged_tensor.KeyedJaggedTensor "torchrec.sparse.jagged_tensor.KeyedJaggedTensor"））- 形式为 [F X B X L] 的 KJT。

返回：

KeyedTensor

```py
classmethod from_float(module: FeatureProcessedEmbeddingBagCollection) → FeatureProcessedEmbeddingBagCollection¶
```

```py
tbes: torch.nn.ModuleList¶
```

```py
training: bool¶
```

```py
torchrec.quant.embedding_modules.for_each_module_of_type_do(module: Module, module_types: List[Type[Module]], op: Callable[[Module], None]) → None¶
```

```py
torchrec.quant.embedding_modules.pruned_num_embeddings(pruning_indices_mapping: Tensor) → int¶
```

```py
torchrec.quant.embedding_modules.quant_prep_customize_row_alignment(module: Module, module_types: List[Type[Module]], row_alignment: int) → None¶
```

```py
torchrec.quant.embedding_modules.quant_prep_enable_quant_state_dict_split_scale_bias(module: Module) → None¶
```

```py
torchrec.quant.embedding_modules.quant_prep_enable_quant_state_dict_split_scale_bias_for_types(module: Module, module_types: List[Type[Module]]) → None¶
```

```py
torchrec.quant.embedding_modules.quant_prep_enable_register_tbes(module: Module, module_types: List[Type[Module]]) → None¶
```

```py
torchrec.quant.embedding_modules.quantize_state_dict(module: Module, table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]], table_name_to_data_type: Dict[str, DataType], table_name_to_pruning_indices_mapping: Optional[Dict[str, Tensor]] = None) → device¶
```  ## 模块内容[](#module-0 "Permalink to this heading")

Torchrec量化

Torchrec为推断提供了EmbeddingBagCollection的量化版本。它依赖于fbgemm量化操作。这减少了模型权重的大小并加快了模型执行速度。

示例

```py
>>> import torch.quantization as quant
>>> import torchrec.quant as trec_quant
>>> import torchrec as trec
>>> qconfig = quant.QConfig(
>>>     activation=quant.PlaceholderObserver,
>>>     weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
>>> )
>>> quantized = quant.quantize_dynamic(
>>>     module,
>>>     qconfig_spec={
>>>         trec.EmbeddingBagCollection: qconfig,
>>>     },
>>>     mapping={
>>>         trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
>>>     },
>>>     inplace=inplace,
>>> ) 
```
