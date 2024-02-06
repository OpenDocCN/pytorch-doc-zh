# torchrec.inference

> 原文：[https://pytorch.org/torchrec/torchrec.inference.html](https://pytorch.org/torchrec/torchrec.inference.html)

Torchrec 推理

Torchrec 推理提供了一个基于Torch.Deploy的库，用于GPU推理。

这包括：

+   Python中的模型打包

    +   PredictModule和PredictFactory是Python模型编写和C++模型服务之间的契约。

    +   PredictFactoryPackager可用于使用torch.package打包PredictFactory类。

+   在C++中提供模型服务

    +   BatchingQueue是一个通用的基于配置的请求张量批处理实现。

    +   GPUExecutor处理前向调用，进入Torch.Deploy中的推理模型。

我们实现了如何使用这个库与TorchRec DLRM模型的示例。

+   examples/dlrm/inference/dlrm_packager.py: 这演示了如何将DLRM模型导出为torch.package。

+   examples/dlrm/inference/dlrm_predict.py: 这展示了如何基于现有模型使用PredictModule和PredictFactory。

## torchrec.inference.model_packager[](#torchrec-inference-model-packager "Permalink to this heading")

```py
class torchrec.inference.model_packager.PredictFactoryPackager¶
```

基类：`object`

```py
classmethod save_predict_factory(predict_factory: ~typing.Type[~torchrec.inference.modules.PredictFactory], configs: ~typing.Dict[str, ~typing.Any], output: ~typing.Union[str, ~pathlib.Path, ~typing.BinaryIO], extra_files: ~typing.Dict[str, ~typing.Union[str, bytes]], loader_code: str = '\nimport %PACKAGE%\n\nMODULE_FACTORY=%PACKAGE%.%CLASS%\n', package_importer: ~typing.Union[~torch.package.importer.Importer, ~typing.List[~torch.package.importer.Importer]] = <torch.package.importer._SysImporter object>) → None¶
```

```py
abstract classmethod set_extern_modules()¶
```

指示抽象类方法的装饰器。

已弃用，改用‘classmethod’和‘abstractmethod’。

```py
abstract classmethod set_mocked_modules()¶
```

指示抽象类方法的装饰器。

已弃用，改用‘classmethod’和‘abstractmethod’。

```py
torchrec.inference.model_packager.load_config_text(name: str) → str¶
```

```py
torchrec.inference.model_packager.load_pickle_config(name: str, clazz: Type[T]) → T¶
```

## torchrec.inference.modules[](#torchrec-inference-modules "Permalink to this heading")

```py
class torchrec.inference.modules.BatchingMetadata(type: str, device: str, pinned: List[str])¶
```

基类：`object`

用于批处理的元数据类，这应该与C++定义保持同步。

```py
device: str¶
```

```py
pinned: List[str]¶
```

```py
type: str¶
```

```py
class torchrec.inference.modules.PredictFactory¶
```

基类：`ABC`

创建一个（已学习权重的）模型，用于推理时间。

```py
abstract batching_metadata() → Dict[str, BatchingMetadata]¶
```

返回一个从输入名称到BatchingMetadata的字典。此信息用于输入请求的批处理。

```py
batching_metadata_json() → str¶
```

将批处理元数据序列化为JSON，以便在torch::deploy环境中进行解析。

```py
abstract create_predict_module() → Module¶
```

返回已经分片模型并分配了权重。state_dict()必须匹配TransformModule.transform_state_dict()。它假定torch.distributed.init_process_group已经被调用，并将根据torch.distributed.get_world_size()对模型进行分片。

```py
model_inputs_data() → Dict[str, Any]¶
```

返回一个包含各种数据的字典，用于基准测试输入生成。

```py
qualname_metadata() → Dict[str, QualNameMetadata]¶
```

返回一个从qualname（方法名）到QualNameMetadata的字典。这是执行模型特定方法时的附加信息。

```py
qualname_metadata_json() → str¶
```

将qualname元数据序列化为JSON，以便在torch::deploy环境中进行解析。

```py
abstract result_metadata() → str¶
```

返回表示结果类型的字符串。此信息用于结果拆分。

```py
abstract run_weights_dependent_transformations(predict_module: Module) → Module¶
```

运行依赖于预测模块权重的转换。例如降级到后端。

```py
abstract run_weights_independent_tranformations(predict_module: Module) → Module¶
```

运行不依赖于预测模块权重的转换。例如 fx 追踪，模型拆分等。

```py
class torchrec.inference.modules.PredictModule(module: Module)¶
```

基类：`Module`

模块在基于torch.deploy的后端中工作的接口。用户应该重写predict_forward以将批处理输入格式转换为模块输入格式。

调用参数：

batch: 一个输入张量的字典

返回：

一个输出张量的字典

返回类型：

输出

参数：

+   **module** – 实际的预测模块

+   **device** – 此模块的主要设备，将在前向调用中使用。

示例：

```py
module = PredictModule(torch.device("cuda", torch.cuda.current_device())) 
```

```py
forward(batch: Dict[str, Tensor]) → Any¶
```

定义每次调用执行的计算。

应该被所有子类重写。

注意

虽然前向传递的步骤需要在此函数内定义，但应该在此之后调用`Module`实例，而不是在此处调用，因为前者负责运行已注册的钩子，而后者会默默忽略它们。

```py
abstract predict_forward(batch: Dict[str, Tensor]) → Any¶
```

```py
property predict_module: Module¶
```

```py
state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) → Dict[str, Any]¶
```

返回一个包含模块整个状态引用的字典。

包括参数和持久缓冲区（例如运行平均值）。键是相应的参数和缓冲区名称。设置为`None`的参数和缓冲区不包括在内。

注意

返回的对象是一个浅拷贝。它包含对模块参数和缓冲区的引用。

警告

目前`state_dict()`也接受`destination`，`prefix`和`keep_vars`的位置参数。但是，这将被弃用，并且将在未来版本中强制使用关键字参数。

警告

请避免使用参数`destination`，因为它不是为最终用户设计的。

参数：

+   **destination**（*dict**，*可选*）- 如果提供，模块的状态将更新到字典中，并返回相同的对象。否则，将创建并返回一个`OrderedDict`。默认值：`None`。

+   **prefix**（*str**，*可选*）- 用于组成state_dict中参数和缓冲区名称的键的前缀。默认值：`''`。

+   **keep_vars**（*bool**，*可选*）- 默认情况下，state dict中返回的`Tensor`会从autograd中分离。如果设置为`True`，则不会执行分离。默认值：`False`。

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
training: bool¶
```

```py
class torchrec.inference.modules.QualNameMetadata(need_preproc: bool)¶
```

基类：`object`

```py
need_preproc: bool¶
```

```py
torchrec.inference.modules.quantize_dense(predict_module: PredictModule, dtype: dtype, additional_embedding_module_type: List[Type[Module]] = []) → Module¶
```

```py
torchrec.inference.modules.quantize_embeddings(module: Module, dtype: dtype, inplace: bool, additional_qconfig_spec_keys: Optional[List[Type[Module]]] = None, additional_mapping: Optional[Dict[Type[Module], Type[Module]]] = None, output_dtype: dtype = torch.float32, per_table_weight_dtype: Optional[Dict[str, dtype]] = None) → Module¶
```

```py
torchrec.inference.modules.quantize_feature(module: Module, inputs: Tuple[Tensor, ...]) → Tuple[Tensor, ...]¶
```

```py
torchrec.inference.modules.trim_torch_package_prefix_from_typename(typename: str) → str¶
```  ## 模块内容[]（#module-0“此标题的永久链接”）

Torchrec推理

Torchrec推理提供了一个基于Torch.Deploy的GPU推理库。

这些包括：

+   Python中的模型打包

    +   PredictModule和PredictFactory是Python模型编写和C++模型服务之间的合同。

    +   PredictFactoryPackager可以用于使用torch.package打包PredictFactory类。

+   C++中的模型服务

    +   BatchingQueue是一个基于通用配置的请求张量批处理实现。

    +   GPUExecutor处理Torch.Deploy内部推理模型的前向调用。

我们实现了一个如何使用这个库与TorchRec DLRM模型的示例。

+   示例/dlrm/inference/dlrm_packager.py：这演示了如何将DLRM模型导出为torch.package。

+   示例/dlrm/inference/dlrm_predict.py：这展示了如何基于现有模型使用PredictModule和PredictFactory。
