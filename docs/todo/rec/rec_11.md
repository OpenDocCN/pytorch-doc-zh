# torchrec.optim

> 原文：[`pytorch.org/torchrec/torchrec.optim.html`](https://pytorch.org/torchrec/torchrec.optim.html)

Torchrec 优化器

Torchrec 包含一个名为 KeyedOptimizer 的特殊优化器。KeyedOptimizer 公开具有有意义键的 state_dict-它使得可以在原地加载 torch.tensor 和[ShardedTensor](https://github.com/pytorch/pytorch/issues/55207)，并且禁止将空状态加载到已初始化的 KeyedOptimizer 中，反之亦然。

还包括-几个包装 KeyedOptimizer 的模块，称为 CombinedOptimizer 和 OptimizerWrapper-RecSys 中使用的优化器：例如逐行的 adagrad/adam 等

## torchrec.optim.clipping

```py
class torchrec.optim.clipping.GradientClipping(value)
```

基类：`Enum`

一个枚举。

```py
NONE = 'none'
```

```py
NORM = 'norm'
```

```py
VALUE = 'value'
```

```py
class torchrec.optim.clipping.GradientClippingOptimizer(optimizer: KeyedOptimizer, clipping: GradientClipping = GradientClipping.NONE, max_gradient: float = 0.1)
```

基类：`OptimizerWrapper`

在执行优化步骤之前裁剪梯度。

参数：

+   **optimizer** (*KeyedOptimizer*) – 要包装的优化器

+   **clipping** (*GradientClipping*) – 如何裁剪梯度

+   **max_gradient** (*float*) – 裁剪的最大值

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure** (*Callable*) – 重新评估模型并返回损失的闭包。对大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。## torchrec.optim.fused

```py
class torchrec.optim.fused.EmptyFusedOptimizer
```

基类：`FusedOptimizer`

融合优化器类，无操作步骤和无需优化的参数

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure** (*Callable*) – 重新评估模型并返回损失的闭包。对大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
zero_grad(set_to_none: bool = False) → None
```

重置所有优化过的`torch.Tensor`的梯度。

参数：

**set_to_none** (*bool*) – 将梯度设置为 None 而不是零。这通常会减少内存占用，并可能略微提高性能。但是，它会改变某些行为。例如：1. 当用户尝试访问梯度并对其执行手动操作时，具有 None 属性或全为 0 的张量会表现不同。2. 如果用户请求`zero_grad(set_to_none=True)`，然后进行反向传播，对于未接收梯度的参数，`.grad`保证为 None。3. `torch.optim`优化器在梯度为 0 或 None 时具有不同的行为（在一种情况下，它使用梯度为 0 进行步骤，在另一种情况下，它完全跳过步骤）。

```py
class torchrec.optim.fused.FusedOptimizer(params: Mapping[str, Union[Tensor, ShardedTensor]], state: Mapping[Any, Any], param_groups: Collection[Mapping[str, Any]])
```

基类：`KeyedOptimizer`, `ABC`

假设权重更新在反向传播期间完成，因此 step()是一个无操作。

```py
abstract step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure** (*Callable*) – 重新评估模型并返回损失的闭包。对大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
abstract zero_grad(set_to_none: bool = False) → None
```

重置所有优化过的`torch.Tensor`的梯度。

参数：

**set_to_none**（*bool*）-将梯度设置为 None 而不是设置为零。这通常会减少内存占用，并可能略微提高性能。但是，它会改变某些行为。例如：1.当用户尝试访问梯度并对其执行手动操作时，None 属性或一个全为 0 的张量会有不同的行为。2.如果用户请求`zero_grad(set_to_none=True)`，然后进行反向传播，对于未接收梯度的参数，`.grad`保证为 None。3.`torch.optim`优化器在梯度为 0 或 None 时具有不同的行为（在一种情况下，它使用梯度为 0 执行步骤，在另一种情况下，它完全跳过步骤）。

```py
class torchrec.optim.fused.FusedOptimizerModule
```

基类：`ABC`

在反向传播期间执行权重更新的模块。

```py
abstract property fused_optimizer: KeyedOptimizer
```  ## torchrec.optim.keyed

```py
class torchrec.optim.keyed.CombinedOptimizer(optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]])
```

基类：`KeyedOptimizer`

将多个 KeyedOptimizers 组合成一个。

旨在将不同的优化器组合用于不同的子模块

```py
property optimizers: List[Tuple[str, KeyedOptimizer]]
```

```py
property param_groups: Collection[Mapping[str, Any]]
```

```py
property params: Mapping[str, Union[Tensor, ShardedTensor]]
```

```py
post_load_state_dict() → None
```

```py
static prepend_opt_key(name: str, opt_key: str) → str
```

```py
save_param_groups(save: bool) → None
```

```py
property state: Mapping[Tensor, Any]
```

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure**（*Callable*）-重新评估模型并返回损失的闭包。对于大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
zero_grad(set_to_none: bool = False) → None
```

重置所有优化的`torch.Tensor`的梯度。

参数：

**set_to_none**（*bool*）-将梯度设置为 None 而不是设置为零。这通常会减少内存占用，并可能略微提高性能。但是，它会改变某些行为。例如：1.当用户尝试访问梯度并对其执行手动操作时，None 属性或一个全为 0 的张量会有不同的行为。2.如果用户请求`zero_grad(set_to_none=True)`，然后进行反向传播，对于未接收梯度的参数，`.grad`保证为 None。3.`torch.optim`优化器在梯度为 0 或 None 时具有不同的行为（在一种情况下，它使用梯度为 0 执行步骤，在另一种情况下，它完全跳过步骤）。

```py
class torchrec.optim.keyed.KeyedOptimizer(params: Mapping[str, Union[Tensor, ShardedTensor]], state: Mapping[Any, Any], param_groups: Collection[Mapping[str, Any]])
```

基类：`Optimizer`

接受参数字典并按参数键公开 state_dict。

此实现比 torch.Optimizer 中的实现要严格得多：它要求实现在第一次优化迭代期间完全初始化其状态，并禁止将空状态加载到已初始化的 KeyedOptimizer 中，反之亦然。

默认情况下，它也不会在 state_dict()中公开 param_groups。可以通过设置 save_param_groups 标志来切换到旧行为。原因是在分布式训练期间，并非所有参数都存在于所有排名上，我们通过其参数来识别 param_group。此外，param_groups 通常在训练初始化期间重新设置，因此将它们保存为状态的一部分起初没有太多意义。

```py
add_param_group(param_group: Any) → None
```

向`Optimizer`的 param_groups 添加一个参数组。

这在微调预训练网络时可能会很有用，因为冻结的层可以在训练过程中变为可训练，并添加到`Optimizer`中。

参数：

**param_group**（*dict*）-指定应优化的张量以及组特定的优化选项。

```py
init_state(sparse_grad_parameter_names: Optional[Set[str]] = None) → None
```

运行一个虚拟的优化器步骤，允许初始化通常是懒惰的优化器状态。这使我们能够从检查点中就地加载优化器状态。

```py
load_state_dict(state_dict: Mapping[str, Any]) → None
```

此实现比 torch.Optimizer 中的实现要严格得多：它要求实现在第一次优化迭代期间完全初始化其状态，并禁止将空状态加载到已初始化的 KeyedOptimizer 中，反之亦然。

由于引入了严格性，它使我们能够：

+   对状态和 param_groups 进行兼容性检查，以提高可用性

+   通过直接复制到状态张量来避免状态重复，例如 optimizer.step() # 确保优化器已初始化 sd = optimizer.state_dict() load_checkpoint(sd) # 直接将状态复制到张量中，如果需要，重新分片 optimizer.load_state_dict(sd) # 替换 param_groups

```py
post_load_state_dict() → None
```

```py
save_param_groups(save: bool) → None
```

```py
state_dict() → Dict[str, Any]
```

返回的状态和 param_groups 将包含参数键而不是 torch.Optimizer 中的参数索引。这允许实现像优化器重新分片这样的高级功能。

还可以处理遵循 PyTorch 有状态协议的类和支持的数据结构。

```py
class torchrec.optim.keyed.KeyedOptimizerWrapper(params: Mapping[str, Union[Tensor, ShardedTensor]], optim_factory: Callable[[List[Union[Tensor, ShardedTensor]]], Optimizer])
```

基类：`KeyedOptimizer`

接受参数字典并按参数键公开 state_dict。

方便的包装器，接受 optim_factory 可调用以创建 KeyedOptimizer

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure**（*Callable*）- 重新评估模型并返回损失的闭包。对于大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
zero_grad(set_to_none: bool = False) → None
```

重置所有优化的`torch.Tensor`的梯度。

参数：

**set_to_none**（*bool*）- 不设置为零，将梯度设置为 None。这通常会减少内存占用，并可能略微提高性能。但是，它会改变某些行为。例如：1. 当用户尝试访问梯度并对其执行手动操作时，一个 None 属性或一个全为 0 的张量会有不同的行为。2. 如果用户请求`zero_grad(set_to_none=True)`，然后进行反向传播，对于未接收梯度的参数，`.grad`将保证为 None。3. `torch.optim`优化器在梯度为 0 或 None 时具有不同的行为（在一种情况下，它使用梯度为 0 进行步骤，在另一种情况下，它完全跳过步骤）。

```py
class torchrec.optim.keyed.OptimizerWrapper(optimizer: KeyedOptimizer)
```

基类：`KeyedOptimizer`

接受 KeyedOptimizer 并且是 KeyedOptimizer 的包装器

用于像 GradientClippingOptimizer 和 WarmupOptimizer 这样的优化器的子类

```py
add_param_group(param_group: Any) → None
```

向`Optimizer`的 param_groups 添加一个参数组。

当微调预训练网络时，冻结的层可以在训练进行时变为可训练，并添加到`Optimizer`中。

参数：

**param_group**（*dict*）- 指定应优化的张量以及组特定的优化选项。

```py
load_state_dict(state_dict: Mapping[str, Any]) → None
```

此实现比 torch.Optimizer 中的实现严格得多：它要求在第一次优化迭代期间完全初始化其状态，并禁止将空状态加载到已初始化的 KeyedOptimizer 中，反之亦然。

由于引入了严格性，它使我们能够：

+   对状态和 param_groups 进行兼容性检查，以提高可用性

+   通过直接复制到状态张量来避免状态重复，例如 optimizer.step() # 确保优化器已初始化 sd = optimizer.state_dict() load_checkpoint(sd) # 直接将状态复制到张量中，如果需要，重新分片 optimizer.load_state_dict(sd) # 替换 param_groups

```py
post_load_state_dict() → None
```

```py
save_param_groups(save: bool) → None
```

```py
state_dict() → Dict[str, Any]
```

返回的状态和 param_groups 将包含参数键而不是 torch.Optimizer 中的参数索引。这允许实现像优化器重新分片这样的高级功能。

还可以处理遵循 PyTorch 有状态协议的类和支持的数据结构。

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure**（*Callable*）- 重新评估模型并返回损失的闭包。对于大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
zero_grad(set_to_none: bool = False) → None
```

重置所有优化的`torch.Tensor`的梯度。

参数：

**set_to_none**（*bool*）- 将梯度设置为 None 而不是设置为零。这通常具有更低的内存占用，并且可以适度提高性能。但是，它会改变某些行为。例如：1\. 当用户尝试访问梯度并对其执行手动操作时，具有 None 属性或全为 0 的张量会有不同的行为。2\. 如果用户请求`zero_grad(set_to_none=True)`，然后进行反向传播，对于未接收梯度的参数，`.grad`保证为 None。3\. `torch.optim`优化器在梯度为 0 或 None 时具有不同的行为（在一种情况下，它使用梯度为 0 进行步骤，在另一种情况下，它完全跳过步骤）。  ## torchrec.optim.warmup

```py
class torchrec.optim.warmup.WarmupOptimizer(optimizer: KeyedOptimizer, stages: List[WarmupStage], lr: float = 0.1, lr_param: str = 'lr', param_name: str = '__warmup')
```

基类：`OptimizerWrapper`

根据时间表调整学习率。

参数：

+   **optimizer**（*KeyedOptimizer*）- 要包装的优化器

+   **stages**（*List***[*WarmupStage**]*）- 要经过的阶段

+   **lr**（*float*）- 初始学习率

+   **lr_param**（*str*）- 参数组中的学习率参数。

+   **param_name** - 用于保存预热状态的虚拟参数的名称。

```py
post_load_state_dict() → None
```

```py
step(closure: Optional[Any] = None) → None
```

执行单个优化步骤（参数更新）。

参数：

**closure**（*Callable*）- 重新评估模型并返回损失的闭包。对于大多数优化器来说是可选的。

注意

除非另有说明，否则此函数不应修改参数的`.grad`字段。

```py
class torchrec.optim.warmup.WarmupPolicy(value)
```

基类：`Enum`

一个枚举。

```py
CONSTANT = 'constant'
```

```py
INVSQRT = 'inv_sqrt'
```

```py
LINEAR = 'linear'
```

```py
NONE = 'none'
```

```py
POLY = 'poly'
```

```py
STEP = 'step'
```

```py
class torchrec.optim.warmup.WarmupStage(policy: torchrec.optim.warmup.WarmupPolicy = <WarmupPolicy.LINEAR: 'linear'>, max_iters: int = 1, value: float = 1.0, lr_scale: float = 1.0, decay_iters: int = -1)
```

基类：`object`

```py
decay_iters: int = -1
```

```py
lr_scale: float = 1.0
```

```py
max_iters: int = 1
```

```py
policy: WarmupPolicy = 'linear'
```

```py
value: float = 1.0
```  ## 模块内容

Torchrec 优化器

Torchrec 包含一个名为 KeyedOptimizer 的特殊优化器。KeyedOptimizer 公开具有有意义键的 state_dict- 它使得可以在原地加载 torch.tensor 和[ShardedTensor](https://github.com/pytorch/pytorch/issues/55207)，并且禁止将空状态加载到已初始化的 KeyedOptimizer 中，反之亦然。

它还包含 - 几个包装 KeyedOptimizer 的模块，称为 CombinedOptimizer 和 OptimizerWrapper - 用于 RecSys 的优化器：例如逐行 adagrad/adam 等
