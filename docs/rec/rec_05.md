# torchrec.distributed.planner

> 原文：[`pytorch.org/torchrec/torchrec.distributed.planner.html`](https://pytorch.org/torchrec/torchrec.distributed.planner.html)
>
> 译者：[飞龙](https://github.com/wizardforcel)
>
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


Torchrec 规划者

规划者提供了模块被分片所需的规格，考虑了构建优化计划的可能选项。

该功能包括：

+   生成所有可能的分片选项。

+   为每个分片估算性能和存储。

+   估算峰值内存使用量，以消除可能导致 OOM 的分片计划。

+   参数约束、分区、提议者或性能建模的可定制性。

+   自动构建和选择优化的分片计划。

## torchrec.distributed.planner.constants

```py
torchrec.distributed.planner.constants.kernel_bw_lookup(compute_device: str, compute_kernel: str, hbm_mem_bw: float, ddr_mem_bw: float, caching_ratio: Optional[float] = None, prefetch_pipeline: bool = False) → Optional[float]
```

根据给定的计算设备、计算内核和缓存比率计算设备带宽。

参数：

+   **compute_kernel** (*str*) – 计算内核。

+   **compute_device** (*str*) – 计算设备。

+   **hbm_mem_bw** (*float*) – 设备 HBM 的带宽。

+   **ddr_mem_bw** (*float*) – 系统 DDR 内存的带宽。

+   **caching_ratio** (*Optional**[**float**]*) – 用于确定设备带宽的缓存比率，如果启用了 UVM 缓存。

+   **prefetch_pipeline** (*bool*) – 是否启用预取管道。

返回：

设备带宽。

返回类型：

可选[float]  ## torchrec.distributed.planner.enumerators

```py
class torchrec.distributed.planner.enumerators.EmbeddingEnumerator(topology: Topology, batch_size: int, constraints: Optional[Dict[str, ParameterConstraints]] = None, estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None)
```

基类：`Enumerator`

为给定的 nn.Module 生成嵌入分片选项，考虑用户提供的约束。

参数：

+   **topology** (*Topology*) – 设备拓扑。

+   **batch_size** (*int*) – 批量大小。

+   **constraints** (*Optional****Dict**[**str**,* [*ParameterConstraints**]**]*) – 参数名称到提供的 ParameterConstraints 的字典。

```py
enumerate(module: Module, sharders: List[ModuleSharder[Module]]) → List[ShardingOption]
```

生成给定模块和分片器的相关分片选项。

参数：

+   **module** (*nn.Module*) – 要分片的模块。

+   **sharders** (*List***[*ModuleSharder**[**nn.Module**]**]*) – 模块的提供的分片器。

返回：

填充值的有效分片选项。

返回类型：

List[ShardingOption]

```py
populate_estimates(sharding_options: List[ShardingOption]) → None
```

查看类描述。

```py
torchrec.distributed.planner.enumerators.get_partition_by_type(sharding_type: str) → str
```

根据提供的分片类型获取相应的分区。

参数：

**sharding_type** (*str*) – 分片类型字符串。

返回：

相应的 PartitionByType 值。

返回类型：

str  ## torchrec.distributed.planner.partitioners

```py
class torchrec.distributed.planner.partitioners.GreedyPerfPartitioner(sort_by: SortBy = SortBy.STORAGE, balance_modules: bool = False)
```

基类：`Partitioner`

贪婪分区器

参数：

+   **sort_by** (*SortBy*) – 按存储或性能降序排序分片选项（即，大表将首先放置）。

+   **balance_modules** (*bool*) – 是否首先按模块排序，其中较小的模块将首先排序。实际上，这将以平衡的方式在每个模块中放置表。

```py
partition(proposal: List[ShardingOption], storage_constraint: Topology) → List[ShardingOption]
```

根据每个分片选项的 partition_by 属性将分片选项放置在拓扑上。在放置结束时，拓扑、存储和性能将被更新。

参数：

+   **proposal** (*List***[*ShardingOption**]*) – 填充的分片选项列表。

+   **storage_constraint**（*Topology*）-设备拓扑。

返回：

所选计划的分片选项列表。

返回类型：

List[ShardingOption]

示例：

```py
sharding_options = [
        ShardingOption(partition_by="uniform",
                shards=[
                    Shards(storage=1, perf=1),
                    Shards(storage=1, perf=1),
                ]),
        ShardingOption(partition_by="uniform",
                shards=[
                    Shards(storage=2, perf=2),
                    Shards(storage=2, perf=2),
                ]),
        ShardingOption(partition_by="device",
                shards=[
                    Shards(storage=3, perf=3),
                    Shards(storage=3, perf=3),
                ])
        ShardingOption(partition_by="device",
                shards=[
                    Shards(storage=4, perf=4),
                    Shards(storage=4, perf=4),
                ]),
    ]
topology = Topology(world_size=2)

# First [sharding_options[0] and sharding_options[1]] will be placed on the
# topology with the uniform strategy, resulting in

topology.devices[0].perf.total = (1,2)
topology.devices[1].perf.total = (1,2)

# Finally sharding_options[2] and sharding_options[3]] will be placed on the
# topology with the device strategy (see docstring of `partition_by_device` for
# more details).

topology.devices[0].perf.total = (1,2) + (3,4)
topology.devices[1].perf.total = (1,2) + (3,4)

# The topology updates are done after the end of all the placements (the other
# in the example is just for clarity). 
```

```py
class torchrec.distributed.planner.partitioners.MemoryBalancedPartitioner(max_search_count: int = 10, tolerance: float = 0.02, balance_modules: bool = False)
```

基类：`Partitioner`

内存平衡分区器。

参数：

+   **max_search_count**（*整数*）-调用 GreedyPartitioner 的最大次数。

+   **容差**（*浮点数*）-原始计划和新计划之间的最大可接受差异。如果容差为 1，这意味着如果新计划的性能是原始计划的 200％（即计划比原计划差 100％），则将拒绝新计划。

+   **balance_modules**（*布尔值*）-是否首先按模块排序，其中较小的模块将首先排序。实际上，这将以平衡的方式放置每个模块中的表。

```py
partition(proposal: List[ShardingOption], storage_constraint: Topology) → List[ShardingOption]
```

重复调用 GreedyPerfPartitioner，以找到性能在原始计划容差范围内且使用最少内存量的计划。

```py
class torchrec.distributed.planner.partitioners.OrderedDeviceHardware(device: torchrec.distributed.planner.types.DeviceHardware, local_world_size: int)
```

基类：`object`

```py
device: DeviceHardware
```

```py
local_world_size: int
```

```py
class torchrec.distributed.planner.partitioners.ShardingOptionGroup(sharding_options: List[torchrec.distributed.planner.types.ShardingOption], storage_sum: torchrec.distributed.planner.types.Storage, perf_sum: float, param_count: int)
```

基类：`object`

```py
param_count: int
```

```py
perf_sum: float
```

```py
sharding_options: List[ShardingOption]
```

```py
storage_sum: Storage
```

```py
class torchrec.distributed.planner.partitioners.SortBy(value)
```

基类：`Enum`

一个枚举。

```py
PERF = 'perf'
```

```py
STORAGE = 'storage'
```

```py
torchrec.distributed.planner.partitioners.set_hbm_per_device(storage_constraint: Topology, hbm_per_device: int) → None
```  ## torchrec.distributed.planner.perf_models

```py
class torchrec.distributed.planner.perf_models.NoopPerfModel(topology: Topology)
```

基类：`PerfModel`

```py
rate(plan: List[ShardingOption]) → float
```  ## torchrec.distributed.planner.planners

```py
class torchrec.distributed.planner.planners.EmbeddingShardingPlanner(topology: Optional[Topology] = None, batch_size: Optional[int] = None, enumerator: Optional[Enumerator] = None, storage_reservation: Optional[StorageReservation] = None, proposer: Optional[Union[Proposer, List[Proposer]]] = None, partitioner: Optional[Partitioner] = None, performance_model: Optional[PerfModel] = None, stats: Optional[Union[Stats, List[Stats]]] = None, constraints: Optional[Dict[str, ParameterConstraints]] = None, debug: bool = True)
```

基类：`ShardingPlanner`

根据提供的分片器、拓扑和约束为给定模块提供优化的分片计划。

```py
collective_plan(module: Module, sharders: Optional[List[ModuleSharder[Module]]] = None, pg: Optional[ProcessGroup] = None) → ShardingPlan
```

在 rank 0 上调用 self.plan(…)并广播

```py
plan(module: Module, sharders: List[ModuleSharder[Module]]) → ShardingPlan
```

为提供的模块和给定的分片器进行分片。

参数：

+   **module**（*nn.Module*）-计划分片的模块。

+   **sharders**（*List***[*ModuleSharder**[**nn.Module**]**]*)-模块的提供的分片器。

返回：

计算得到的分片计划。

返回类型：

ShardingPlan  ## torchrec.distributed.planner.proposers

```py
class torchrec.distributed.planner.proposers.EmbeddingOffloadScaleupProposer(use_depth: bool = True)
```

基类：`Proposer`

```py
static allocate_budget(model: Tensor, clfs: Tensor, budget: int, allocation_priority: Tensor) → Tensor
```

```py
static build_affine_storage_model(uvm_caching_sharding_options: List[ShardingOption], enumerator: Enumerator) → Tensor
```

```py
static clf_to_bytes(model: Tensor, clfs: Union[float, Tensor]) → Tensor
```

```py
feedback(partitionable: bool, plan: Optional[List[ShardingOption]] = None, perf_rating: Optional[float] = None, storage_constraint: Optional[Topology] = None) → None
```

```py
static get_budget(proposal: List[ShardingOption], storage_constraint: Topology) → int
```

返回额外的 HBM 预算，可用于 GPU 缓存。

```py
static get_cacheability(sharding_option: ShardingOption) → Optional[float]
```

```py
static get_expected_lookups(sharding_option: ShardingOption) → Optional[float]
```

```py
load(search_space: List[ShardingOption], enumerator: Optional[Enumerator] = None) → None
```

```py
static next_plan(starting_proposal: List[ShardingOption], budget: Optional[int], enumerator: Optional[Enumerator]) → Optional[List[ShardingOption]]
```

```py
propose() → Optional[List[ShardingOption]]
```

```py
class torchrec.distributed.planner.proposers.GreedyProposer(use_depth: bool = True, threshold: Optional[int] = None)
```

基类：`Proposer`

以贪婪的方式提出分片计划。

按性能对每个可分片参数的分片选项进行排序。在每次迭代中，找到当前存储使用量最大的参数，并尝试其下一个分片选项。

参数：

+   **use_depth**（*布尔值*）-启用时，根据 max(shard.perf.total)对 fqn 的 sharding_options 进行排序，否则根据 sum(shard.perf.total)对 sharding_options 进行排序。

+   **threshold**（*可选**[**整数**]**）-提前停止的阈值。当指定时，当提议的性能连续比最佳性能差时，提议者停止提议。

```py
feedback(partitionable: bool, plan: Optional[List[ShardingOption]] = None, perf_rating: Optional[float] = None, storage_constraint: Optional[Topology] = None) → None
```

```py
load(search_space: List[ShardingOption], enumerator: Optional[Enumerator] = None) → None
```

```py
propose() → Optional[List[ShardingOption]]
```

```py
class torchrec.distributed.planner.proposers.GridSearchProposer(max_proposals: int = 10000)
```

基类：`Proposer`

```py
feedback(partitionable: bool, plan: Optional[List[ShardingOption]] = None, perf_rating: Optional[float] = None, storage_constraint: Optional[Topology] = None) → None
```

```py
load(search_space: List[ShardingOption], enumerator: Optional[Enumerator] = None) → None
```

```py
propose() → Optional[List[ShardingOption]]
```

```py
class torchrec.distributed.planner.proposers.UniformProposer(use_depth: bool = True)
```

基类：`Proposer`

提出统一的分片计划，即所有分片选项都具有相同的分片类型的计划。

```py
feedback(partitionable: bool, plan: Optional[List[ShardingOption]] = None, perf_rating: Optional[float] = None, storage_constraint: Optional[Topology] = None) → None
```

```py
load(search_space: List[ShardingOption], enumerator: Optional[Enumerator] = None) → None
```

```py
propose() → Optional[List[ShardingOption]]
```

```py
torchrec.distributed.planner.proposers.proposers_to_proposals_list(proposers_list: List[Proposer], search_space: List[ShardingOption]) → List[List[ShardingOption]]
```

仅适用于静态反馈提议者（要检查的提议路径与提议的性能无关）## torchrec.distributed.planner.shard_estimators

```py
class torchrec.distributed.planner.shard_estimators.EmbeddingOffloadStats(cacheability: float, expected_lookups: int, mrc_hist_counts: Tensor, height: int)
```

基类：`CacheStatistics`

为 uvm_fused_cache 表计算缓存统计信息。

参数：

cachebility (float):

未命中率曲线的曲线下面积。

expected_lookups (float):

全局批次中预期的唯一嵌入 id 数量。

mrc_hist_counts (torch.Tensor):

一个 1 维张量（大小为 n），保存 LRU 未命中率曲线的直方图。每个 bin 代表可能的 LRU 缓存大小的 1/n（从 load_factor 0 到 load_factor 1.0）。如果 LRU load_factor 至少为该大小，则 bin 包含可以处理的预期 LRU 操作数量，而不会发生缓存未命中。

高度（int）：

嵌入表的高度（num_embeddings）。

```py
property cacheability: float
```

缓存数据集的难度的总结度量，独立于缓存大小。得分为 0 表示数据集非常适合缓存（例如，访问之间的局部性很高），得分为 1 表示非常难以缓存。

```py
static estimate_cache_miss_rate(cache_sizes: Tensor, hist: Tensor, bins: Tensor) → Tensor
```

根据提议的 cache_sizes 计算给定 MRC 直方图的估计缓存未命中率。

```py
property expected_lookups: int
```

每个训练步骤的预期缓存查找次数。

这是全局训练批次中预期的不同值的数量。

```py
expected_miss_rate(clf: float) → float
```

给定缓存大小的预期缓存查找未命中率。

当 clf（缓存加载因子）为 0 时，返回 1.0（100%未命中）。当 clf 为 1.0 时，返回 0（100%命中）。对于介于这些极端之间的 clf 值，根据对训练数据集的统计属性的了解，返回缓存的估计未命中率。

```py
class torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator(topology: Topology, constraints: Optional[Dict[str, ParameterConstraints]] = None, is_inference: bool = False)
```

基类：`ShardEstimator`

嵌入墙时间性能估计器

```py
estimate(sharding_options: List[ShardingOption], sharder_map: Optional[Dict[str, ModuleSharder[Module]]] = None) → None
```

```py
class torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator(topology: Topology, constraints: Optional[Dict[str, ParameterConstraints]] = None)
```

基类：`ShardEstimator`

嵌入存储使用量估计器

```py
estimate(sharding_options: List[ShardingOption], sharder_map: Optional[Dict[str, ModuleSharder[Module]]] = None) → None
```

```py
torchrec.distributed.planner.shard_estimators.calculate_shard_storages(sharder: ModuleSharder[Module], sharding_type: str, tensor: Tensor, compute_device: str, compute_kernel: str, shard_sizes: List[List[int]], batch_sizes: List[int], world_size: int, local_world_size: int, input_lengths: List[float], num_poolings: List[float], caching_ratio: float, is_pooled: bool) → List[Storage]
```

计算每个分片张量的估计存储大小，包括输入、输出、张量、梯度和优化器大小。

参数：

+   **sharder** (*ModuleSharder**[**nn.Module**]*) – 支持分片的模块的分片器。

+   **sharding_type** (*str*) – 提供的 ShardingType 值。

+   **tensor** (*torch.Tensor*) – 要分片的张量。

+   **compute_device** (*str*) – 要使用的计算设备。

+   **compute_kernel** (*str*) – 要使用的计算内核。

+   **shard_sizes** (*List**[**List**[**int**]**]*) – 每个分片张量的维度列表。

+   **batch_sizes** (*List**[**int**]*) – 每个输入特征的批次大小。

+   **world_size** (*int*) – 拓扑中的设备总数。

+   **local_world_size** (*int*) – 主机组拓扑中的设备总数。

+   **input_lengths** (*List**[**float**]*) – 平均输入长度，与池化因子相同。

+   **num_poolings** (*List**[**float**]*) – 每个样本的平均池化次数（通常为 1.0）。

+   **caching_ratio** (*float*) – HBM 到 DDR 内存的 UVM 缓存比率。

+   **is_pooled** (*bool*) – 如果嵌入输出是池化的（即 EmbeddingBag），则为 True，如果是未池化/顺序的（即 Embedding），则为 False。

返回：

拓扑中每个设备的存储对象。

返回类型：

List[Storage]

```py
torchrec.distributed.planner.shard_estimators.perf_func_emb_wall_time(shard_sizes: List[List[int]], compute_kernel: str, compute_device: str, sharding_type: str, batch_sizes: List[int], world_size: int, local_world_size: int, input_lengths: List[float], input_data_type_size: float, table_data_type_size: float, fwd_a2a_comm_data_type_size: float, bwd_a2a_comm_data_type_size: float, fwd_sr_comm_data_type_size: float, bwd_sr_comm_data_type_size: float, num_poolings: List[float], hbm_mem_bw: float, ddr_mem_bw: float, intra_host_bw: float, inter_host_bw: float, bwd_compute_multiplier: float, is_pooled: bool, is_weighted: bool = False, caching_ratio: Optional[float] = None, is_inference: bool = False, prefetch_pipeline: bool = False, expected_cache_fetches: float = 0) → List[Perf]
```

尝试将性能建模为相对墙时间的函数。

参数：

+   **shard_sizes** (*List**[**List**[**int**]**]*) – 每个分片的（local_rows, local_cols）列表。

+   **compute_kernel** (*str*) – 计算内核。

+   **compute_device** (*str*) – 计算设备。

+   **sharding_type** (*str*) – tw, rw, cw, twrw, dp。

+   **batch_sizes** (*List**[**int**]*) – 每个输入特征的批次大小。

+   **world_size** (*int*) – 所有主机的设备数量。

+   **local_world_size** (*int*) – 每个主机的设备数量。

+   **input_lengths** (*List**[**float**]*) – 每个输入查询特征的平均查找次数列表。

+   **input_data_type_size** (*float*) – 分布式数据并行输入的数据类型大小。

+   **table_data_type_size** (*float*) – 表格的数据类型大小。

+   **fwd_comm_data_type_size** (*float*) – 正向通信期间分布式数据并行输入的数据类型大小。

+   **bwd_comm_data_type_size** (*float*) – 反向通信期间分布式数据并行输入的数据类型大小。

+   **num_poolings** (*List**[**float**]*) – 每个样本的池化次数，通常为 1.0。

+   **hbm_mem_bw** (*float*) – 设备 HBM 的带宽。

+   **ddr_mem_bw** (*float*) – 系统 DDR 内存的带宽。

+   **intra_host_bw** (*float*) – 单个主机内的带宽，如多个线程。

+   **inter_host_bw** (*float*) – 两个主机之间的带宽，如多台机器。

+   **is_pooled** (*bool*) – 如果嵌入输出是池化的（即 EmbeddingBag），则为 True；如果未池化/顺序（即 Embedding），则为 False。

+   **is_weighted** (*bool = False*) – 如果模块是 EBC 并且是加权的，则为 True，通常表示 id 分数列表特征。

+   **is_inference** (*bool = False*) – 是否为推断进行规划。

+   **caching_ratio** (*Optional**[**float**]* *= None*) – 缓存比例，用于确定设备的带宽。

+   **prefetch_pipeline** (*bool = False*) – 是否启用预取管道。

+   **expected_cache_fetches** (*float*) – 全局批次中预期的缓存获取次数

返回：

每个分片的性能列表。

返回类型：

List[float]  ## torchrec.distributed.planner.stats

```py
class torchrec.distributed.planner.stats.EmbeddingStats
```

基类：`Stats`

用于分片规划执行的统计信息。

```py
log(sharding_plan: ShardingPlan, topology: Topology, batch_size: int, storage_reservation: StorageReservation, num_proposals: int, num_plans: int, run_time: float, best_plan: List[ShardingOption], constraints: Optional[Dict[str, ParameterConstraints]] = None, sharders: Optional[List[ModuleSharder[Module]]] = None, debug: bool = True) → None
```

记录给定分片计划的统计信息。

提供给定分片计划的每个设备存储使用情况（HBM 和 DDR）、性能、输入、输出和分片数量/类型的统计表格视图。

参数：

+   **sharding_plan** (*ShardingPlan*) – 规划者选择的分片计划。

+   **topology** (*Topology*) – 设备拓扑结构。

+   **batch_size** (*int*) – 批次大小。

+   **storage_reservation** (*StorageReservation*) – 为模型的未分片部分保留存储空间

+   **num_proposals** (*int*) – 评估的提案数量。

+   **num_plans** (*int*) – 成功分区的提案数量。

+   **run_time** (*float*) – 找到计划所需的时间（以秒为单位）。

+   **best_plan** (*List***[*ShardingOption**]*) – 预期性能的计划。

+   **constraints** (*Optional****Dict**[**str**,* [*ParameterConstraints**]**]*) – 参数名称到提供的参数约束的字典。

+   **debug** (*bool*) – 是否启用调试模式。

```py
class torchrec.distributed.planner.stats.NoopEmbeddingStats
```

基类：`Stats`

用于分片规划执行的 Noop 统计。

```py
log(sharding_plan: ShardingPlan, topology: Topology, batch_size: int, storage_reservation: StorageReservation, num_proposals: int, num_plans: int, run_time: float, best_plan: List[ShardingOption], constraints: Optional[Dict[str, ParameterConstraints]] = None, sharders: Optional[List[ModuleSharder[Module]]] = None, debug: bool = True) → None
```

查看类描述

```py
torchrec.distributed.planner.stats.round_to_one_sigfig(x: float) → str
```  ## torchrec.distributed.planner.storage_reservations

```py
class torchrec.distributed.planner.storage_reservations.FixedPercentageStorageReservation(percentage: float)
```

基类：`StorageReservation`

```py
reserve(topology: Topology, batch_size: int, module: Module, sharders: List[ModuleSharder[Module]], constraints: Optional[Dict[str, ParameterConstraints]] = None) → Topology
```

```py
class torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation(percentage: float, parameter_multiplier: float = 6.0, dense_tensor_estimate: Optional[int] = None)
```

基类：`StorageReservation`

为要分片的模型保留存储空间，使用启发式计算。存储保留包括稠密张量存储、KJT 存储和总存储的额外百分比。

参数：

+   **百分比**（*浮点数*）- 额外的存储百分比，作为启发式计算存储空间之外的误差边界。

+   参数乘数（*浮点数*）- 用于总参数存储的启发式乘数。

+   **dense_tensor_estimate**（*可选**[**int**]*）- 稠密张量的存储估计，如果未提供，则使用默认的启发式估计。

```py
reserve(topology: Topology, batch_size: int, module: Module, sharders: List[ModuleSharder[Module]], constraints: Optional[Dict[str, ParameterConstraints]] = None) → Topology
```

```py
class torchrec.distributed.planner.storage_reservations.InferenceStorageReservation(percentage: float, dense_tensor_estimate: Optional[int] = None)
```

基类：`StorageReservation`

为要分片进行推理而保留存储空间。存储保留包括稠密张量存储、KJT 存储和总存储的额外百分比。请注意，在估算存储空间时，假定稠密模块位于 GPU 上，并在各个秩上复制。如果不是这种情况，请使用 dense_tensor_estimate 覆盖估算。

参数：

+   **百分比**（*浮点数*）- 保留的额外存储百分比，作为存储计算之外的误差边界。

+   **dense_tensor_estimate**（*可选**[**int**]*）- 稠密张量的存储估计，如果未提供，则使用默认的启发式估计。

```py
reserve(topology: Topology, batch_size: int, module: Module, sharders: List[ModuleSharder[Module]], constraints: Optional[Dict[str, ParameterConstraints]] = None) → Topology
```  ## torchrec.distributed.planner.types

```py
class torchrec.distributed.planner.types.DeviceHardware(rank: int, storage: Storage, perf: Perf)
```

基类：`object`

表示进程组中设备的存储容量。‘perf’是网络、CPU 和存储使用的估计。

```py
perf: Perf
```

```py
rank: int
```

```py
storage: Storage
```

```py
class torchrec.distributed.planner.types.Enumerator(topology: Topology, batch_size: int = 512, constraints: Optional[Dict[str, ParameterConstraints]] = None, estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None)
```

基类：`ABC`

为给定拓扑、约束、nn.Module 和分片器生成所有相关的分片选项。

```py
abstract enumerate(module: Module, sharders: List[ModuleSharder[Module]]) → List[ShardingOption]
```

查看类描述。

```py
abstract populate_estimates(sharding_options: List[ShardingOption]) → None
```

查看类描述。

```py
class torchrec.distributed.planner.types.ParameterConstraints(sharding_types: ~typing.Optional[~typing.List[str]] = None, compute_kernels: ~typing.Optional[~typing.List[str]] = None, min_partition: ~typing.Optional[int] = None, pooling_factors: ~typing.List[float] = <factory>, num_poolings: ~typing.Optional[~typing.List[float]] = None, batch_sizes: ~typing.Optional[~typing.List[int]] = None, is_weighted: bool = False, cache_params: ~typing.Optional[~torchrec.distributed.types.CacheParams] = None, enforce_hbm: ~typing.Optional[bool] = None, stochastic_rounding: ~typing.Optional[bool] = None, bounds_check_mode: ~typing.Optional[~fbgemm_gpu.split_table_batched_embeddings_ops_common.BoundsCheckMode] = None)
```

基类：`object`

存储用户提供的关于分片计划的约束。

如果提供了 pooling_factors、num_poolings 和 batch_sizes，必须与样本中的长度匹配。

```py
batch_sizes: Optional[List[int]] = None
```

```py
bounds_check_mode: Optional[BoundsCheckMode] = None
```

```py
cache_params: Optional[CacheParams] = None
```

```py
compute_kernels: Optional[List[str]] = None
```

```py
enforce_hbm: Optional[bool] = None
```

```py
is_weighted: bool = False
```

```py
min_partition: Optional[int] = None
```

```py
num_poolings: Optional[List[float]] = None
```

```py
pooling_factors: List[float]
```

```py
sharding_types: Optional[List[str]] = None
```

```py
stochastic_rounding: Optional[bool] = None
```

```py
class torchrec.distributed.planner.types.PartitionByType(value)
```

基类：`Enum`

众所周知的分区类型。

```py
DEVICE = 'device'
```

```py
HOST = 'host'
```

```py
UNIFORM = 'uniform'
```

```py
class torchrec.distributed.planner.types.Partitioner
```

基类：`ABC`

分区分片。

今天我们有多种策略，即（贪婪、BLDM、线性）。

```py
abstract partition(proposal: List[ShardingOption], storage_constraint: Topology) → List[ShardingOption]
```

```py
class torchrec.distributed.planner.types.Perf(fwd_compute: float, fwd_comms: float, bwd_compute: float, bwd_comms: float, prefetch_compute: float = 0.0)
```

基类：`object`

表示单个嵌入表分片的性能估计的细分。

```py
bwd_comms: float
```

```py
bwd_compute: float
```

```py
fwd_comms: float
```

```py
fwd_compute: float
```

```py
prefetch_compute: float = 0.0
```

```py
property total: float
```

```py
class torchrec.distributed.planner.types.PerfModel
```

基类：`ABC`

```py
abstract rate(plan: List[ShardingOption]) → float
```

```py
exception torchrec.distributed.planner.types.PlannerError(message: str, error_type: PlannerErrorType = PlannerErrorType.OTHER)
```

基类：`Exception`

```py
class torchrec.distributed.planner.types.PlannerErrorType(value)
```

基类：`Enum`

根据以下情况分类 PlannerError。

```py
INSUFFICIENT_STORAGE = 'insufficient_storage'
```

```py
OTHER = 'other'
```

```py
PARTITION = 'partition'
```

```py
STRICT_CONSTRAINTS = 'strict_constraints'
```

```py
class torchrec.distributed.planner.types.Proposer
```

基类：`ABC`

提出可以分区生成计划的分片选项的完整列表。

```py
abstract feedback(partitionable: bool, plan: Optional[List[ShardingOption]] = None, perf_rating: Optional[float] = None, storage_constraint: Optional[Topology] = None) → None
```

```py
abstract load(search_space: List[ShardingOption], enumerator: Optional[Enumerator] = None) → None
```

```py
abstract propose() → Optional[List[ShardingOption]]
```

```py
class torchrec.distributed.planner.types.Shard(size: List[int], offset: List[int], storage: Optional[Storage] = None, perf: Optional[Perf] = None, rank: Optional[int] = None)
```

基类：`object`

表示嵌入表的子集。‘size’和‘offset’完全确定了分片中的张量。‘storage’是存储分片所需的估计‘perf’。

```py
offset: List[int]
```

```py
perf: Optional[Perf] = None
```

```py
rank: Optional[int] = None
```

```py
size: List[int]
```

```py
storage: Optional[Storage] = None
```

```py
class torchrec.distributed.planner.types.ShardEstimator(topology: Topology, constraints: Optional[Dict[str, ParameterConstraints]] = None)
```

基类：`ABC`

估算分片的性能或存储，需要完全指定的分片选项。

```py
abstract estimate(sharding_options: List[ShardingOption], sharder_map: Optional[Dict[str, ModuleSharder[Module]]] = None) → None
```

```py
class torchrec.distributed.planner.types.ShardingOption(name: str, tensor: Tensor, module: Tuple[str, Module], input_lengths: List[float], batch_size: int, sharding_type: str, partition_by: str, compute_kernel: str, shards: List[Shard], cache_params: Optional[CacheParams] = None, enforce_hbm: Optional[bool] = None, stochastic_rounding: Optional[bool] = None, bounds_check_mode: Optional[BoundsCheckMode] = None, dependency: Optional[str] = None, is_pooled: Optional[bool] = None)
```

基类：`object`

分片嵌入表的一种方式。

```py
property cache_load_factor: Optional[float]
```

```py
property fqn: str
```

```py
property is_pooled: bool
```

```py
property module: Tuple[str, Module]
```

```py
static module_pooled(module: Module, sharding_option_name: str) → bool
```

确定模块是否池化输出（例如 EmbeddingBag）或使用未池化/顺序输出。

```py
property num_inputs: int
```

```py
property num_shards: int
```

```py
property path: str
```

```py
property tensor: Tensor
```

```py
property total_perf: float
```

```py
property total_storage: Storage
```

```py
class torchrec.distributed.planner.types.Stats
```

基类：`ABC`

记录与分片计划相关的统计信息。

```py
abstract log(sharding_plan: ShardingPlan, topology: Topology, batch_size: int, storage_reservation: StorageReservation, num_proposals: int, num_plans: int, run_time: float, best_plan: List[ShardingOption], constraints: Optional[Dict[str, ParameterConstraints]] = None, sharders: Optional[List[ModuleSharder[Module]]] = None, debug: bool = False) → None
```

查看类描述

```py
class torchrec.distributed.planner.types.Storage(hbm: int, ddr: int)
```

基类：`object`

表示用于训练的硬件的存储容量。

```py
ddr: int
```

```py
fits_in(other: Storage) → bool
```

```py
hbm: int
```

```py
class torchrec.distributed.planner.types.StorageReservation
```

基类：`ABC`

为模型的非分片部分保留存储空间。

```py
abstract reserve(topology: Topology, batch_size: int, module: Module, sharders: List[ModuleSharder[Module]], constraints: Optional[Dict[str, ParameterConstraints]] = None) → Topology
```

```py
class torchrec.distributed.planner.types.Topology(world_size: int, compute_device: str, hbm_cap: Optional[int] = None, ddr_cap: Optional[int] = None, local_world_size: Optional[int] = None, hbm_mem_bw: float = 963146416.128, ddr_mem_bw: float = 54760833.024, intra_host_bw: float = 644245094.4, inter_host_bw: float = 13421772.8, bwd_compute_multiplier: float = 2)
```

基类：`object`

```py
property bwd_compute_multiplier: float
```

```py
property compute_device: str
```

```py
property ddr_mem_bw: float
```

```py
property devices: List[DeviceHardware]
```

```py
property hbm_mem_bw: float
```

```py
property inter_host_bw: float
```

```py
property intra_host_bw: float
```

```py
property local_world_size: int
```

```py
property world_size: int
```  ## torchrec.distributed.planner.utils

```py
class torchrec.distributed.planner.utils.BinarySearchPredicate(A: int, B: int, tolerance: int)
```

基类：`object`

生成 X 在 A 和 B 之间的值，以调用外部谓词 F(X)以发现 F(X)为真的最大 X。使用二进制搜索来最小化对 F 的调用次数。假设 F 是一个阶跃函数，即如果 F(X)为假，则没有必要尝试 F(X+1)。

```py
next(prior_result: bool) → Optional[int]
```

next()返回下一个要探测的值，给定先前探测的结果。第一次调用 next()时，忽略 prior_result。如果探索整个范围或达到阈值，则返回 None。

```py
torchrec.distributed.planner.utils.bytes_to_gb(num_bytes: int) → float
```

```py
torchrec.distributed.planner.utils.bytes_to_mb(num_bytes: Union[float, int]) → float
```

```py
torchrec.distributed.planner.utils.gb_to_bytes(gb: float) → int
```

```py
torchrec.distributed.planner.utils.placement(compute_device: str, rank: int, local_size: int) → str
```

返回放置，格式化为字符串

```py
torchrec.distributed.planner.utils.prod(iterable: Iterable[int]) → int
```

```py
torchrec.distributed.planner.utils.reset_shard_rank(proposal: List[ShardingOption]) → None
```

```py
torchrec.distributed.planner.utils.sharder_name(t: Type[Any]) → str
```

```py
torchrec.distributed.planner.utils.storage_repr_in_gb(storage: Optional[Storage]) → str
```
