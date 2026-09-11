# 实验谱系 v3

从实际run的config.resolved.json与最新checkpoint元数据重建，每个run独立一行。未从旧CSV截断/猜测错位字段。

共65个有配置记录的run；旧版/v2保留历史，不用于机器分析。

scratch：未指定外部init/resume；continuation：明确外部init checkpoint；resume：沿用指定断点，但若初始来源未保存则标记未知。

GPU实际数量没有持久化时明确标记未知，require_single_gpu仅为配置证据。噪声层以resolved字符串记录，checkpoint里的实际模块属性已在主结果检查中核对。

[逐run CSV](full_experiment_lineage_v3.csv)含完整resolved配置、全部关键训练字段及元数据不一致记录。没有重新训练或运行历史seed。
