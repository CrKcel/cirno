# Cirno：微型混合视觉语言模型

130M 参数的混合视觉语言模型（Gated DeltaNet + Gated Attention，3:1 交替层），含 6 层 ViT 视觉塔和 1 个 MTP 训练块。

## 文档

| 文档 | 内容 |
|---|---|
| [模型架构](doc/architecture.md) | 模型设计、参数量、显存控制手段、实测吞吐、精度与 kernel |
| [训练规划](doc/training.md) | 三阶段训练、数据准备、评测 |
| [开发进度](doc/progress.md) | 完成的部分与待办事项 |
| [AGENTS.md](AGENTS.md) | 仓库结构和安装、运行命令 |