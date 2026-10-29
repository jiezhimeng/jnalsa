# 江南百景图助手

个人自用，纯ai代码，仅适配特定游戏进度，使用大概需自行更改以适配，仅推荐大佬部署

基于 [AzurLaneAutoScript](https://github.com/LmeSzinc/AzurLaneAutoScript) 的江南百景图专用衍生项目，保留 GPLv3 许可证和原作者归属。2026-10-29 整理为独立源码发布目录。

支持签到、每日奖励、仓库盘点、物资调配、码头、税课司、朝暮鼓、逐城生产、徽商、宁波商船、大运河、客栈、家具铺及驿站。当前识别模板按 Windows、江南 TapTap 包 `com.cis.jiangnan.taptap` 和 1280×720 画面准备；其他环境需另行验证。

## 下载完整源码和内置地图

大运河原图完整保存在 `assets/jiangnan/grand_canal/map/overview_stitched.png`，约 121 MiB，使用 **Git LFS** 管理。地图原图、网页切片、客栈全景和定位数据均是程序内置资源。

安装 Git 与 Git LFS 后克隆本仓库，并在仓库目录执行：

```powershell
git lfs install --local
git lfs pull
python tools/validate_release.py
```

`validate_release.py` 会检查原图的实际文件大小及 SHA-256，避免把 LFS 指针误当成图片。网页下载 ZIP 是否包含 LFS 实体取决于 GitHub 仓库归档设置；推荐使用 Git 克隆并执行 `git lfs pull`。

## 准备两个 Python 环境

此目录是源码发布版，不含 Python 安装包、已安装虚拟环境或 OCR 模型缓存。主程序继续使用原有 Python 3.7 依赖；OCR 在单独的现代 Python 环境中运行。Python 3.7 已停止维护，目前并未完成整个主程序向新版本迁移。

在仓库根目录使用自己的 Python 3.7 64 位解释器创建主环境：

```powershell
py -3.7 -m venv .runtime
.\.runtime\Scripts\python.exe -m pip install -r requirements.txt
```

OCR 工作进程沿用已开发环境的 Python 3.13；创建名为 `.venv` 的独立环境：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ppocr.txt
```

PP-OCRv6 medium 模型在缺失时由 PaddleX 根据模型名获取，也可以放在 `bin/ppocr_models/PP-OCRv6_medium_rec_infer`。首次获取需要网络，模型下载及全新环境依赖安装未在本次整理中验证。

## 启动与配置

运行 `start.bat`，默认只监听本机 `127.0.0.1`。它会优先使用 `toolkit/python.exe`，其次 `.runtime/Scripts/python.exe`。也可设置环境变量 `JIANGNAN_PYTHON` 指向已有主环境解释器；该环境仍需具备 `requirements.txt` 中的依赖。

启动后在界面选择模拟器和设备序列号，检查截图与控制方式，再启用需要的江南任务。只提供干净模板；个人配置、账号截图、运行状态和日志不随源码发布。运行时会在 `config/` 和 `log/` 下生成本地文件。

默认界面为简体中文，部署模板统一使用 `main` 分支和本机地址。默认关闭自动更新，更新仓库地址留空。请为自己的江南仓库设置更新地址；原版 ALAS 更新来源会被拒绝，避免覆盖衍生版。

## 内置素材和开发工具

`assets/jiangnan/` 中的整张截图被代码用作识别模板，不能仅因看起来像普通截图而删除。发布目录已排除根目录调试截图、测试样本集、临时实验结果、旧压缩包及无调用的参考图。

- `tools/validate_release.py`：检查文件完整性、图片引用、地图原图、全部 152 张地图切片、配置和 Python 语法，不操作模拟器。
- `tools/regenerate_config.py`：在主 Python 环境中重新生成配置。
- `tools/build_canal_tiles.py`：从内置原图重建网页地图切片，需要 Pillow 和足够内存。
- `RELEASE_MANIFEST.json`：发布文件 SHA-256 清单与筛选规则。

驿站与奖励说明见 [驿站](doc/jiangnan_post_station.md) 和 [任务奖励](doc/jiangnan_task_reward.md)。文档中提及的历史离线测试和 `dev_tools` 开发样本不随此发布目录提供。

本版本的验证以文件检查、导入及离线回归为主；不等同于实机运行验证，也不保证在不同模拟器和游戏版本下直接可用。

## 上传 GitHub

此目录使用全新的 Git 仓库，没有继承原工作目录中缺失对象的 Git 历史。先运行完整性检查，再提交并推送到你自己的 GitHub 仓库。大地图由 Git LFS 上传，工作目录中保留完整 PNG，不需要删除或另行手动放置。

发布代码遵循 GPLv3；上游来源及许可证保留在此。游戏截图和素材、第三方 Android 工具、OCR 模型分别遵循其权利人的条款；本次整理不构成对这些资产再分发权限的确认。
