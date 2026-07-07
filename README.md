# Habitat / ReplicaCAD 语言条件 RGB-D 视觉导航机器人

**网页展示链接**：https://feihaochen.github.io/embodied-nav-demo/  
**GitHub 仓库地址**：https://github.com/feihaochen/embodied-nav-demo


本项目实现了一个基于 Habitat / ReplicaCAD 的具身智能导航演示。用户可以输入中文指令，例如“请到沙发旁边”或“请到椅子旁边”，机器人 Agent 会在居家仿真场景中基于第一视角 RGB-D 图像、机器人本体状态和动作反馈，执行目标搜索、方向对齐、靠近目标和停止动作。任务完成后，Agent 会回复：“已到达 xxx 旁边，还需要什么？”

## 一、项目目标

本项目对应面试题中的导航方向，目标是构建一个可交互的具身导航 Agent：

1. 在 Habitat / ReplicaCAD 居家场景中搭建仿真环境。
2. 构建可以接受中文文字指令的具身 Agent。
3. 用户输入“请到沙发旁边”后，Agent 根据视觉观测和本体状态执行导航。
4. 任务完成后，Agent 回复“已到达 xxx 旁边，还需要什么？”
5. 构建网页进行可视化展示。
6. 提交 GitHub 仓库和网页链接。

当前版本以文字输入为主。语音输入可以作为后续扩展。

## 二、项目功能

本项目已经实现以下功能：

- Habitat / ReplicaCAD 居家仿真环境。
- 中文文字命令输入。
- 沙发和椅子两个已验证导航目标。
- 第一视角 RGB-D 视觉观测。
- 开放词汇视觉目标检测。
- 基于 Depth 的目标距离估计。
- 可解释的闭环导航状态机。
- FastAPI 本地交互网页。
- GitHub Pages 静态展示网页。
- 视频、状态日志和最终回复可视化。
- 不使用特权信息的约束说明。

## 三、最终展示目标

当前最终稳定展示目标为：

| 目标 | 中文命令 | 场景 | 随机种子 |
|---|---|---|---|
| 沙发 | 请到沙发旁边 | apt_1 | 5 |
| 椅子 | 请到椅子旁边 | apt_1 | 3 |

命令解析器和视觉检测接口已经按多目标方式设计，后续可以扩展到桌子、床等更多目标。当前最终展示页聚焦沙发和椅子，是因为这两个目标已经在当前演示场景和轨迹中完成了稳定验证；桌子和床作为后续切换场景后的扩展目标保留，不作为本次最终主展示目标。

## 四、系统结构

系统主要包含以下模块：

    用户中文命令
      -> 命令解析器
      -> 视觉目标检测器
      -> 深度安全导航 Agent
      -> Habitat 仿真后端
      -> ReplicaCAD 居家场景

核心代码文件包括：

    src/sim/habitat_backend.py
    src/agent/command_parser.py
    src/agent/owlvit_detector.py
    src/agent/depth_safe_search_agent.py
    src/utils/vis.py
    src/web/fastapi_app.py
    scripts/test_visual_nav_owlvit.py

## 五、导航状态机

导航 Agent 使用可解释的状态机：

    搜索 -> 对齐 -> 靠近 -> 停止

各状态含义如下：

- 搜索：机器人旋转或低速探索，寻找目标物体。
- 对齐：目标出现在画面中后，机器人根据检测框位置调整朝向。
- 靠近：机器人根据目标区域的 Depth 距离向目标靠近。
- 停止：机器人到达目标附近后停止，并回复用户“还需要什么？”

## 六、运行时输入约束

机器人 Agent 在运行时只使用：

- 第一视角 RGB 图像。
- 第一视角 Depth 图像。
- 机器人本体状态。
- 动作反馈和碰撞反馈。
- 用户文字命令。

机器人 Agent 在运行时不使用：

- 仿真器中的物体真值位置。
- 仿真器中的物体类别真值。
- 语义传感器。
- 最短路径跟随器。
- 上帝视角地图。
- 场景图元数据。
- 仿真器直接给出的目标坐标。

详细说明见：

    NO_PRIVILEGED_INFO.md

## 七、已验证环境

本项目在以下环境中验证通过：

    Windows 10
    RTX 2060
    WSL Ubuntu
    conda 环境：habitat-src
    Python：3.9
    Habitat-Sim：0.3.3 源码编译版本
    数据集：ReplicaCAD

在当前 Windows + WSL 环境中，所有 Habitat-Sim 脚本必须保持：

    sim_cfg.gpu_device_id = -1

不要改成：

    sim_cfg.gpu_device_id = 0

否则可能触发 EGL / CUDA 设备映射错误。

## 八、运行本地 FastAPI 交互网页

进入项目目录：

    conda activate habitat-src
    cd ~/embodied-nav-demo

启动本地网页：

    PYTHONPATH=. uvicorn src.web.fastapi_app:app --host 127.0.0.1 --port 7860

浏览器打开：

    http://127.0.0.1:7860

页面支持：

- 加载沙发缓存演示。
- 加载椅子缓存演示。
- 查看第一视角视频。
- 查看状态日志。
- 查看最终回复。
- 本地触发真实运行。

## 九、测试 GitHub Pages 静态网页

静态网页位于：

    docs/index.html

本地测试命令：

    python -m http.server 8080 --directory docs

浏览器打开：

    http://127.0.0.1:8080

该页面用于 GitHub Pages，提供稳定的网页展示链接。

## 十、运行沙发导航演示

    PYTHONPATH=. python scripts/test_visual_nav_owlvit.py \
      --scene apt_1 \
      --command "请到沙发旁边" \
      --seed 5 \
      --width 768 \
      --height 576 \
      --max-steps 140 \
      --threshold 0.25 \
      --detect-every 1 \
      --keep-last-for 0 \
      --align-threshold 0.30 \
      --stop-distance 1.60 \
      --lost-stop-distance 1.80 \
      --out-dir outputs/final_sofa_768

## 十一、运行椅子导航演示

    PYTHONPATH=. python scripts/test_visual_nav_owlvit.py \
      --scene apt_1 \
      --command "请到椅子旁边" \
      --seed 3 \
      --width 768 \
      --height 576 \
      --max-steps 180 \
      --threshold 0.25 \
      --detect-every 1 \
      --keep-last-for 0 \
      --align-threshold 0.30 \
      --stop-distance 1.20 \
      --lost-stop-distance 1.45 \
      --out-dir outputs/final_chair_768

## 十二、仓库结构

推荐仓库结构如下：

    embodied-nav-demo/
      assets/
        demo_sofa_768.mp4
        demo_sofa_768.txt
        demo_chair_768.mp4
        demo_chair_768.txt
      docs/
        index.html
        assets/
          demo_sofa_768.mp4
          demo_sofa_768.txt
          demo_chair_768.mp4
          demo_chair_768.txt
      scripts/
      src/
        agent/
        sim/
        utils/
        web/
      README.md
      NO_PRIVILEGED_INFO.md
      requirements-web.txt
      .gitignore

其中：

- assets/ 保存最终展示视频和日志。
- docs/ 用于 GitHub Pages 静态网页。
- outputs/ 保存运行过程中的中间输出，不提交到仓库。
- data/ 保存数据集，不提交到仓库。

## 十三、当前范围和后续扩展

当前版本聚焦导航任务，机器人被抽象为带 RGB-D 传感器的轮式移动平台；双臂在本导航任务中固定，不参与操作控制。

当前最终展示目标为沙发和椅子。系统接口按多目标方式设计，后续可以通过切换到包含相应物体的场景，继续扩展到床、桌子、厨房台面等目标。

后续可以继续扩展：

- 增加更多 ReplicaCAD / MP3D 场景。
- 增加床、桌子等更多目标的稳定演示。
- 增加语音输入。
- 增加更真实的轮式双臂机器人模型展示。
- 增加只用于可视化的轨迹图。
- 增加更丰富的在线交互部署方式。

## 十四、视觉检测依赖

本项目的视觉目标检测模块使用 OWL-ViT / Transformers。检测器相关依赖记录在：

    requirements_detector.txt

如果需要在已有 Habitat 环境中补装视觉检测依赖，可以执行：

    python -m pip install -r requirements_detector.txt

注意：当前项目的 Habitat-Sim 是源码编译版本，不应通过 requirements 文件重新安装 Habitat-Sim。Habitat-Sim 的环境配置请参考 README 中的已验证环境说明。
