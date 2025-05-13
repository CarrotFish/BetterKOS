# BetterKOS 开发框架

## 简介

BetterKOS开发框架是一个基于K-Scale机器人项目，对机器人编程进行优化和扩展的开发框架。

## 安装

克隆本仓库到本地，环境配置参考[KOS_Deploy_Guidance](https://github.com/CarrotFish/KOS_Deploy_Guidance)

## 主程序开发

***注意将所有的自己编写的代码放在`BetterApp`包下，以避免同步***

在根目录下新建文件夹 `BetterApp` ，新建文件 `main.py` 参照examples/test.py编写自己的程序。

主程序逻辑如下：
- 导入需要的库
- 建立自定义App类，继承BetterKOS类
- 在异步函数 `update` 和 `update_after` 中编写每帧需要的逻辑处理
- 调用BetterKOS.app.run函数，传入App类（不用实例化）运行程序

## 配置config

将model_100.onnx放入根目录

在项目根目录新建 `config.json` 文件，并填写以下内容：

```json
{
    "robot_ip": "192.168.42.1",
    "robot_port": 50051,
    "model_file": "model_100.onnx"
}
```
其中 `robot_ip` 为机器人IP地址，`robot_port` 为机器人端口号。

然后在根目录运行 `python BetterApp/main.py` 即可启动程序。

## 扩展组件开发

为了将拓展组件统一化开发，我们准备将所有的拓展程序放在 `BetterMods.<开发者>` 包下，由开发者自行维护。

对所有的拓展组件做如下规定：
- 如果需要调用到机器人本体，需要将 *BetterKOS.app.BetterKOS* 实例作为第一个参数传入。
- 所有的需要被显式调用的拓展都需要通过长字符串编写参数说明以及功能介绍。
- 如果是未完成的组件，请在说明长字符串之前标注 `[测试中]` 标签
