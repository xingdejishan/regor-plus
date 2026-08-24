# OrbitRegor

严格 pairwise、PLY-only、无新增训练的轨道商空间再生成实现。

在线输入仅包含当前 source/target XYZ 和冻结官方 REGOR 特征。核心链路为：patch atlas、云内近似自同构 orbit cover、orbit-valued lift atoms、交换关系收缩、实例物化、原生 REGOR 点级 densification 和唯一位姿输出。

`run_orbit_regor_online.py` 通过无身份 framed stream 运行 baseline 和完整消融；`evaluate_orbit_regor.py` 只能在在线目录完成密封后由独立进程读取 GT。
