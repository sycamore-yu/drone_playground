# P2 独立 CPU 评测

日期：2026-09-25。当前汇总包含 4 个正式运行，均已完成开发集选模后的独立 CPU 评测。第四条 `p2-random-apg-seed0-v1` 的训练已在 655,360 步完整返回，随后仅在可选实时发布阶段因共享查看器切换全局模型发生后处理失败；恢复过程没有重新训练。

所有独立评测都由开发集 `best.json` 已固定的检查点启动新 CPU 进程完成。留出集从未参与检查点选择。评测前后检查点文件 SHA-256 一致，报告均记录 `parameters_frozen=true`，并使用安装的原版 rscope 完成轨迹和模型读回。

| 正式运行 | 方法 / 任务 | 训练步数 | 开发集 | 开发 RMSE | 留出集 | 留出 RMSE |
|---|---|---:|---:|---:|---:|---:|
| `p2-figure8-ppo-seed0-v2` | PPO / FigureEight | 2,097,152 | 32/32 | 0.0275328675 m | 128/128 | 0.0268792909 m |
| `p2-figure8-apg-seed0-v1` | APG / FigureEight | 655,360 | 32/32 | 0.0263830072 m | 128/128 | 0.0257421473 m |
| `p2-random-ppo-seed0-v1` | PPO / Random spline | 4,194,304 | 32/32 | 0.0101957350 m | 128/128 | 0.0099163009 m |
| `p2-random-apg-seed0-v1` | APG / Random spline | 655,360 | 32/32 | 0.0079232355 m | 128/128 | 0.0076164509 m |

## FigureEight PPO：`p2-figure8-ppo-seed0-v2`

已有证据保持不变。开发集选择 `step-0001310720.pkl`；训练阶段 `best.json` 为 32/32、RMSE `0.027532807865246953 m`。检查点文件 SHA-256 为 `1a58835b49c0242ef94e4cebd283b771a09380a6901a1a771a15d6d7fa9665e7`，参数摘要为 `2eadc9c5e3ddc4d066833a685eb382dc8f2ecd1859d84916dffa8894a884ac56`，规范化状态摘要为 `c01d1f60937d7505cd023faefdbbe39faa4e0b786c88e72a298636776153e960`。

开发集命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-figure8-ppo-seed0-v2/checkpoints/step-0001310720.pkl \
  --split dev --episodes 32 --device cpu \
  --output experiments/p2-figure8-ppo-seed0-v2/independent-dev
```

退出码 `0`；32/32 完成，RMSE `0.027532867460819364 m`。留出集使用同一检查点、`--split heldout --episodes 128`，退出码 `0`；128/128 完成，RMSE `0.02687929089618562 m`。两个输出的原版 rscope 读回均通过，500 帧、5 个代表 case，模型 `nq=1`、`nmocap=1`、12 个资源。

## FigureEight APG：`p2-figure8-apg-seed0-v1`

`checkpoints/best.json` 选择最终 `step-0000655360.pkl`，训练阶段开发集结果为 32/32、RMSE `0.026382944181246003 m`。检查点文件评测前后 SHA-256 均为 `a4f3382c65a629c01fc788e3c4d8aa7c2d9d7c0f774d9d7a173a754ed933d6db`；sidecar、开发集报告和留出集报告的参数摘要均为 `93824a48885feb436b66c738ba7db537a418555362cb2c0ea1ab2d033d7a3fe4`。两个 fresh CPU load 的规范化状态摘要均为 `1d4ee72bb42a12179a0bc7c7b2ba7ac26b47fea5a5aa47632932c64a4060ab2d`。

开发集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-figure8-apg-seed0-v1/checkpoints/step-0000655360.pkl \
  --split dev --episodes 32 --device cpu \
  --output experiments/p2-figure8-apg-seed0-v1/independent-dev
```

退出码 `0`，PID `2413530`；32/32 完成、0 失败，平均 RMSE `0.026383007203766295 m`，平均 return `494.22784423828125`。完整报告 SHA-256 为 `fba14f006cfc32f115b64cf98d4522a5b6d1f8e5807dc2b3186a2c60f7fec850`。与训练期 best RMSE 的差约 `6.30e-8 m`。

留出集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-figure8-apg-seed0-v1/checkpoints/step-0000655360.pkl \
  --split heldout --episodes 128 --device cpu \
  --output experiments/p2-figure8-apg-seed0-v1/independent-heldout
```

退出码 `0`，PID `2414633`；128/128 完成、0 失败，平均 RMSE `0.025742147275704633 m`，平均 return `494.1583557128906`。完整报告 SHA-256 为 `54088356d145de12de9ceb78260cc2b74f963b041182e05d16ec54e11bf92e70`。

开发集和留出集的 `.mj_unroll` 均经原版 `rollout.append_unroll` 与 `model_loader.load_model_and_data(False)` 实际读回；两者均为 500 帧、5 个代表 case，重建模型 `nq=1`、`nmocap=1`，metadata 含 12 个模型资源。

## Random PPO：`p2-random-ppo-seed0-v1`

`checkpoints/best.json` 从开发集选择 `step-0003670016.pkl`，训练阶段开发集结果为 32/32、RMSE `0.01019574133424919 m`。检查点文件评测前后 SHA-256 均为 `4fbdd779c089dfae090018c6fa6b42c089bbe7acd5840dd1fe66b610d29d61af`；sidecar、开发集报告和留出集报告的参数摘要均为 `75fdea9fb9ba48a11b68fc149f5b089d57645e2a4cdeb6911f3f609110b6f07e`。两个 fresh CPU load 的规范化状态摘要均为 `71e054e09e80f58bc019f96a3bb817000cbd70c2e8b50b50b406c03bc437654a`。

开发集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-random-ppo-seed0-v1/checkpoints/step-0003670016.pkl \
  --split dev --episodes 32 --device cpu \
  --output experiments/p2-random-ppo-seed0-v1/independent-dev
```

退出码 `0`，PID `2415936`；32/32 完成、0 失败，平均 RMSE `0.010195735010533299 m`，平均 return `744.9356689453125`。完整报告 SHA-256 为 `5f89748530f9e1a450908df19ba00335cec1fdba91d68aa020a45bef7f4ebfca`。

留出集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-random-ppo-seed0-v1/checkpoints/step-0003670016.pkl \
  --split heldout --episodes 128 --device cpu \
  --output experiments/p2-random-ppo-seed0-v1/independent-heldout
```

退出码 `0`，PID `2416728`；128/128 完成、0 失败，平均 RMSE `0.009916300922920975 m`，平均 return `745.0072021484375`。完整报告 SHA-256 为 `aaa45b453cd9ce422c60d45df9d81e394e4a4b02be2b5072f5707dcc02f1ed42`。

随机样条的三套 reference bank 已独立重建并校验：train 使用 `reference_seed=10000`、256 条、摘要 `aeb068011737c86bdb56d690781b8d4cf78a92b7cb8fd875282a9382969899b1`；dev 使用 `20000`、32 条、摘要 `fd770a32b4868d1b528d00511a44bb4ffb02ca6701804e8f7404b09a5a1fd95b`；heldout 使用 `30000`、128 条、摘要 `513ccae6f143b6b6a49a05fe3b2e03fc4bd61c0fac223226e375bf12d22c2d52`。三者摘要互异，heldout 报告中 128 个 `reference_seed` 精确覆盖 30000–30127，因此留出集没有复用 train 或 dev 轨迹库。

Random PPO 的开发集和留出集 `.mj_unroll` 均经原版 rscope 实际读回；两者均为 750 帧、5 个代表 case，重建模型 `nq=1`、`nmocap=1`，metadata 含 12 个模型资源。

## Random APG：`p2-random-apg-seed0-v1`

`checkpoints/best.json` 明确由开发集选择最终 `step-0000655360.pkl`，训练阶段开发集结果为 32/32、RMSE `0.007923819882126884 m`。训练本体已经完成 655,360 steps，最终策略相对初始策略的参数 L2 变化为 `7.35811709750219`。原运行随后在最终实时发布阶段记录 `ValueError('Live snapshots must use the same model; select a new run explicitly')`：当时另一个查看器已经切换共享活动目录中的全局模型。`postprocessing-recovery.json` 将其明确归类为 `postprocessing-only`，记录 `training_rerun=false`、`native_training_returned=true`、`checkpoint_updates=none`。

恢复脚本在写收尾状态前验证了最终 step、32 回合开发集、750×5 原生轨迹和模型读回，并保存了原失败结果、失败状态和恢复前 `best.json`。恢复记录中的五个训练产物 SHA 在本次独立评测之后再次逐项核对，仍分别为：final checkpoint `35c048a62ff6d1298b6ccc0dcbc0e63848f7215ef5417b21e23ac7ab80db6ecd`、checkpoint sidecar `a34c5fa81d8c8afdee1588f4a269312e2bca5ba9fabf8e80323e9f8f65bdfdef`、final dev report `724b7799b33740180c9421eb893d495586d2b87c9a751d389792817b63d83eb6`、训练期 final rollout `0a5edffa711e79e77e848a3d210e8dd6d1e6d15c48b38aaa4896860ea115aca2`、resolved config `7080852740bcf5e9319f5abe919b77cb05ca54f95b027f2766e16b30a395c820`。因此恢复和本次独立评测均未改写训练权重、配置或原始评测证据。

final checkpoint 的参数摘要为 `adbb76d1bedbce61e4859ad9bc19b4e3d8d3eb04167a582007ea8580310e6923`；两个 fresh CPU load 的规范化状态摘要均为 `2b51eb952717b03b37f94f16329df7eda03d44b00b20999618daaf6f01feb655`。

开发集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-random-apg-seed0-v1/checkpoints/step-0000655360.pkl \
  --split dev --episodes 32 --device cpu \
  --output experiments/p2-random-apg-seed0-v1/independent-dev
```

退出码 `0`，PID `2433402`；32/32 完成、0 失败，平均 RMSE `0.00792323546847087 m`，平均 return `747.0143432617188`。完整报告 SHA-256 为 `a3d94550a6ba941dcfdf5adb373f88c8407e83146cba95b622266c0af598be4b`。

留出集实际命令：

```bash
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run evaluate \
  --checkpoint experiments/p2-random-apg-seed0-v1/checkpoints/step-0000655360.pkl \
  --split heldout --episodes 128 --device cpu \
  --output experiments/p2-random-apg-seed0-v1/independent-heldout
```

退出码 `0`，PID `2434685`；128/128 完成、0 失败，平均 RMSE `0.007616450902146312 m`，平均 return `747.1568603515625`。完整报告 SHA-256 为 `aa1ba71d0a54ff4e8d1fcd83d93b97a2292440f90f30ff39d15c52b2beb7e36f`。

随机样条 reference bank 与 Random PPO 使用同一冻结任务协议：train 为 `10000/256`、dev 为 `20000/32`、heldout 为 `30000/128`，三套 SHA 分别为 `aeb068011737c86bdb56d690781b8d4cf78a92b7cb8fd875282a9382969899b1`、`fd770a32b4868d1b528d00511a44bb4ffb02ca6701804e8f7404b09a5a1fd95b`、`513ccae6f143b6b6a49a05fe3b2e03fc4bd61c0fac223226e375bf12d22c2d52`。heldout 报告中的 128 个 `reference_seed` 精确覆盖 30000–30127。

独立 dev/heldout 轨迹均由原版 rscope 成功读回，都是 750 帧、5 个代表 case，重建模型 `nq=1`、`nmocap=1`，metadata 含 12 个资源。

## 当前结论

四个正式 P2 运行都完成了开发集选出的 best checkpoint 的新 CPU 进程独立重载。8 次独立评测全部退出码 `0`：每个运行均包含开发集 32 回合和留出集 128 回合，总独立试次分母为 `4 × (32 + 128) = 640`，其中 640/640 完成、0 失败。所有检查点文件、参数摘要与规范化状态在评测前后保持冻结，8 个独立轨迹包均可由原版 rscope 读取。两条 Random 任务的 heldout 都明确使用 `reference_seed=30000` 起的独立 128 条随机样条轨迹。Random APG 的训练后发布故障已经作为后处理恢复单独保留，恢复过程 `training_rerun=false`，且训练期 final checkpoint、sidecar、开发集评测、轨迹与配置原始字节均经 SHA 再验证保持不变。
