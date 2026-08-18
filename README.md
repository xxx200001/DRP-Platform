# 病情预测平台 (Disease Risk Prediction Platform)

> 基于多维临床检验时序数据的高精度慢病风险预测平台。包含数据清洗与三态缺失处理、单位与生理极限强校验、临床衍生及纵向特征、LightGBM/Cox-PH 多时程建模、三层严格验证门禁、SHAP 归因解释、漂移监控、审计日志、模型版本/灰度/回滚管理、OCR 化验单结构化入库、时序趋势报告与随访回流。

## 📸 界面预览

| 评估主页 (包含 1/3/5 年预测、SHAP 归因与分层刻度) | 趋势追踪 (风险走势与指标近三次变化) | 移动端自适应 (支持手机相册上传与拍照) |
| :---: | :---: | :---: |
| ![评估页](docs/images/preview-assessment.png) | ![趋势页](docs/images/preview-trend.png) | ![移动端](docs/images/preview-mobile.png) |

> ⚠️ 重要说明：仓库首次启动产生的模型只使用合成纵向数据，**仅供开发联调**。生产部署必须使用真实、脱敏、获授权且带随访结局的数据重新训练，并完成时间拆分、K 折和外部独立数据验证。代码与任何离线指标都不能保证真实线上 AUC。

## 本地启动

要求 Python 3.11 或 3.12。

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# 生产必须换成不可预测的随机盐；下面仅是本地开发示例
export DRP_PII_SALT='replace-with-a-long-random-secret'
python run_app.py
```

打开 `http://127.0.0.1:8000/`；接口文档位于 `http://127.0.0.1:8000/docs`。首次启动会完成开发模型自举，通常需要数分钟，结果持久化在 `app_data/`，后续启动不会重复训练。自举使用单独的开发联调门禁，并在元数据中写入 `development_only: true`；它不能作为生产模型发布依据。

只生成开发制品而不启动服务：

```bash
python run_app.py --bootstrap
```

快速联调可减少合成患者数，但仍可能因验证集阳性样本不足而被开发门禁阻止：

```bash
python run_app.py --bootstrap --n-patients 1200
```

## 运行测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_app.py` 会执行一次小规模模型训练和完整 HTTP 链路测试，因此耗时明显长于普通单元测试。

## Docker

```bash
docker build -t drp-platform .
docker run --rm -p 8000:8000 \
  -e DRP_PII_SALT='replace-with-a-long-random-secret' \
  -v "$PWD/app_data:/opt/drp/app_data" \
  drp-platform
```

公网部署必须由网关强制 HTTPS，并限制管理接口访问；不要直接把内置开发服务器暴露到互联网。

## 目录

- `src/drp/data`：参考区间、单位校验、清洗与三态状态。
- `src/drp/features`：人口学、偏离度、比值、时序和干扰因子特征。
- `src/drp/models`：LightGBM、Cox-PH、多时程模型库和版本注册表。
- `src/drp/validation`：防泄露切分、指标、交叉验证与发布门禁。
- `src/drp/ingest`：化验文本词典和解析器。
- `src/drp/serving`：预测、SHAP、漂移、合规、审计、趋势、回流和灰度。
- `app`：FastAPI、SQLite 和零构建静态前端。
- `configs`：参考区间与干扰因子配置。

前端为零构建、零 CDN 的静态资源（`app/static`），导航按临床动线切成
患者 / 评估 / 趋势 / 管理四个目的地，窄屏自动切换为底部标签栏。
设计规则与规范符合度核对见 `合规核对与UI改版说明.md`。

## 生产上线前必做

1. 以真实脱敏随访数据重训，按时程设置空白期，显式使用 `ValidationGate()` 生产门禁，并提供外部独立验证集；禁止沿用开发自举门禁。
2. 固化并评审每个时程的验证报告、风险切点、特征清单和漂移基线。
3. 使用密钥管理系统注入 `DRP_PII_SALT`、路由盐和数据库/对象存储凭据。
4. 将 SQLite 替换为适合多实例部署的数据库；审计日志接入不可变存储和保留策略。
5. 对报告上传、OCR、授权、权限、网关、备份恢复和告警进行合规与安全验收。
6. 将平台定位为风险评估辅助工具；所有页面必须保留免责声明，不能输出确诊、治疗或开药结论。
