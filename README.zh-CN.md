# lumina

[English](./README.md) | **简体中文**

Typed-decision System 1 引擎：用 [candle](https://github.com/huggingface/candle) 实现的 [laya](https://github.com/NandhaKishorM/laya)（Jev）Rust 移植版。

输入一段 **state**（工单、邮件、帖子或 JSON 文档）和一组**类型化问题**，一次非自回归前向传播返回所有答案及校准过的概率。它从不生成文本——没有解析错误，没有幻觉。

```text
state ──► 脚本检测 ──► 路由 ──► checkpoint ──► 前向传播 ──► {答案: 概率}
```

典型用途：工单分诊与转人工判定、大模型网关路由、内容审核、输入护栏、邮件过滤。

## Checkpoints

三个 checkpoint 打包在一个 hub 仓库里，用哪个才下载哪个。

| Checkpoint | 骨干网络 | 参数 | 上下文 | 定位 |
| --- | --- | --- | --- | --- |
| `convaiinnovations/laya`（english） | ModernBERT-large | 421M | 512 | 英文文本、护栏、邮件分诊 |
| `convaiinnovations/laya:multilingual` | mmBERT-base | 322M | 1024（最高 8k） | 100+ 语言 |
| `convaiinnovations/laya:typed-decisions` | ModernBERT-large | 421M | 1024 | 四个专用工作流（需显式开启） |

## 环境要求

- Rust 1.85+（edition 2024）
- CPU 即可运行；加 `--features cuda` 编译可走 GPU 推理
- 首次使用需要访问 Hugging Face hub，或改绑本地目录（见下文）完全离线运行

## 构建

```bash
git clone <this-repo> && cd lumina
cargo build --release
# 带 CUDA：
cargo build --release --features cuda
```

产出两个二进制：`lumina`（命令行）和 `lumina-serve`（HTTP 服务）。

## 配置 checkpoint 来源

lumina 从环境变量 `LAYA_MODELS` 读取 checkpoint 绑定，格式为逗号分隔的 `name=source` 对；`source` 是 hub 仓库 id 或本地目录，可带 `:subfolder` 后缀：

```bash
export LAYA_MODELS='english=convaiinnovations/laya,multilingual=convaiinnovations/laya:multilingual,typed-decisions=convaiinnovations/laya:typed-decisions'
```

`source` 也可以是本地目录（HF 缓存的 snapshot 或训练产出目录），这是完全离线运行的方式：

```bash
export LAYA_MODELS="english=/path/to/snapshot,multilingual=/path/to/snapshot:multilingual"
```

`LAYA_MODELS` 为空或未设置时，路由照常工作，预测会报告缺少 source。

## 命令行用法

```bash
# 只跑路由——决定这段文本该交给哪个 checkpoint；不加载权重，毫秒级返回。
lumina "I was charged twice, please refund"
# Model     : english
# Reason    : English Latin text

# 用内置预设预测（隐含 --predict）
lumina "订单#30072 已经超过7天没收到货，请人工帮我核实" --preset triage
# Model     : multilingual
# Reason    : non-Latin script (han, 100% of letters); ...
# intent      : technical_help (p=0.544)
# is_urgent   : 0.044
# frustration : 1.71
# refund_requested: 0.002
# churn_risk  : 0.012

# 用自己的问题文件
lumina "Some request" --predict --questions questions.json

# 固定 checkpoint 或语言，不走自动路由
lumina "Some request" --predict --model ml
lumina "Some request" --predict --lang de

# 批量：一行一个 state，共享 checkpoint 加载；--json 输出 JSONL
lumina --batch tickets.txt --predict --json
```

`questions.json` 可以是裸的 `{问题id: 定义}` 映射，也可以是带 `questions` 字段、外加可选 `state_key`（问题 instructions 引用的 state 字段名）的对象：

```json
{
  "state_key": "request",
  "questions": {
    "needs_human": {
      "type": "noul",
      "instructions": "用户是否提出需要人工客服介入的诉求？"
    }
  }
}
```

常用参数：`--device cpu|cuda`、`--max-len N`、`--head-max-len N`、`--json`、`--task NAME`。

## 问题类型

每个问题属于三种类型之一（laya 的 QTYPES）：

| 类型 | 形态 | 答案 |
| --- | --- | --- |
| `choice` | N 个标签选一，每个带描述 | 所选标签 + 各选项概率 |
| `noul` | 是非题 | `noul`：P(true) |
| `score` | 有序分级，由低到高 | 分数 + 各级概率 |

## 内置预设

开箱即用的问题集；每套预设都用反引号标出所读的 state 字段，所以请把文本放在该字段下（或用 `state_key` 覆盖）：

| 预设 | State 字段 | 问题 |
| --- | --- | --- |
| `triage` | `message` | intent, is_urgent, frustration, refund_requested, churn_risk |
| `email` | `body` | category, is_spam, is_phishing, urgency, needs_reply |
| `guard` | `prompt` | jailbreak, prompt_injection, sensitive_data, harm_severity, topic |
| `moderation` | `post` | toxic, harassment, threat, spam, severity |
| `router` | `request` | difficulty, domain, needs_tools, is_sensitive |

## 路由

路由器按以下优先级为每个请求选择 checkpoint（对应 laya 的 `_route`）：

1. 显式 `model`
2. 显式 `task`
3. typed-decisions 工作流匹配——仅当启用 `auto_task_detection`（`LAYA_AUTO_TASK=1`）时
4. 显式 `lang`
5. 调用方 `lang_guess`
6. state 的脚本/语言检测
7. 默认（`english`）

检测最多读取 state 的 4000 个字符：非拉丁文路由到 `multilingual`（英文 checkpoint 在非英文上不是平滑退化而是直接崩掉，置信度还很高）；拉丁文且是英文路由到 `english`；拉丁文但语言未定则落到默认值。

## HTTP 服务

`lumina-serve` 通过 Jev 线协议暴露整个库——按 Jev 的 `/v1/systemone` API 写的客户端可以直接把 baseUrl 指过来，无需改动。配置全部走环境变量：

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `LAYA_HOST` | 绑定地址 | `0.0.0.0` |
| `LAYA_PORT` | 绑定端口 | `8000` |
| `LAYA_DEVICE` | 所有 checkpoint 的设备 | （自动） |
| `LAYA_PRELOAD` | 启动时预加载 checkpoint | `1` |
| `LAYA_MODELS` | checkpoint 绑定（见上文） | （无） |
| `LAYA_AUTO_TASK` | 自动路由到 typed-decisions | `0` |
| `LAYA_MAX_LOADED` | 常驻内存的 checkpoint 数 | `2` |
| `LAYA_API_KEY` | 要求 `Authorization: Bearer <it>` | （无） |
| `LAYA_LOG_LEVEL` | 日志级别 | `info` |
| `LAYA_MAX_CONCURRENT` | 并发请求上限，超出返回 503 | `16` |
| `LAYA_MAX_TOKEN_BUDGET` | 单请求 `max_len`/`head_max_len` 上限 | `8192` |

```bash
lumina-serve &

curl -s http://localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": {"request": "客户要求退还上个月发票的款项。"},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle `request`?",
        "criteria": {"billing": "Payment, invoices, refunds", "technical": "Bugs and outages"}
      }
    }
  }'
```

`GET /health` 是存活探针。线程数由 candle 直接读 `CANDLE_NUM_THREADS`。

## 库用法

```rust
use lumina::{PredictOptions, Router, RouterOptions, parse_questions};
use serde_json::json;

fn main() -> lumina::Result<()> {
    let router = Router::new(&RouterOptions {
        models: &[
            ("english", "convaiinnovations/laya", None),
            ("multilingual", "convaiinnovations/laya", Some("multilingual")),
            ("typed-decisions", "convaiinnovations/laya", Some("typed-decisions")),
        ],
        ..Default::default()
    })?;

    let state = json!("客户要求退还上个月发票的款项。");
    let questions = parse_questions(&json!({
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "billing": "Payment, invoices, refunds",
                "technical": "Bugs and outages"
            }
        }
    }))?;

    let result = router.predict(&state, &questions, &PredictOptions::default())?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
```

不需要路由、固定用单个 checkpoint 时，直接用 `Agent::load` / `Agent::load_with`。完整 API 见 crate 文档（`cargo doc --open`）。

## 注意事项

- **state 只放用户文本**，问题放 questions 文件或请求体。把 JSON 元数据当 state 送进去会污染脚本检测，也浪费上下文窗口。
- typed-decisions 是它四个工作流的专科模型，不显式开启永远不会被自动选中。
- 与 laya 的差异（无 hooks 层、无 `predict_long`、`lang_guess` 收语言代码而非 callable 等）列在各模块文档里。

## 许可证

Apache-2.0，与 [laya](https://github.com/NandhaKishorM/laya) 及其 checkpoints 相同。

上游：[laya (Jev)](https://github.com/NandhaKishorM/laya) · [Hugging Face 上的 convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) · [candle](https://github.com/huggingface/candle)
