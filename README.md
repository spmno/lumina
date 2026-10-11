# lumina

**English** | [简体中文](./README.zh-CN.md)

Typed-decision System 1 engine: a Rust port of [laya](https://github.com/NandhaKishorM/laya) (Jev) built on [candle](https://github.com/huggingface/candle).

Give it a **state** (a ticket, an email, a post, a JSON document) and a set of **typed questions**, and it answers every question with a calibrated probability in a single non-autoregressive forward pass. It never generates text — no parsing errors, no hallucinations.

```text
state ──► script detection ──► router ──► checkpoint ──► forward pass ──► {answer: probability}
```

Typical uses: ticket triage and human-escalation decisions, LLM gateway routing, content moderation, input guardrails, email filtering.

## Checkpoints

Three checkpoints ship in one hub repo; only the one you ask for is downloaded.

| Checkpoint | Backbone | Params | Context | Best at |
| --- | --- | --- | --- | --- |
| `convaiinnovations/laya` (english) | ModernBERT-large | 421M | 512 | English text, guardrails, email triage |
| `convaiinnovations/laya:multilingual` | mmBERT-base | 322M | 1024 (up to 8k) | 100+ languages |
| `convaiinnovations/laya:typed-decisions` | ModernBERT-large | 421M | 1024 | four typed workflows (opt-in) |

## Requirements

- Rust 1.85+ (edition 2024)
- CPU is enough; build with `--features cuda` for GPU inference
- Network access to the Hugging Face hub on first use, or bind local directories instead (see below)

## Build

```bash
git clone <this-repo> && cd lumina
cargo build --release
# or, with CUDA:
cargo build --release --features cuda
```

Two binaries are produced: `lumina` (CLI) and `lumina-serve` (HTTP server).

## Configure checkpoint sources

lumina reads checkpoint bindings from the `LAYA_MODELS` environment variable — comma-separated `name=source` pairs, where `source` is a hub repo id or a local directory, with an optional `:subfolder`:

```bash
export LAYA_MODELS='english=convaiinnovations/laya,multilingual=convaiinnovations/laya:multilingual,typed-decisions=convaiinnovations/laya:typed-decisions'
```

A local directory works too (an HF cache snapshot or a training-run output), which is how you run fully offline:

```bash
export LAYA_MODELS="english=/path/to/snapshot,multilingual=/path/to/snapshot:multilingual"
```

With an empty or unset `LAYA_MODELS`, routing still works but prediction reports the missing source.

## CLI usage

```bash
# Routing only — decides which checkpoint would read the text; no weights loaded, milliseconds.
lumina "I was charged twice, please refund"
# Model     : english
# Reason    : English Latin text

# Predict with a built-in preset (implies --predict)
lumina "订单#30072 已经超过7天没收到货，请人工帮我核实" --preset triage
# Model     : multilingual
# Reason    : non-Latin script (han, 100% of letters); ...
# intent      : technical_help (p=0.544)
# is_urgent   : 0.044
# frustration : 1.71
# refund_requested: 0.002
# churn_risk  : 0.012

# Your own questions from a JSON file
lumina "Some request" --predict --questions questions.json

# Pin a checkpoint or language instead of auto-routing
lumina "Some request" --predict --model ml
lumina "Some request" --predict --lang de

# Batch: one state per line, states share checkpoint loads; --json emits JSONL
lumina --batch tickets.txt --predict --json
```

`questions.json` is either a bare `{qid: definition}` mapping, or an object with `questions` plus an optional `state_key` naming the field the instructions refer to:

```json
{
  "state_key": "request",
  "questions": {
    "needs_human": {
      "type": "noul",
      "instructions": "Does the customer ask for a human agent?"
    }
  }
}
```

Useful flags: `--device cpu|cuda`, `--max-len N`, `--head-max-len N`, `--json`, `--task NAME`.

## Question types

Every question is one of three types (laya's QTYPES):

| Type | Shape | Answer |
| --- | --- | --- |
| `choice` | pick one of N labels, each with a description | chosen label + probability per option |
| `noul` | yes/no question | `noul`: P(true) |
| `score` | ordered levels, low to high | score + level probabilities |

## Built-in presets

Ready-made question sets; each names the state field it reads in backticks, so put your text under that key (or override with `state_key`):

| Preset | State field | Questions |
| --- | --- | --- |
| `triage` | `message` | intent, is_urgent, frustration, refund_requested, churn_risk |
| `email` | `body` | category, is_spam, is_phishing, urgency, needs_reply |
| `guard` | `prompt` | jailbreak, prompt_injection, sensitive_data, harm_severity, topic |
| `moderation` | `post` | toxic, harassment, threat, spam, severity |
| `router` | `request` | difficulty, domain, needs_tools, is_sensitive |

## Routing

The router picks a checkpoint per request, in this order (laya's `_route`):

1. explicit `model`
2. explicit `task`
3. typed-decisions workflow match — only when `auto_task_detection` is enabled (`LAYA_AUTO_TASK=1`)
4. explicit `lang`
5. caller `lang_guess`
6. script/language detection on the state
7. default (`english`)

Detection reads up to 4000 characters of the state: non-Latin script routes to `multilingual` (the English checkpoint collapses off-English while staying confident), Latin-English routes to `english`, and undecided Latin falls through to the default.

## HTTP server

`lumina-serve` exposes the library over the Jev wire protocol — a client written against Jev's `/v1/systemone` API can point at it unchanged. Configuration is entirely via environment variables:

| Env var | Meaning | Default |
| --- | --- | --- |
| `LAYA_HOST` | bind address | `0.0.0.0` |
| `LAYA_PORT` | bind port | `8000` |
| `LAYA_DEVICE` | device for every checkpoint | (auto) |
| `LAYA_PRELOAD` | build checkpoints at startup | `1` |
| `LAYA_MODELS` | checkpoint bindings (see above) | (none) |
| `LAYA_AUTO_TASK` | auto-route to typed-decisions | `0` |
| `LAYA_MAX_LOADED` | checkpoints kept resident | `2` |
| `LAYA_API_KEY` | require `Authorization: Bearer <it>` | (none) |
| `LAYA_LOG_LEVEL` | log level | `info` |
| `LAYA_MAX_CONCURRENT` | cap on in-flight requests; excess gets 503 | `16` |
| `LAYA_MAX_TOKEN_BUDGET` | cap on per-request `max_len`/`head_max_len` | `8192` |

```bash
lumina-serve &

curl -s http://localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": {"request": "Customer asks for a refund of last month'"'"'s invoice."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle `request`?",
        "criteria": {"billing": "Payment, invoices, refunds", "technical": "Bugs and outages"}
      }
    }
  }'
```

`GET /health` is a liveness probe. Thread count is read by candle from `CANDLE_NUM_THREADS`.

## Library usage

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

    let state = json!("Customer asks for a refund of last month's invoice.");
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

For a single pinned checkpoint without routing, use `Agent::load` / `Agent::load_with` directly. See the crate docs (`cargo doc --open`) for the full API.

## Notes

- Keep `state` to the user's text only; put the questions in a questions file or request body. Passing JSON metadata as state pollutes script detection and wastes context window.
- The typed-decisions checkpoint is a specialist for its four workflows; it is never selected automatically unless you opt in.
- Deviations from laya (no hooks layer, no `predict_long`, `lang_guess` takes language codes not callables, …) are listed in the module docs.

## License

Apache-2.0, same as [laya](https://github.com/NandhaKishorM/laya) and its checkpoints.

Upstream: [laya (Jev)](https://github.com/NandhaKishorM/laya) · [convaiinnovations/laya on Hugging Face](https://huggingface.co/convaiinnovations/laya) · [candle](https://github.com/huggingface/candle)
