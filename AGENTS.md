# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

## Gỡ DVC — 26/11/2026

- Repo không còn dùng DVC. Đã gỡ `.dvc/`, `.dvcignore` và 56 pointer `*.dvc` khỏi nhánh `master`. Pointer cũ còn trong lịch sử Git, xem commit `f9014e1`.
- Đã xóa `NCKH/veremi_build/fl_dagsnet_veremi/dvc-storage` trên OneDrive (5.412 object, 73,4 GB) vào thùng rác OneDrive. Remote cũ trên Google Drive `/mnt/g/My Drive/NCKH/veremi_build/fl_dagsnet_veremi` không còn.
- `dvc pull` không còn khôi phục được dữ liệu. Code và notebook vẫn trong Git. Git bundle `fl_dagsnet_veremi.bundle` và `secrets/fl_dagsnet_veremi` vẫn giữ ở `NCKH/veremi_build/` trên OneDrive. Phần DVC trong `backup.sh` không còn dùng được.
- Lý do: owner chỉ giữ DVC cho các dự án đang làm (`odixe06/experiment`, `experiment_10k_clients`). Các hướng dẫn DVC bên dưới chỉ còn giá trị lịch sử.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Rules: 

**If anything needs to be done manually, please let me know so I can do it.**

**During the process, if you are unsure about anything, ask me questions to clarify and confirm the requirements before proceeding**

