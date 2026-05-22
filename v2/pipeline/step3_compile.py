"""
Step 3 — Prompt Compilation.
Compiles structured t2i + i2v prompts for every shot using the prompt compiler.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from v2.prompts.shot_compiler import compile_t2i_prompt, compile_i2v_prompt

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 3 — Prompt Compilation")
    print("=" * 60)

    dialogue_lang = getattr(state, 'dialogue_lang', 'auto')
    print(f"  Dialogue language: {dialogue_lang}")

    for shot in state.shots:
        shot.t2i_prompt = compile_t2i_prompt(shot, style=state.style)
        shot.i2v_prompt = compile_i2v_prompt(shot, style=state.style, dialogue_lang=dialogue_lang)
        print(f"  Shot {shot.shot_id}: t2i={len(shot.t2i_prompt)}c  i2v={len(shot.i2v_prompt)}c")

    return state
