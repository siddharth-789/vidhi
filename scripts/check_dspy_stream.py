"""Verify DSPy streaming works end to end against the configured task model.

Run with: python scripts/check_dspy_stream.py
Requires GEMINI_API_KEY_TASK in the environment or .env file.

Findings from running this against gemini-3.1-flash-lite via dspy 3.2.1 and litellm,
recorded here so app/pipeline.py's streaming code is built on observed behavior
rather than an assumed API surface:

1. dspy.streamify plus dspy.streaming.StreamListener is the correct current API. No
   stream=True argument is needed on dspy.LM, streamify wires that up internally.
2. Incremental token chunks arrive as dspy.streaming.StreamResponse objects with a
   chunk field and an is_last_chunk flag. The final dspy.Prediction is emitted last.
3. For a short answer of a few words, Gemini's streaming response can come back as a
   single server sent event, which in one run surfaced as one StreamResponse with
   is_last_chunk=True, and in other runs surfaced as zero StreamResponse events with
   only the final Prediction. This happens with caching disabled, so it is not a
   dspy.LM cache artefact, it is Gemini or litellm choosing not to split a very short
   completion into multiple chunks.
4. For a longer answer, real incremental streaming works as expected, confirmed with
   18 separate StreamResponse chunks arriving progressively for a 300 word answer.

Conclusion: streaming is real and works for realistic length answers, which is the
production case for this project since a GST manual answer spans multiple sentences.
The zero or one chunk behaviour on trivially short answers is a Gemini API nuance, not
a bug in this code, and app/pipeline.py should not assume at least one intermediate
StreamResponse always arrives before the final Prediction.
"""

from __future__ import annotations

import asyncio
import os

import dspy
from dotenv import load_dotenv


class AnswerBriefly(dspy.Signature):
    """Answer the question in a few complete sentences."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


async def main() -> None:
    """Stream a trivial two-field DSPy signature end to end and print each chunk
    as it arrives."""

    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY_TASK"]
    model_name = os.environ.get("TASK_MODEL", "gemini-3.1-flash-lite")
    if not model_name.startswith("gemini/"):
        model_name = f"gemini/{model_name}"

    lm = dspy.LM(model_name, api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    program = dspy.Predict(AnswerBriefly)
    stream_program = dspy.streamify(
        program,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )

    print(f"Streaming from model: {model_name}")
    print("Chunks as they arrive:")
    final_prediction = None
    chunk_count = 0
    question = "Explain in a few sentences why the sky appears blue during the day."
    async for chunk in stream_program(question=question):
        if isinstance(chunk, dspy.streaming.StreamResponse):
            chunk_count += 1
            print(f"  [chunk {chunk_count}, is_last={chunk.is_last_chunk}] {chunk.chunk!r}")
        elif isinstance(chunk, dspy.Prediction):
            final_prediction = chunk

    print(f"\nTotal StreamResponse chunks received: {chunk_count}")
    print("Final prediction:", final_prediction)


if __name__ == "__main__":
    asyncio.run(main())
