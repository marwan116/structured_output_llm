from unittest.mock import MagicMock, patch
import openai
from pydantic import Field, BaseModel
import guardrails as gd
from guardrails.validators import ValidRange


class Answer(BaseModel):
    value: int = Field(
        description="The answer to the question.",
        validators=[ValidRange(min=0, max=1, on_fail="fix")],
    )


instructions = """You are a helpful assistant."""
query = "What is 1 + 1?"
prompt = """
${query}

${gr.complete_json_suffix_v2}
"""

guard = gd.Guard.from_pydantic(
    output_class=Answer, prompt=prompt, instructions=instructions
)

magic_mock = MagicMock()
magic_mock.return_value = {
    "id": "chatcmpl-8CazZKUCp8KbiUt49J5x7eINiMlvl",
    "object": "chat.completion",
    "created": 1698012825,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "text": '{"value": "-2"}',
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 18, "completion_tokens": 12, "total_tokens": 30},
}

with patch("openai.Completion.create", magic_mock):
    raw_llm_output, validated_output = guard(
        llm_api=openai.Completion.create,
        prompt_params={"query": query},
        num_reasks=1,
        engine="text-davinci-003",
        max_tokens=1024,
        temperature=0,
    )

print(f"{raw_llm_output=}")
print(f"{validated_output=}")
