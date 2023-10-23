import guardrails as gd
from pydantic import BaseModel, Field
from unittest.mock import MagicMock
import openai


class PatientInfo(BaseModel):
    gender: str = Field(description="Patient's gender")
    age: int = Field(description="Patient's age")


prompt = """
Given the following doctor's notes about a patient,
please extract a dictionary that contains the patient's information.

${doctors_notes}

${gr.complete_json_suffix_v2}
"""

magic_mock = MagicMock()
magic_mock.return_value = {
    "id": "chatcmpl-8CazZKUCp8KbiUt49J5x7eINiMlvl",
    "object": "chat.completion",
    "created": 1698012825,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "text": "{\"gender\": \"male\", \"age\": 30}",
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 18, "completion_tokens": 12, "total_tokens": 30},
}
openai.Completion.create = magic_mock
guard = gd.Guard.from_pydantic(output_class=PatientInfo, prompt=prompt)
raw_llm_output, validated_output = guard(
    openai.Completion.create,
    prompt_params={"doctors_notes": "doctors_notes"},
    engine="text-davinci-003",
    max_tokens=1024,
    temperature=0,
)
print(raw_llm_output, validated_output)
