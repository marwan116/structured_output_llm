# early stopping
# no option to pass temperature or top_p/top_k when generating the logits
# caching a random number generator - means same output for same input
# import outlines.text.generate as generate
# import outlines.models as models

# model = models.transformers("gpt2")
# answer = generate.continuation(model, stop=["."])("Tell me a one-sentence joke.")
# print(f"{answer=}")

# multiple choices
import outlines.text.generate as generate
import outlines.models as models

model = models.transformers("gpt2", device="mps")

prompt = """sqrt(2)="""
# answer = generate.continuation(model, stop=[], max_tokens=20)(prompt)
# answer = generate.choice(model, ["Positive", "Negative"])(prompt)
answer = generate.float(model, max_tokens=20)(prompt)
print(answer)
