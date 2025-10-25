from modeling import MicroLLM, MicroLLMConfig

model = MicroLLM(MicroLLMConfig())

print(model)

params = sum([p.numel() for p in model.parameters()]) / 1_000_000

print(f"{params=}")