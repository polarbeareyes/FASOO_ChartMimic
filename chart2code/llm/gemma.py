from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from common.registry import registry

@registry.register_llm("gemma")
class GEMMA:
    def __init__(
        self,
        engine="google/gemma-3-4b-it",
        temperature=0.1,
        max_tokens=4096,
        top_p=0.95,
        stop=[],
        retry_delays=1,
        max_retry_iters=3,
        context_length=4096,
        **kwargs
    ):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length

        self.processor = AutoProcessor.from_pretrained(engine)
        self.model = AutoModelForCausalLM.from_pretrained(
            engine,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

    def generate(self, conversation):
        # #디버깅
        # print("[Gemma] Received conversation:", conversation)

        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        image = None
        for content in conversation[0]["content"]:
            if content["type"] == "image":
                image = Image.open(content["image_url"]).convert("RGB")
                break
        if image is None:
            raise ValueError("No image found in conversation input.")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # # 디버깅
        # print("[Gemma] Processor inputs prepared.")


        for _ in range(self.max_retry_iters):
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p
                    )
                decoded = self.processor.decode(output[0], skip_special_tokens=True).strip()
                # #디버깅
                # print("[Gemma] Generation output:", decoded)
                return decoded
            except Exception as e:
                print(f"[Gemma Retry] {str(e)}")
                time.sleep(self.retry_delays)

        return "[Error] Model generation failed."

    @classmethod
    def from_config(cls, config):
        return cls(
            engine=config.get("engine", "google/gemma-3-4b-it"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
            top_p=config.get("top_p", 0.95),
            stop=config.get("stop", []),
            retry_delays=config.get("retry_delays", 1),
            max_retry_iters=config.get("max_retry_iters", 3),
            context_length=config.get("context_length", 4096),
        )
