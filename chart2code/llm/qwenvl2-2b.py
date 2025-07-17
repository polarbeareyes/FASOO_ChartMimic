from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@registry.register_llm("qwenvl2b")
class QwenVL2B:
    def __init__(self, engine="Qwen/Qwen2-VL-2B-Instruct", max_tokens=512):
        self.engine = engine
        self.max_tokens = max_tokens

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            engine,
            device_map="auto",
            torch_dtype="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(engine)

    def generate(self, messages):
        # Format conversation into chat prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process visual info (for images)
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize full input
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        # Trim input IDs to get only new tokens
        trimmed = [
            output[len(inputs.input_ids[i]):]
            for i, output in enumerate(generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    @classmethod
    def from_config(cls, config):
        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        system_message = config.get("system_message", "")
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        retry_delays = config.get("retry_delays", 10)
        context_length = config.get("context_length", 4096)
        return cls(
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            retry_delays=retry_delays,
            system_message=system_message,
            context_length=context_length,
            stop=stop,
        )
