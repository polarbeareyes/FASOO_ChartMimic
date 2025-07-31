from agents.base_agent import BaseAgent
from common.registry import registry
import json
import base64
import PyPDF2
import os
import pandas as pd
# from prompt_template import prompt_templates

from dotenv import load_dotenv

load_dotenv()


@registry.register_agent("DirectAgent")
class DirectAgent(BaseAgent):
    def __init__(
        self,
        llm_model,
        prompt_path=None,
    ):
        super().__init__()
        self.llm_model = llm_model
        self.dimensions_info = pd.read_json(
            f"{os.environ['PROJECT_PATH']}/dimentions_info.jsonl",
            lines=True,
        )
        if prompt_path is not None:  # load from file
            self.init_prompt_dict = json.load(open(prompt_path, "r"))
            self.instruction = self.init_prompt_dict["instruction"]
            # self.system_msg = self.init_prompt_dict.get("system_msg", "")
        else:
            raise Exception("init_prompt_path is None")

    def run(self, file):
        width, height = self._get_pdf_dimensions(file)
        conversation = self._constract_conversation(file, width, height)
        response = self.llm_model.generate(conversation)
        return response

    def _get_pdf_dimensions(self, pdf_path):
        file_idx = "{}_{}".format(
            pdf_path.split("/")[-1].split("_")[0],
            pdf_path.split("/")[-1].split("_")[1],
        ).replace(".pdf", "")
        width = self.dimensions_info[self.dimensions_info["idx"] == file_idx][
            "width"
        ].values[0]
        height = self.dimensions_info[self.dimensions_info["idx"] == file_idx][
            "height"
        ].values[0]
        return width, height
    
    def _constract_conversation(self, file, width, height):
        # user_prompt = self.instruction.format(height=height, width=width)

        # if self.prompt_format not in prompt_templates:
        #     raise ValueError(f"Unsupported prompt format: {self.prompt_format}")
    
        # template = prompt_templates[self.prompt_format]
        # formatted_prompt = template.format(
        #     system_prompt=self.system_msg,
        #     prompt=user_prompt
        # )

        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": formatted_prompt},
        #             {"type": "image", "image_url": file.replace(".pdf", ".png")},
        #         ],
        #     }
        # ]
        # return conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.instruction.format(height=height, width=width),
                    },
                    {"type": "image", "image_url": file.replace(".pdf", ".png")},
                ],
            }
        ]
        return conversation

    @classmethod
    def from_config(cls, llm_model, config):
        init_prompt_path = config.get("prompt_path", None)
        # prompt_format = config.get("prompt_format", "gemma")
        return cls(llm_model, init_prompt_path)
