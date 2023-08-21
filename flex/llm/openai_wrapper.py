import openai
import time
import json
from pydantic import BaseModel, ValidationError, ConfigDict
from typing import Type,  Optional
import pdb  # For debugging


class LLM:
    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0.5, enhanced=False):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.enhanced = enhanced
        openai.api_key = self.api_key

    def model_structure_repr(self, model: Type[BaseModel]) -> str:
        fields = model.model_fields
        field_reprs = []

        for name, field_info in fields.items():
            description = field_info.description or "No description"
            field_type = field_info.annotation

            # If it's a list type
            if getattr(field_type, '__origin__', None) == list:
                inner_type = field_type.__args__[0]

                # Check if the inner type of the list is a BaseModel
                if issubclass(inner_type, BaseModel):
                    inner_repr = self.model_structure_repr(inner_type)
                    field_reprs.append(f"{name}: [{inner_repr}]({description})")
                else:
                    field_reprs.append(f"{name}: [{inner_type.__name__}]({description})")

            # If it's a BaseModel (but not a list)
            elif issubclass(field_type, BaseModel):
                inner_repr = self.model_structure_repr(field_type)
                field_reprs.append(f"{name}: {inner_repr} ({description})")

            # For basic types (e.g. str, int, ...)
            else:
                field_reprs.append(f"{name}: {field_type.__name__} ({description})")

        return f"{{{', '.join(field_reprs)}}}"

    def is_valid_json_for_model(self, text: str, model: Type[BaseModel]) -> bool:
        """
        Check if a text is valid JSON and if it respects the provided BaseModel.
        """
        model.model_config = ConfigDict(strict=True)

        try:
            parsed_data = json.loads(text)
            model(**parsed_data)
            return True
        except (json.JSONDecodeError, ValidationError) as e:
            return False

    def generate_text(self, prompt, output_format: Optional[BaseModel] = None, n_completions=1, max_tokens=None):
        retry_delay = 0.1  # initial delay is 100 milliseconds
        valid_responses = []

        while len(valid_responses) < n_completions:
            try:
                system_message = "You are a helpful assistant."
                if output_format:
                    system_message += f" Respond in a json format that contains the following keys: {self.model_structure_repr(output_format)}"

                params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": self.temperature,
                    "n": n_completions
                }
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens

                response = openai.ChatCompletion.create(**params)
                choices = response["choices"]
                responses = [choice["message"]["content"] for choice in choices]

                if output_format:
                    valid_responses.extend([json.loads(res) for res in responses if self.is_valid_json_for_model(res, output_format)])
                else:
                    valid_responses.extend(responses)


            except openai.error.RateLimitError as err:
                print(f"Hit rate limit. Retrying in {retry_delay} seconds.")
                time.sleep(retry_delay)
                retry_delay *= 2
            except Exception as err:
                print(f"Error: {err}")
                break

        return valid_responses[:n_completions]
