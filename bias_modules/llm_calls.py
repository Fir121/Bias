from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

client = OpenAI(
    api_key=os.environ.get("OPENAIKEY"),
)

class ModelHandler:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        return response.choices[0].message.content
    
class Constants:
    @staticmethod
    def get_classifier_determination_system_prompt(classes_col_name, text_col_name, classes_array):
        return f"I received a dataset containing a text column named {text_col_name} and a label column named \
{classes_col_name}. The classes in the label column are {classes_array}.\
\
Output a single sentence stating the use case for what this dataset is used to predict."