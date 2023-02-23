import json
import os
import textwrap
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


def write_log(index, tokens_used):
  output = {'tokens_used': tokens_used, 'index': index}
  json_string = json.dumps(output)
  timestamp_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  with open(f'./output/{timestamp_str}_build-index.json', 'w') as f:
    f.write(json_string)


def write_index(index):
  json_string = json.dumps(index)
  with open(f'index.json', 'w') as f:
    f.write(json_string)


if __name__ == '__main__':

  # Split the input file into chunks
  with open('input.txt', 'r') as file:
    allText = file.read()
  chunks = textwrap.wrap(allText, 4000)

  index = list()
  total_tokens = 0

  # Send each of the chunks to get a matching vector
  for chunk in chunks:
    # response gives a big nested object and all we need off it is the embedding (vector)
    response = openai.Embedding.create(input=chunk,
                                       engine='text-embedding-ada-002')
    vector = response['data'][0]['embedding']

    # Usage just for informational purposes
    total_tokens += response['usage']['total_tokens']

    index.append({'content': chunk, 'vector': vector})

  write_log(index, tokens_used=total_tokens)
  write_index(index)
