import json
import os
import textwrap
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


def log_chunks(chunks):
  # Create folder structure based on timestamp
  timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  output_folder = f"output/chunks/{timestamp}"
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Loop over chunks and write each one to its own file in the output folder
  for i, chunk in enumerate(chunks):
    filename = f"chunk_{i}.txt"
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w') as file:
      file.write(chunk)


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

  max_chunk_size = 16000

  chunks = []

  current_chunk = ""
  for line in allText.splitlines():
    if len(current_chunk + line) > max_chunk_size:
      chunks.append(current_chunk.strip())
      current_chunk = ""
    current_chunk += line + "\n"

  if current_chunk:
    chunks.append(current_chunk.strip())

  log_chunks(chunks)

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
