import json
import os
import numpy
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

QA_PROMPT = (
    "You will be given an excerpt out of a glossary. This glossary is for the "
    "rules of a board game. Answer the question given using only the "
    "information below, as if you have no prior information about the question. "
    "Give a thorough and detailed answer if you can, including all relevant "
    "information. This is only a small portion of the rules, so the answer may "
    "not appear in the excerpt I give you. If there is no mention of anything "
    "related to the question in the excerpt, respond with \"-\"."
    "\n\n"
    "GLOSSARY EXCERPT:\n"
    "---\n"
    "<<PASSAGE>>\n"
    "---"
    "\n\n"
    "QUESTION:\n"
    "<<QUERY>>"
    "\n\n"
    "ANSWER:")

SUMMARY_PROMPT = (
    "Compile the statements below into a single comprehensive answer:"
    "\n\n"
    "<<ANSWERS>>"
    "\n\n"
    "COMPILED ANSWER:")


def similarity(v1, v2):
  return numpy.dot(v1, v2)


def write_query_embedding_log(data):
  timestamp_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  with open(f'./output/{timestamp_str}_query-embedding.txt', 'w') as f:
    f.write(data)


def write_answer_log(data):
  timestamp_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  with open(f'./output/{timestamp_str}_single-answer.txt', 'w') as f:
    f.write(data)


def write_summary_log(data):
  timestamp_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  with open(f'./output/{timestamp_str}_final-summary.txt', 'w') as f:
    f.write(data)


if __name__ == '__main__':
  with open('index.json', 'r') as index:
    index_data = json.load(index)

  while True:
    query = input("Enter your question here: ")

    # Get the embedding vector for the query
    response = openai.Embedding.create(input=query,
                                       engine='text-embedding-ada-002')
    query_vector = response['data'][0]['embedding']

    query_embedding_usage = response['usage']['total_tokens']
    write_query_embedding_log(f'USAGE: {query_embedding_usage}\n\n'
                              'QUERY:\n\n'
                              f'{query}\n\n'
                              'VECTOR:\n\n'
                              f'{query_vector}')

    # `scores` will store a value for each chunk, representing
    # how similar it is to the query
    scores = list()
    for chunk in index_data:
      score = similarity(query_vector, chunk['vector'])
      scores.append({'content': chunk['content'], 'score': score})

    # Sort the scores list so the that the most similar chunks
    # are at the top
    sorted_scores = sorted(scores, key=lambda d: d['score'], reverse=True)

    # `num_chunks` is how many chunks we will take, starting with the most similar
    num_chunks = 3

    most_relevant_chunks = sorted_scores[0:num_chunks]

    answers = list()

    # Answer the question using each chunk in turn
    for chunk in most_relevant_chunks:
      prompt = QA_PROMPT.replace('<<PASSAGE>>',
                                 chunk['content']).replace('<<QUERY>>', query)

      completion_response = openai.Completion.create(engine='text-davinci-003',
                                                     prompt=prompt,
                                                     temperature=0.0,
                                                     max_tokens=300,
                                                     top_p=1.0,
                                                     frequency_penalty=0.25,
                                                     presence_penalty=0.0,
                                                     stop=['<<END>>'])
      answer_text = completion_response['choices'][0]['text'].strip()

      total_tokens = completion_response['usage']['total_tokens']
      write_answer_log(f'USAGE: {total_tokens}\n\n'
                       'PROMPT:\n\n'
                       f'{prompt}\n\n'
                       'RESPONSE:\n\n'
                       f'{answer_text}')

      answers.append(answer_text)

    # Compile all the answers together to give to gpt to summarize
    all_answers = '\n\n'.join(answers)

    final_prompt = SUMMARY_PROMPT.replace('<<ANSWERS>>', all_answers)

    summary_completion_response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=final_prompt,
        temperature=0.0,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.25,
        presence_penalty=0.0,
        stop=['<<END>>'])

    summary_text = summary_completion_response['choices'][0]['text'].strip()

    total_summary_tokens = summary_completion_response['usage']['total_tokens']

    write_summary_log(f'USAGE: {total_summary_tokens}\n\n'
                      'QUESTION:\n\n'
                      f'{query}\n\n'
                      'SUMMARY PROMPT:\n\n'
                      f'{final_prompt}\n\n'
                      'ANSWER:\n\n'
                      f'{summary_text}')

    print(f'\n{summary_text}\n\n')