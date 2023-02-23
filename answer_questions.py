import json
import os
import numpy
from datetime import datetime
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

qa_prompt = (
    "Use the following passage to give a concise answer to the following question, disregarding any irrelevant information. If there is no relevant information in the passage, say nothing.\n\n"
    "QUESTION: <<QUERY>>\n\n"
    "PASSAGE: <<PASSAGE>>\n\n"
    "ANSWER: ")

summary_prompt = ("Write a detailed summary of the following:\n\n"
                  "<<SUMMARY>>\n\n"
                  "DETAILED SUMMARY: ")


def similarity(v1, v2):  # return dot product of two vectors
  return numpy.dot(v1, v2)


# TODO - create separate logging methods for the initial methods and the summary,
# and add the type to the filename. Also log usage


def write_log(data):
  timestamp_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  with open(f'./output/output_{timestamp_str}.txt', 'w') as f:
    f.write(data)


if __name__ == '__main__':
  with open('index.json', 'r') as index:
    index_data = json.load(index)

  while True:
    query = input("Enter your question here: ")

    # Get the embedding vector for the query
    response = openai.Embedding.create(input=query,
                                       engine='text-similarity-ada-001')
    query_vector = response['data'][0]['embedding']

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
      prompt = qa_prompt.replace('<<PASSAGE>>',
                                 chunk['content']).replace('<<QUERY>>', query)

      completion_response = openai.Completion.create(engine='text-davinci-003',
                                                     prompt=prompt,
                                                     temperature=0.6,
                                                     max_tokens=2000,
                                                     top_p=1.0,
                                                     frequency_penalty=0.25,
                                                     presence_penalty=0.0,
                                                     stop=['<<END>>'])
      answer_text = completion_response['choices'][0]['text'].strip()

      write_log('PROMPT:\n\n' + prompt + '\n\nRESPONSE:\n\n' + answer_text)

      answers.append(answer_text)

    # Compile all the answers together to give to gpt to summarize
    all_answers = '\n\n'.join(answers)
    final_prompt = summary_prompt.replace('<<SUMMARY>>', all_answers)
    summary_completion_response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=final_prompt,
        temperature=0.6,
        max_tokens=2000,
        top_p=1.0,
        frequency_penalty=0.25,
        presence_penalty=0.0,
        stop=['<<END>>'])
    summary_text = summary_completion_response['choices'][0]['text'].strip()

    write_log('QUESTION: ' + query + '\n\n' + 'ANSWER: ' + summary_text)