import json
import argparse
from tqdm import tqdm
import codecs

from utils_mistral import *

intent_detail_list = read_prompt_csv('client')

def generate_client_intent(context,condition,intent='None',n_turn=0,type='None'):

	if condition == 'DA':
		messages = create_message_client_generation_conditionned_da(intent_detail_list,intent,context)
	elif condition == 'Type':
		messages = create_message_client_generation_conditionned_type(intent_detail_list,type,n_turn,context)
	elif condition == 'Unconditionned':
		messages = create_message_client_generation_unconditionned(intent_detail_list,n_turn,context)
	else:
		print('Error: condition not recognized')
		return None
	response = get_completion_from_messages_local(messages, temperature=0.7)

	return response


parser = argparse.ArgumentParser(description='Behavior Inference')

parser.add_argument('--condition', type=str, default='DA', help='method', choices=['DA', 'Type', 'Unconditionned',])
parser.add_argument('--input_path', type=str, default='dataset/sample_client_input.jsonl', help='Path to input')
parser.add_argument('--output_path', type=str, default='dataset/sample_client_output.jsonl', help='Path to output')
args = parser.parse_args()

condition = args.condition
input_path = args.input_path
output_path = args.output_path

print('Condition: ', condition)
print('Input Path: ', input_path)
print('Output Path: ', output_path)
	
f = codecs.open(input_path, 'r', 'utf-8')
output_f = codecs.open(output_path, 'w', 'utf-8')
df = pd.read_json(input_path, lines=True)
for row in tqdm(f):
	curr_json = json.loads(row.strip())

	n_turn = curr_json['ID'].split('_')[1]
	intent = curr_json['intent']
	type = curr_json['type']
	curr_json['Generated Utterance'] = generate_client_intent(curr_json['context'],condition,n_turn=n_turn,intent=intent,type=type)
	print(curr_json['intent'])
	print(json.dumps(curr_json), file=output_f)

f.close()
output_f.close()