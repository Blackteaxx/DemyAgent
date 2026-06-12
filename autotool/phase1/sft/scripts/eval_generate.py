import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Evaluation script for LLaMA-Factory")
  parser.add_argument('--model', type=str, required=True, help='Path to the model')
  # parser.add_argument('--data', type=str, required=True, help='Path to the evaluation data')
  # parser.add_argument('--output', type=str, required=True, help='Path to save the output')
  return parser.parse_args()

args = parse_args()


def main():
  # Load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    )

  print(model)

  # Load evaluation data
  # with open(args.data, 'r') as f:
    # eval_data = f.readlines()
    
  eval_data = """\nYou are given a problem and a toolset. Solve the problem step by step by selecting tools to use during your thinking. \n\nHere is the description of the toolset\n\n## Toolset\n\n- ToolName: Code Interpreter (CI)\n**Description:** Tool for code generation for mathematical calculations, data analysis, or programming tasks.\n**Utialization Guideline:** \nWrite executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. \nThe Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with \n`<code>\n```python\ncode snippet\n```\n</code>`\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n- ToolName: Search Engine (SE)\n**Description:** Tool for searching the web for relevant information.\n**Utialization Guideline:**\nWrite the search content wrapped with `<search>search content</search>`. \nThe search engine will return relevant information (wrapped in `<search>\nsearch output_str\n</search>`) to aid your reasoning and help you arrive at the final answer.\n\n\n- ToolName: Multimodal (MM)\n**Description:** Tool for analyzing images, videos, or other media content.\n**Utialization Guideline:**  \nExtracts any text from an image (such as axis labels or annotations). If no text is present, returns\nan empty string. Note: the text may not always be accurate or in order.\nArguments: {\"image\": \"the image from which to extract text\"}\nReturns: {\"text\": \"the text extracted from the image\"}\nExamples: {\"name\": \"OCR\", \"arguments\": {\"image\": \"img1\"}}\n\n## Format Instruction\n\n- Tool Selection Instruction\nBefore you decide to invoke and execute a tool, you need to first think which tool is most appropriate for the current step. And write down your reasoning following the FORMAT:\n<select>\nYour reasoning on why select the specific tool(s) based on previous reasoning steps...\n<tool>Your selected tool name</tool>\n</select>\n\n- Output Instruction\nRemember to place the final answer in the last part using the FORMAT: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n\n## Question:\n\nThere are 152 students at Dala High School. Assume the following:  \n- 100 students take a Math class  \n- 94 students take a Science class  \n- 57 students take an English class  \n- 73 students take a Math class and a Science class  \n- 24 students take a Math class and an English class  \n- 27 students take a Science class and an English class  \n- 22 students take a Math class and a Science class and an English class\n  \nHow many students take neither a Math class nor a Science class nor an Eglish class?\n\nNow, output your response according to the above instructions below:"""

  # Open output file
  with open(args.output, 'w') as out_file:
    for line in eval_data:
      inputs = tokenizer(line, return_tensors='pt')
      outputs = model.generate(**inputs, max_length=512)
      generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      out_file.write(generated_text + '\n')
      
      
if __name__ == "__main__":
  main()