import torch
from PIL import Image
from transformers import TextStreamer
import numpy as np
import random
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import sys
import csv

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# Redirect the console output results to a.log file
sys.stdout = Logger('Tiny_Transfer_Attack_Owl2_25train_PGD_16_600_A.log', sys.stdout)
device = torch.device('cuda:0')

random_number = random.randint(1, 2000)
random.seed(random_number)
np.random.seed(random_number)
torch.manual_seed(random_number)

model_path = '../mplug-owl2-llama2-7b'
"""query process"""
test_num = 100
test_rnd_idx = [7, 8, 11, 17, 18, 22, 26, 27, 28, 38, 39, 41, 52, 57, 64, 72, 77, 81, 84, 85, 98, 105, 107, 127, 141, 144, 153, 154, 156, 170, 175, 177, 185, 188, 193, 197, 211, 213, 216, 220, 223, 236, 238, 242, 248, 249, 254, 258, 259, 264, 266, 269, 272, 275, 281, 286, 299, 301, 302, 306, 308, 316, 323, 331, 339, 342, 343, 344, 348, 349, 362, 363, 364, 366, 369, 373, 377, 379, 380, 385, 388, 400, 406, 408, 409, 412, 425, 432, 439, 443, 448, 451, 452, 462, 466, 468, 469, 471, 474, 475]
test_goal_index = test_rnd_idx
print("test_goal_index:", test_goal_index)
test_goal = []
test_target = []
rr = 0
with open('./dataset/advbench/harmful_behaviors_2.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if rr in test_rnd_idx:
            test_goal.append(row['goal'])
            test_target.append(row['target'])
        rr += 1

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)


"""Image process"""
image_file = './images/train_size_25_PGD_16_600_iters_A_pure_noise.png' # Image Path
image = Image.open(image_file).convert('RGB')
max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
image = image.resize((max_edge, max_edge))

image_tensor = process_images([image], image_processor)
image_tensor = image_tensor.to(model.device, dtype=torch.float16)
for i in range(test_num):
    print("==============================" + str(i) + "====================================")
    print('----------Response Goal ' + str(test_goal_index[i]) + ':' + test_goal[i] + '--------')
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles
    inp = DEFAULT_IMAGE_TOKEN + test_goal[i]
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 1.0
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
