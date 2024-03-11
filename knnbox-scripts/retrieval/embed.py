from sentence_transformers import SentenceTransformer, util
import os
import pickle
import math
from token_stat import SentenceStat, get_mt_model_name
from typing import List
from dotenv import load_dotenv

load_dotenv()

print("Prepare model")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")
file_name = ""
mode = os.getenv('DATASET').lower()

if mode == "ted":
    mode = 3
elif mode == "news":
    mode = 4
elif mode == "qa":
    mode = 2
elif mode == "qe":
    mode = 6
elif mode == "display":
    mode = 7
elif mode == "custom":
    mode = 8
else:
    print("Error: Non existing mode")
    exit()

store_number = int(os.getenv('LAYER'))
if mode == 1:
        file_name = "data/ted_stat_hyp_layer_" + str(store_number)
        #file_name = "data/knn_stats_target_layer_" + str(store_number)
elif mode == 2:
        file_name = "data/premade_anotations_layer_" + str(store_number) + ".bin"
elif mode == 3:
        file_name = "data/standard_annotation"
elif mode == 4:
     file_name = "data/out_of_domain"
elif mode == 5:
     file_name = "data/mt_gender_bin"
elif mode == 6:
    judge_name = "rater1"
    judge_name = "rater4"
    modelname = "Tencent_Translation.1520"
    modelname = "Online-A.1574"
    file_name = "data/wmt_human_" + judge_name + "_" + modelname + ".bin"
elif mode == 7:
    for sub in [0,1,2]:
        mode_name = "random data" if sub == 0 else "training data" if sub == 1 else "non-training data"
        file_name = "data/demonstrate_" + mode_name + ".pyobject"
elif mode == 8:
    file_name = f"data/{get_mt_model_name()}/custom/{os.getenv('DATASTORE_NAME')}_{os.getenv('LAYER')}/{os.getenv('CUSTOM_FILE_NAME')}.bin"


print("Loading file")
if os.path.exists(file_name):
    print("Found existing file")
    with open(file_name, "rb") as f:
        stats: List[SentenceStat] = pickle.load(f)
else:
     print("File not found, exiting")
     exit()
model.cuda()

total = len(stats)

# data = "Make your Linux terminal more useful with tmux, a terminal multiplexer that allows you to run multiple Linux programs over a single connection."
# data = [data] * 1000
# breakpoint()
# test = model.encode(data)
# breakpoint()
if mode == 3 or mode == 4 or mode == 8:
    i = 0
    for key, stat in stats.items():
        src_str_enc = model.encode(stat.src_str)
        for j in range(len(stat.tokens)):
            if len(stat.tokens[j].cos_sims) > 0 and stat.tokens[j].avg_cos_sim != 0:
                continue
            retrieved_enc = model.encode(stat.tokens[j].retrieved_sentences)
            score = util.cos_sim(src_str_enc, retrieved_enc)
            average_sim = math.fsum(score[0]) / len(score[0])
            stat.tokens[j].cos_sims = score[0].tolist()
            stat.tokens[j].avg_cos_sim = average_sim
            # print(j)
        stat.total_avg_cos_sim = math.fsum(map(lambda x: x.avg_cos_sim, stat.tokens))
        stats[key] = stat
        i += 1
        print(str(i) + "/" + str(total) + " " + str(100 * i/total) + "%" , end='\r')
else:
    for i, stat in enumerate(stats):
        src_str_enc = model.encode(stat.src_str)
        for j in range(len(stat.tokens)):
            if len(stat.tokens[j].cos_sims) > 0 and stat.tokens[j].avg_cos_sim != 0:
                continue
            retrieved_enc = model.encode(stat.tokens[j].retrieved_sentences)
            score = util.cos_sim(src_str_enc, retrieved_enc)
            average_sim = math.fsum(score[0]) / len(score[0])
            stat.tokens[j].cos_sims = score[0].tolist()
            stat.tokens[j].avg_cos_sim = average_sim
            # print(j)
        stat.total_avg_cos_sim = math.fsum(map(lambda x: x.avg_cos_sim, stat.tokens))
        print(str(i) + "/" + str(total) + " " + str(100 * i/total) + "%" , end='\r')


with open(file_name, "wb") as f: # "wb" because we want to write in binary mode
    pickle.dump(stats, f)