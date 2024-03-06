import sentencepiece as spm

spm.SentencePieceTrainer.train(input="europarl/train.de-en.de,europarl/train.de-en.en",
                                model_prefix="bpe",
                                vocab_size=10000)

# print('Finished training sentencepiece model.')

spm_model = spm.SentencePieceProcessor(model_file="bpe.model")

for partition in ["train"]:
    for lang in ["de", "en"]:
        f_train = open(f"europarl/spm.train.de-en.{lang}", "w")
        f_dev = open(f"europarl/spm.dev.de-en.{lang}", "w")
        f_test = open(f"europarl/spm.test.de-en.{lang}", "w")
        with open(f"europarl/{partition}.de-en.{lang}", "r") as f_in:
            for line_idx, line in enumerate(f_in.readlines()):
                # Segmented into subwords
                line_segmented = spm_model.encode(line.strip(), out_type=str)
                # Join the subwords into a string
                line_segmented = " ".join(line_segmented)
                if line_idx < 1000000:
                    f_train.write(line_segmented + "\n")
                elif line_idx < 1002500:
                    f_dev.write(line_segmented + "\n")
                elif line_idx < 1007500:
                    f_test.write(line_segmented + "\n")
                else:
                    break
        f_train.close()
        f_test.close()
        f_dev.close()

# for partition in ["train", "dev", "test"]:
#     for lang in ["de", "en"]:
#         f_train = open(f"europarl/spm.train.de-en.{lang}", "w")
#         f_dev = open(f"europarl/spm.dev.de-en.{lang}", "w")
#         f_test = open(f"europarl/spm.test.de-en.{lang}", "w")
#         with open(f"europarl/{partition}.de-en.{lang}", "r") as f_in:
#             for line_idx, line in enumerate(f_in.readlines()):
#                 # Segmented into subwords
#                 line_segmented = spm_model.encode(line.strip(), out_type=str)
#                 # Join the subwords into a string
#                 line_segmented = " ".join(line_segmented)
#                 if line_idx < 1000000:
#                     f_train.write(line_segmented + "\n")
#                 elif line_idx < 1002500:
#                     f_dev.write(line_segmented + "\n")
#                 elif line_idx < 1007500:
#                     f_test.write(line_segmented + "\n")
#                 else:
#                     break
#         f_train.close()
#         f_test.close()
#         f_dev.close()