de = open("ted/spm.train.de-en.de", "r").readlines()
print("Read de")
en = open("ted/spm.train.de-en.en", "r").readlines()
print("Read en")
print(len(de) == len(en))

limit = 0.01

de_red = open("ted/spm." + str(limit) + "train.de-en.de", "w")
en_red = open("ted/spm." + str(limit) + "train.de-en.en", "w")

for i in range(0, int(len(de) * limit)):
    de_red.write(de[i])
    en_red.write(en[i])