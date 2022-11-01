import json

if __name__ == "__main__":
    path1 = 'corpus.zh'
    path2 = 'corpus.ms'
    lines1 = []
    lines2 = []

    with open("./mini_train.txt", "r", encoding="utf-8") as file:
        corpus = file.readlines()
    with open("./mini_dev.txt", "r", encoding="utf-8") as file:
        corpus2 = file.readlines()
        corpus = corpus + corpus2
    for item in corpus:
        item = item.replace('\n', '').strip()
        arr = item.split('\t')
        if len(arr) != 2:
            continue
        lines1.append(arr[0] + '\n')
        lines2.append(arr[1] + '\n')

    with open(path1, "w", encoding="utf-8") as fch:
        fch.writelines(lines1)

    with open(path2, "w", encoding="utf-8") as fen:
        fen.writelines(lines2)

    # lines of Chinese: 252777
    print("lines1: ", len(lines1))
    # lines of English: 252777
    print("lines2: ", len(lines2))
    print("-------- Get Corpus ! --------")
