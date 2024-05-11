import time
import os
import json
from dotenv import load_dotenv

start = time.time()

load_dotenv()

inputPath = "./json/"
jsonPath = "dataset.jsonl"
documentList = "kubedoclist.txt"

fileOutput = open(jsonPath, "w")
fileOutput.close()

fileDocs = open(documentList,"r")
docList = fileDocs.read().splitlines()
fileDocs.close()

for filenamepre in docList:
  if not filenamepre.startswith('#'):
    print("Processing doc " + filenamepre + ".json")

    fileInput = open(inputPath + filenamepre + ".json", "r")
    content = json.load(fileInput)
    fileInput.close()

    for x in content:
        userText = x['question']
        botText = x['answer']
        output = {
          "text" : f"<user>: {userText} <bot>: {botText}"
        }

        fileOutput = open(jsonPath, "a")
        json.dump(output,fileOutput)
        fileOutput.write("\n")
        fileOutput.close()

end = time.time()
print(f"\n\n(took {round(end - start, 2)} s.):")

