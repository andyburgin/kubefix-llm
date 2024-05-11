import time
import os
import json
from dotenv import load_dotenv

start = time.time()

load_dotenv()

inputPath = "./json/"
jsonPath = "alpaca.json"
documentList = "kubedoclist.txt"

fileDocs = open(documentList,"r")
docList = fileDocs.read().splitlines()
fileDocs.close()

fileOutput = open(jsonPath, "w")
fileOutput.write("[")
firstLine = True
linecount = 0
for filenamepre in docList:
  if not filenamepre.startswith('#'):
    print("Processing doc " + filenamepre + ".json")

    fileInput = open(inputPath + filenamepre + ".json", "r")
    content = json.load(fileInput)
    fileInput.close()
  
    for x in content: 
        print(x)
        output = {
          "instruction" : x['question'],
          "input" : "",
          "output" : x['answer'],
          "source" : "/content/en/docs/" + filenamepre + ".md" 
        }
        
        if not firstLine:
            fileOutput.write(",")
        firstLine = False
        fileOutput.write("\n")
        json.dump(output,fileOutput)
        linecount = linecount + 1

fileOutput.write("\n]")
fileOutput.close()

print(f'linecount {linecount}')
end = time.time()
print(f"\n\n(took {round(end - start, 2)} s.):")

