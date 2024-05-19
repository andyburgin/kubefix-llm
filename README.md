# Overview

The objective of this repo is to share the tools, method and learning for generating a Q&A dataset from the Kubernetes website documentation that is suitable for finetuning an LLM model intended for use with [K8sGPT](https://k8sgpt.ai/) for fault analysis and resolution.

Please note - the dataset and resultant model should be considered highly experimental and used with caution, use at your own risk.

## TL;DR

The dataset in alpaca format can be found on huggingface [andyburgin/kubefix](https://huggingface.co/datasets/andyburgin/kubefix)

The quantised 4 bit model in GGUF format can be found on huggingface [andyburgin/Phi-3-mini-4k-instruct-kubefix-v0.1-gguf](https://huggingface.co/andyburgin/Phi-3-mini-4k-instruct-kubefix-v0.1-gguf)

Reminder - the dataset and resultant model should be considered highly experimental and used with caution, use at your own risk.

## The Aims

This repo was primarily a learning exercise to test if finetuning a model is a worthwhile exercise compared to using just the base model, or in conjunction with a RAG based approach.

The dataset is generated from a subset of the Kubernetes documentation from the [English markdown files](https://github.com/kubernetes/website/tree/main/content/en/docs). The Q&A pairs have been generated from the documents using an opensource model (to avoid licencing issues for some free models or SaasS services) - after much trial and error the [openchat-3.5-0106](https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF) model was found to be the least problematic.

Ultimately the resulting LLM is intended to be self-hosted in a GPU free environment running under [local-ai](https://localai.io/basics/kubernetes/) in Kubernetes, therefore a small parameter model were needed, initial tests with TinyLamma proved less than useful so the [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) model was chosen as a base image - fortunately this was available as a base image for training with [unsloth](https://unsloth.ai/introducing)

Although originally intended to not use GPU for either dataset generation or finetuning it soon became apparent that was impractical, therefore Google Collab was used for the creation of both.

# Method

## Which Documents ?

The list of the documents to be included is held in (kubedoclist.txt)[kubedoclist.txt], this lists all the files under the /content/en/docs folder of the Kubernetes website repo including the full path (but missing the .md extension) - some of the lines are commented out with a `#`, these aren't included in the data set because:

* They are command line tool references.
* API references.
* Glossary items.
* Tutorials that link out to 3rd party sites.

## The Collab Notebook

The bulk of the dataset generation will be done by the [k8s qna generation.ipynb](k8s qna generation.ipynb) notebook. However, be warned there is "a lot" of manual fixing to do - improvements to automate that welcome.

### Start the Collab Notebook

Run through the cells/steps:

* Install Prerequisite models, libraries and data.
  * Download openchat model.
  * Install lamma-cpp-python library.
  * Install LangChain libraries.
* Clone k8s website.
* Create folder structure.
* Upload the (kubedoclist.txt)[kubedoclist.txt] file.

### Run Q and A Generation

The notebook cell will loop through each document and generate a bunch of q&a pairs in a json file, the json file will be named the same as the input file (but with a `.json` extension) and the parent file structure will be created inside of the `out` folder.

A special note of thanks https://helixml.substack.com/p/how-we-got-fine-tuning-mistral-7b?utm_source=pocket_saves for their excellent work on finetuning and sharing their experiences, I've been heavily influenced by their prompts. 

Next, sit back and relax as the files are created - until it goes wrong - sorry working with LLMs in early 2024 isn't a straightforward experience (probabilistic vs deterministic system etc).

The generation will fail in "interesting" ways.
* It may produce no output.
* The output produced may not be `json`.
* It may go into a loop of regenerating the same Q&A pairs.
* It may start producing random nonsense.
* It may just fail with a run time error.

In the event of a failure make a note of the document and add it to [kubedoclist-fix.txt](kubedoclist-fix.txt),
in your (kubedoclist.txt)[kubedoclist.txt] comment out any processed files (and the failed one), re-upload and re-run the script continuing from where it left off until the next failure.

It may also be worth using the "Archive files and download qanda.tgz" step to create a tgz of the current `out` folder and download the resulting file in case the whole collab needs restarting (that can sometimes happen).

Eventually, you'll have processed all the files in (kubedoclist.txt)[kubedoclist.txt], there's still work to do as we need to deal with the ones you've complied into [kubedoclist-fix.txt](kubedoclist-fix.txt).

As before create the file archive and download it - those files will need unpacking into your local `json` folder for later.

### Fixing the Files that Failed

Looking at your version of [kubedoclist-fix.txt](kubedoclist-fix.txt) you'll probably have quite a few entries. Usually these are quite long documents (1000 lines plus) and I suspect this is causing the original generation to fail. So locate the original `md` files from the [markdown files](https://github.com/kubernetes/website/tree/main/content/en/docs) and split them into roughly 200 line chunks and update the [kubedoclist-fix.txt](kubedoclist-fix.txt) file.

e.g. failing file
```
concepts/services-networking/service
```
to 200 line chunks named
```
concepts/services-networking/service.1
concepts/services-networking/service.2
concepts/services-networking/service.3
concepts/services-networking/service.4
concepts/services-networking/service.5
```

..and repeat for each of the failed docs.
### Generating Q and A from the Fixed Docs

Upload your [kubedoclist-fix.txt](kubedoclist-fix.txt) file and the split markdown files to the correct place in the website folder structure. Now you can run the "generate q and a pairs from fixed documents" cell to generate the json.

When all have been generated, archive and download as before. Next, by hand merge each of the split files into one json file placed into the correct folder in the `json` folder - just like the output if the original step had worked in the first place.

## Nearly Done
Yep, you may want to take a well-earned rest after creating all those json files, but did anyone say they would be valid json ? did you trust an LLM ? well bad news, although you may have files that look like json they probably aren't syntactically correct. Let's use some code to fix that.

Spin up a python docker container or use the virtual python environment of choice, then install the python requirements
```
pip install -r requirements.txt
```
Included in the repo is a `jsonl-gen.py` script this will create a file called `dataset.jsonl` by reading your (kubedoclist.txt)[kubedoclist.txt] file (so revert any comment lines you've added those failed docs as you've now fixed them) and load the json in your `json` folder.
```
python jsonl-gen.py
```

When you run this script it will handily error and tell you the file and location where your json is invalid, you'll probably need to fix, typically by:
* Escaping double quotes - so going from `"` to `\"`.
* Joining multilines, so adding \n and joining lines.
* Sometimes additional "example" or "explanation" fields will be added - you'll need to remove these or edit them into the answer field.
* Annoyingly additional `,` may be added after the "answer" field, you'll need to remove these.

### Again, Again
OK, so my json is valid? yep, but does it have any duplicates? yes, it probably does, try using this handy command to find any duplicates:
```
cat dataset.jsonl ! sort | uniq -c
```
This will output the count of duplicates and the line - use your favourite editor/IDE to search and remove the duplicates.

FYI the `dataset.jsonl` can be used by the together.ai finetuning service, in the end, I didn't use that but the tools and data are useful, who knows in the future they might be used for other finetuning methods.

#### Now Let's Finally Create that Dataset
Let's make the `alpaca.json` file you've always wanted.
```
python alpaca-gen.py
```

### At last a Dataset

Upload it to your hugging face account and then finetune a model with it.

# FineTuning

As explained earlier the use case for the resultant model was to run under local-ai in Kubernetes in a GPU free environment, therefore we need a small fast model that is compatible with local-ai. I chose the [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) model as a base image - fortunately this was available as a base image for training with [unsloth](https://unsloth.ai/introducing)

## Launch the unsloth Collab

Head to the [unsloth github page](https://github.com/unslothai/unsloth) and choose the "Phi-3 (3.8B)" free Notebook.

You'll need to make a couple of changes.

### Change 1 - the Dataset
Change the dataset to load yours, e.g.
```
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
```
...to...
```
dataset = load_dataset("andyburgin/kubefix", split = "train")
```
### Change 2 - Epocs...
The notebook is set to `max_steps = 60,` remove that and replace with `num_train_epochs=1`

### Change 3 - Test Prompts
There are a few tests that use `fibonnaci sequence`, change these to `How do I` and `restart a Kubernetes pod`

### Change 4 - Merge
We are just interested in the final model so make sure you `load the LoRA adapters` by changing the `False` to `True` in that step.

### Change 5 - Saving the Model
Probably should create the model file after all this hard work, so change `False` to `True` in the step:
```
# Save to q4_k_m GGUF
if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

## Opps, Failure saving Notebook
At the time of writing there's a known issue with the Phi3 model and saving to q4_k_m. Hopefully, this will be fixed soon in unsloth, but in the meantime here's a workaround to add to the notebook if you hit the problem, add a new code cell to the notebook

### Install Llamacpp binary and Tools
```
! git clone --recursive https://github.com/ggerganov/llama.cpp llamafix
! cd llamafix && make clean && LLAMA_CUDA=1 make all -j
! cd llamafix && pip install -r requirements.txt
! cd llamafix && python convert.py /content/model --pad-vocab --outtype f16 --outfile /content/model.fp16.bin
! cd llamafix && ./quantize /content/model.fp16.bin /content/model.q4_k_m.gguf q4_k_m
```
### Download the Model, The Upload

Download the file model.q4_k_m.gguf, rename it Phi-3-mini-4k-instruct-kubefix-v0.1.q4_k_m.gguf and upload to huggingface.

# Still TODO

There's a whole bunch of testing and evaluation to do with the dataset and model, but there's the initial release for evaluation.
