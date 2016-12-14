WORK=work

all: run

run: dataset
	python --train $(WORK)/snli_1.0/snli_1.0/snli_1.0_train.jsonl --dev $(WORK)/snli_1.0/snli_1.0/snli_1.0_dev.jsonl --test $(WORK)/snli_1.0/snli_1.0/snli_1.0_test.jsonl --word2vec $(WORK)/glove.840B.300d.txt

data: $(WORK)/snli_1.0 $(WORK)/glove.840B.300d.txt

$(WORK)/glove.840B.300d.txt: init
	wget -o $(WORK)/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
	unzip $(WORK)/glove.840B.300d.zip

$(WORK)/snli_1.0: init
	wget -o $(WORK)/snli_1.0.zip http://nlp.stanford.edu/projects/snli/snli_1.0.zip
	unzip $(WORK)/snli_1.0.zip

init:
	pip install -r requirements.txt
	mkdir -p $(WORK)
