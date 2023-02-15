import gensim
import pandas as pd
import sys
import logging

if __name__ == '__main__':

    try:
        input_file = sys.argv[1]
    except Exception as e:
        print("Error: Supply input file as first argv param")
        sys.exit(1)

    df = pd.read_json(input_file, lines=True)
    print(df)
    df.shape
    try:
        input_data = df.context.apply(gensim.utils.simple_preprocess, min_len=2, max_len=50)
    except Exception as e:
        print("`context` field not found in JSON data")
        print(e)
        sys.exit(1)
    logging.info("Done reading data file!")

    # build vocabulary and train model
    model = gensim.models.fasttext.FastText(
        sg=1, # use skip-gram
        vector_size=25,
        window=2,
        min_count=1,
        epochs=10,
        workers=10)

    model.build_vocab(input_data, progress_per=1000)
    model.train(input_data, total_examples=len(input_data), epochs=model.epochs)
    gensim.models.fasttext.save_facebook_model(model, "fast-fb-model.bin")
