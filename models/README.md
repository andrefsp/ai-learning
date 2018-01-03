## Linear model

To train the linear model using gcloud

    $ gcloud ml-engine local train --module-name linear.train_task --package-path linear  -- --train-file linear/data/linear.train.csv --job-dir ./linear/serve/
    

It will save the model on a `./linear/serve/` folder.


You can than run predictions using `saved_model_cli`

    $ saved_model_cli run --dir ./linear/serve/ --tag_set serve --signature_def serving_default --input_exprs='X=np.float32(4)'


You can also use google gcloud tools to run the prediction.

    $ gcloud ml-engine local predict --model-dir=./linear/serve/ --json-instances linear/instances.json
