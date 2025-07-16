### Heading

Tool used to deploy a detection model for the purposes of doing inventory detection. Initial dataset is comprised purely of vegetables with future goals to source, tag, and train the model to detect and count the number of items

The value this provides is due to grocers and stores occasionally requiring the need to have an individual scan the store to see if items are damaged, stolen, and additional inventory exists. This product aims to speed up the ability to first detect the number of items, future goals would be to extend ability to determine damaged items


### Run Locally

TODO: Add into a public S3 bucket a simple trained model

With the pytorch trained info (*.pt) placed into a `/models` directory within the `api` directory. In essence (/api/models/*.pt). To build the model to be used and utilized by a FastAPI server you can run 

```bash
docker build -t {image_name_here}:{image_tag_here} .
docker run --name {container_name_here} -p 8000:8000 {image_name_from_above_line}:{image_tage_from_above_line}
```

And then within your browser open a tab with the URL: `localhost:8000` to see your container

NOTE: The `/` root does not have anything as there's nothing to return from the FastAPI server

To see that your container is running you can add `/health` to your URL (e.g. localhost:8000/health) and it will return a response back

You may also use the auto-generated Swagger docs by adding `/docs` to your URL (e.g. localhost:8000/docs) to test out the `/predict` endpoint

The `/predict` endpoint is intended to take in an image of some inventory that the model is trained on and return the number of items detected, their class, confidence score, and bounding box coordinates

TODO: Currently the model is only trained on three items (carrots, beans, and radishes)