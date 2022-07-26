# Identity Verification with PyTorch & Google's FaceNet 
I spent most of the winter and spring '22 researching, prototyping and testing various approaches to building a solution that could be used for identity verification, where you would take a known photo of someone and compare it to a photo taken as they attempt to enter a high security area. This repo is meant to serve as "lab notes" and a canned demo/starting point for building something like this for someone else in the future, I.e. if nothing else this will save me from having to go through various directories on dev box eight months from now, trying to figure out what version is the "the one". The basic structure of an inferencing container for identity verification is here and I plan to add more features and variants in the coming weeks, but for now it's just a basic setup I can use as a starting point on future projects.  



## Acknowledgements 

This project is heavily influenced by [Tim Esler's Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch) a PyTorch based implementation of Google's Facenet research paper, which is in turn heavily influenced by [David Sandberg's TensorFlow implementation](https://github.com/davidsandberg/facenet) of same. 


## How does it work? 
**The short answer:** 
* It's a Flask API and Gunicorn based wrapper around the [Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch), with a number of functional and machine learning based enhancements to make an easy to depoloy/use facial identity solution. The solution is built based on the presumption that the user already has a method for capturing photos and presenting them to the API. 
* This isn't production ready (yet), as it would need some additional work around the server (load balancing, proxy), logging, decoupling the front end API endpoints from the ML components and ideally would be fine tuned for the population/persons it was being used for. 
* You can run the API locally on your  machine either by building the docker container and then running the docker container, or from the command line via: gunicorn  --bind 0.0.0.0:6000  wsgi:app
* The server will run on port 6000, but you can specify any port you like, as long as it isn't already in use. E.g., on OSX Port 5000 is reserved for Airplay  
* Once the API is running you can either ping the endpoints with custom code or use Postman to test it. If you use Postman:
    * Body --> Form Data --> select files under key, and then use the "select files" button to select your image file 

**The long answer:**
* presenting two photos to the /photo_match endpoint via a POST operation returns a JSON with details on the photo match 
* Presenting pre-generated tensors of the reference photo and a recent sample photo to the cached_photo endpoint does the same as the above 
* If more than one face is detected in the sample photo, the solution will attempt to match each face in the photo to the reference photo 
* The default matching method is cosine distance, plan to add Euclidian distance in the near future 
* The "distance" features are provided via built in functions of PyTorch 
* Matches are calculated via taking a pair of embeddings and calculting the "distance" between them using either cosine distance or Euclidian distance. 
* The default matching threshold for cosine distance is <= 0.4


## Technical Details - What's Under The Hood: 
* Facenet-PyTorch is deployed via a Flask and Gunicorn API based Docker Inferencing container OR you can run the gunicorn server directly on your machine IF it's a Unix based OS like OSX or Ubuntu, Windows users will need to run the docker container to use this. 
* MTCNN hyper-parameters were tweaked to increase face detection accuracy 
* PyTorch's built in functions for cosine and Euclidan distance were used to calculate photo similarity. 
* An ideal implementation would use reference photos for which embeddings have already been generated, and then compare those photos to a sample photo. 
* You can run as many workers as your hardware can support for the gunicorn/entrypoint server, but be aware that this would create separate instances of the ML models, which would take up a lot of memory. I.e. for a production level instance you'd need to decouple the front end that receives photos from the ML piece, but it depends on how many matches you're doing per a given unit of time. 
* This was built on a MacBook running OSX, but I also tested it on Ubuntu 20.04 


## Technical Features 
* While cosine distance is the default option for determining if two photos match, the score_service.py script has the code for Euclidian distance. In a future iteration I'll add the option of specifying what match method you want or to try them both in the API call. 



### API Details 
   * /ping just returns an "OK" if the API is running properly 
   * Hitting the /photo_match endpoint with a pair of photos (reference or known photo vs a sample photo) returns a JSON with the following information:
        * Match success (yes/no)
        * Cosine distance, where <0.4 = a match
        * Face Detection probabilities for the face detection part of the detection pipeline
        * Inferencing latency: how long did it take to match the photos 
    * the /cached_data endpoint allows you to pass a photo along with a set of tensors for a cached/pre-processed reference photo 



### Included files 
* Run the docker file to create the docker container, 
* Docker Container + Flask API
    * server.py runs the Flask API 
    * photo_inferencing.py - contains the code that runs the models 
    * score_service.py calculates the distance or similarity scores 



## References 
* [Tim Esler's Facenet-PyTorch Repo](https://github.com/timesler/facenet-pytorch) 
* [Machine Learning Mastery - How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) 
