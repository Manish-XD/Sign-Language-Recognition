## Sign Language Recognition 🧏🏼‍♂️🤖

A simple sign language recognition model that my team built in our minor project.
<br/>

## Flow of the Code </>

- <code>create_dataset.py</code> collects and preprocesses hand sign images, extracting the hand landmarks. The preprocessed data is serialized into 'data.pickle'.

- <code>train_classifier.py</code> loads the preprocessed data and trains a Random Forest classifier. The trained model is serialized into 'model.p'.

- <code>inference_classifier.py</code> loads the pre-trained model and performs real-time gesture recognition using the MediaPipe library.


