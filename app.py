from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from deepface import DeepFace

# detector = MTCNN()
model = VGGFace(model=model_name, include_top=include_tops,
                input_shape=(224,224,3), pooling=poolings)
feature_list = pickle.load(open(features_name,'rb'))
filenames = pickle.load(open(pickle_file,'rb'))
