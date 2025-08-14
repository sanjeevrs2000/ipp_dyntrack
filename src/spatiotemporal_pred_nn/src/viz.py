import visualkeras
import tensorflow as tf
from tensorflow.keras.models import Model
import os

def main():
    # Load your model
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'models/unet_trial_jun14'))

    # Visualize the model architecture
    visualkeras.layered_view(model, to_file='model_architecture.png', legend=True)
    
    
if __name__ == "__main__":
    main()