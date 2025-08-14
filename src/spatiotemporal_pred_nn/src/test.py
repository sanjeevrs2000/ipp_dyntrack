import tensorflow as tf
import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import os

def main():
    data_gen = DataGenerator()
    data_gen.init_parameters()
    
    # Load model 
    model_path = os.path.join(os.path.dirname(__file__), 'models/unet_trial_jun14')
    model = tf.keras.models.load_model(model_path)
    
    input_grid, input_params, output_grids = data_gen.get_test_data(1)
    
    pred_grid = model.predict([input_grid, tf.expand_dims(input_params[0], axis=0)])

    # norm = Normalize(vmin=0, vmax=1)
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(tf.squeeze(input_grid), cmap='viridis', origin='lower')
    axes[0].set_title('Input Grid')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(tf.squeeze(output_grids[0]), cmap='viridis', origin='lower')
    axes[1].set_title('True Output Grid')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(tf.squeeze(pred_grid), cmap='viridis', origin='lower')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title('Predicted Output Grid')

    plt.tight_layout()
    plt.suptitle(f'Sample prediction, vx: {input_params[0][0]:.2f}, vy: {input_params[0][1]:.2f}, t: {input_params[0][2]:.2f}')

    plt.show()
    # plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
    

if __name__ == "__main__":
    main()