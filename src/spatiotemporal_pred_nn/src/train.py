import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from data_generator import DataGenerator
from unet import UNet

epochs = 100
batch_size = 64

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main():
    # Set random seed for reproducibility
    # np.random.seed(42)
    # tf.random.set_seed(42)
       
    model = UNet(input_size=(100, 100, 1), num_params=3)    
    model.summary()
    
    data_gen = DataGenerator()
    
    # Compile the model
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.95
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # # Train the model

    train_dataset, val_dataset = data_gen.get_tf_dataset(batch_size=batch_size, val_split=0.1)
    
    history = model.fit(train_dataset,
              validation_data=val_dataset,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
              epochs=epochs,verbose=True)
    
    # Save the model
    path = os.path.join(os.path.dirname(__file__), 'models')
    model_name = 'trial_model'
    print(f'Saving model to {os.path.join(path, model_name)}')
    save_dir = os.path.join(path, model_name)
    model.save(save_dir)
    
    # Plot results
    plot_results(history.history['loss'], history.history['mae'], history.history['val_mae'], save_dir)

    input_grid, input_params, output_grids = data_gen.get_test_data(3)
    
    for i in range(len(output_grids)):

        pred_grid = model.predict([input_grid, tf.expand_dims(input_params[i], axis=0)])

        # norm = Normalize(vmin=0, vmax=1)
        # Plot the results
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(input_grid), cmap='viridis')
        plt.colorbar()
        plt.title('Input Grid')
        
        plt.subplot(1, 3, 2)
        plt.imshow(tf.squeeze(output_grids[i]), cmap='viridis')
        plt.colorbar()
        plt.title('True Output Grid')
        
        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(pred_grid), cmap='viridis')
        plt.colorbar()
        plt.title('Predicted Output Grid')
        
        plt.tight_layout()
        plt.suptitle(f'Sample prediction, vx: {input_params[i][0]:.2f}, vy: {input_params[i][1]:.2f}, t: {input_params[i][2]:.2f}')
        # plt.show()
        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))

def plot_results(losses, mae, val_mae, save_dir):
    """
    Plot training history
    """
    
    # avg_loss, avg_mae, avg_val_mae = [], [], []
    # interval = 2

    # for i in range(len(losses)-interval):
    #     avg_loss.append(np.mean(losses[i:i+interval]))
    #     avg_mae.append(np.mean(mae[i:i+interval]))
    #     avg_val_mae.append(np.mean(val_mae[i:i+interval]))

    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Training MAE')
    plt.plot(val_mae, label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.tight_layout()
    # plt.show()
    
if __name__ == "__main__":
    main()