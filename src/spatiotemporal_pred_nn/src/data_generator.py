import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

class DataGenerator():
    def __init__(self,):
        self.grid_size = (100, 100)
        self.input_params = []
        self.output_grids = []
        self.input_grids = []
        self.grid = np.zeros((100, 100))

        pass

    # def __len__(self):
    #     return int(np.ceil(len(self.data) / self.batch_size))

    # def __getitem__(self, index):
    #     indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
    #     batch_data = self.data[indices]
    #     batch_labels = self.labels[indices]
    #     return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


    def init_parameters(self):
        self.grid = np.zeros((100, 100))
        self.num_ob = np.random.randint(0, 15)
                
        # Select random indices
        flat_indices = np.random.choice(10000, size=self.num_ob, replace=False)
        self.ob_indices = [(idx // 100, idx % 100) for idx in flat_indices]
        
        for i, j in self.ob_indices:
            self.grid[j, i] = 1
        
        # self.input_params = []
        # self.output_grids = []
        # self.input_grids = []
        
    def generate_data(self):
        self.velocity_list = np.arange(0, 16, 3)
        # self.psi_list = np.arange(0, 2 * np.pi, np.pi/4)
        self.psi_list = np.arange(0, 2 * np.pi, np.pi/12)
        self.t_range = np.arange(5, 26, 5)

        for v in self.velocity_list:
            for psi in self.psi_list:
                vx = v * np.cos(psi)
                vy = v * np.sin(psi)
                for t in self.t_range:
                    grid_out = self.generate_output_grid(vx, vy, t)
                    self.input_params.append([vx, vy, t])
                    # self.input_params.append([v, psi, t])
                    self.output_grids.append(grid_out)
                    self.input_grids.append(self.grid)

        # self.input_params = np.array(self.input_params)
        # self.output_grids = np.array(self.output_grids)
        # self.input_grids = np.array(self.input_grids)
    
    def save_dataset(self):

        path = os.path.join(os.path.dirname(__file__), 'dataset/train_data_250i.npz')

        for _ in range(100):
            print(f'generating data: {_+1}/{100} done')

            self.init_parameters()
            self.generate_data()

        # Convert lists to numpy arrays
        inputs = np.expand_dims(np.array(self.input_grids), axis=-1)
        outputs = np.expand_dims(np.array(self.output_grids), axis=-1)
        params = np.array(self.input_params)
        
        print(f'Input grids shape: {inputs.shape}')
        print(f'Output grids shape: {outputs.shape}')
        print(f'Input parameters shape: {params.shape}')
        
        # Save to .npz file
        np.savez_compressed(path, input_grids=inputs, output_grids=outputs, input_params=params)
        print(f'Dataset saved to {path}')


    def load_dataset(self, path=os.path.join(os.path.dirname(__file__), 'dataset/train_data_250i.npz')):

        data = np.load(path)
        self.input_grids = data['input_grids']
        self.output_grids = data['output_grids']
        self.input_params = data['input_params']

        # Plot a single input and output grid to verify data loading
        idx = np.random.randint(len(self.input_grids))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot input grid
        im = axs[0].imshow(self.input_grids[idx, :, :, 0], cmap='viridis', origin='lower')
        axs[0].set_title('Input Binary Grid')
        fig.colorbar(im, ax=axs[0])
        
        # Plot output grid
        im = axs[1].imshow(self.output_grids[idx, :, :, 0], cmap='viridis', origin='lower')
        axs[1].set_title('Output Grid (Probability)')
        fig.colorbar(im, ax=axs[1])
        
        # Add parameters information
        # vx, vy, t = self.input_params[idx]
        # v = np.sqrt(vx**2 + vy**2)
        # psi = np.arctan2(vy, vx)
        # plt.suptitle(f"Example {idx}: v={v:.2f}, psi={psi:.2f}, t={t}")
        
        plt.suptitle(f"v ={self.input_params[idx][0]:.2f}, psi={self.input_params[idx][1]:.2f}, t={self.input_params[idx][2]:.2f}")
        
        plt.tight_layout()
        plt.show()

    def get_tf_dataset(self, batch_size=64, val_split=0.1):

        path = os.path.join(os.path.dirname(__file__), 'dataset/train_data_250i.npz')
        data = np.load(path)
        
        # input_grids = tf.convert_to_tensor(data['input_grids'], dtype=tf.float32)
        # output_grids = tf.convert_to_tensor(data['output_grids'], dtype=tf.float32)
        # input_params = tf.convert_to_tensor(data['input_params'], dtype=tf.float32)
        
        input_grids = data['input_grids']
        output_grids = data['output_grids']
        input_params = data['input_params']
        
        # Shuffle and split the dataset
        n_samples = len(input_grids)
        val_size = int(n_samples * val_split)
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        
        # train_inputs = input_grids[train_idx]
        # train_params = input_params[train_idx]
        # train_outputs = output_grids[train_idx]
        # val_inputs = input_grids[val_idx]
        # val_params = input_params[val_idx]
        # val_outputs = output_grids[val_idx]
        
        # train_inputs =tf.convert_to_tensor(train_inputs, dtype=tf.float32)
        # train_params = tf.convert_to_tensor(train_params, dtype=tf.float32)
        # train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
        # val_inputs = tf.convert_to_tensor(val_inputs, dtype=tf.float32)
        # val_params = tf.convert_to_tensor(val_params, dtype=tf.float32)
        # val_outputs = tf.convert_to_tensor(val_outputs, dtype=tf.float32)
        
        # train_ds = tf.data.Dataset.from_tensor_slices(((train_inputs, train_params), train_outputs)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # val_ds = tf.data.Dataset.from_tensor_slices(((val_inputs, val_params), val_outputs)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        def generator(indices):
            for i in indices:
                yield (input_grids[i].astype(np.float32), input_params[i].astype(np.float32)), output_grids[i].astype(np.float32)

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_idx),
            output_signature=(
                (tf.TensorSpec(shape=(100, 100, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32)),
                tf.TensorSpec(shape=(100, 100, 1), dtype=tf.float32)
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_idx),
            output_signature=(
                (tf.TensorSpec(shape=(100, 100, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32)),
                tf.TensorSpec(shape=(100, 100, 1), dtype=tf.float32)
            )
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds
        
    def get_train_val_data(self):

        # Initialize parameters and grid
        self.input_grids = []
        self.output_grids = []
        self.input_params = []
        
        print('Generating data')
        # Initialize grid and parameters
        self.init_parameters()
        # Generate data for this instance
        self.generate_data()        
        
        # Generate data for all combinations
        # self.generate_data()
        
        # Convert lists to numpy arrays
        inputs = np.array(self.input_grids)
        outputs = np.array(self.output_grids)
        input_params = np.array(self.input_params)
        
        inputs = np.expand_dims(inputs, axis=-1)  # Add channel dimension for input grid
        outputs = np.expand_dims(outputs, axis=-1)  # Add channel dimension for output grid
        
        # Split data into training and validation sets
        train_size = int(0.9 * len(inputs))
        train_indices = np.random.choice(len(inputs), size=train_size, replace=False)
        val_indices = np.setdiff1d(np.arange(len(inputs)), train_indices)

        train_inputs, train_params, train_outputs = inputs[train_indices], input_params[train_indices], outputs[train_indices]
        val_inputs, val_params, val_outputs = inputs[val_indices], input_params[val_indices], outputs[val_indices]

        # convert to TensorFlow tensors        
        train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
        train_params = tf.convert_to_tensor(train_params, dtype=tf.float32)
        train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
        val_inputs = tf.convert_to_tensor(val_inputs, dtype=tf.float32)
        val_params = tf.convert_to_tensor(val_params, dtype=tf.float32)
        val_outputs = tf.convert_to_tensor(val_outputs, dtype=tf.float32)
        
        return train_inputs, train_params, train_outputs, val_inputs, val_params, val_outputs

    def get_test_data(self, num_samples=1):
        """
        Generate test data with random parameters.
        """
        self.init_parameters()
        
        input_params = []
        output_grids = []
        
        for _ in range(num_samples):
            # Randomly select velocity, angle, and time
            v = np.random.uniform(3, 15)
            psi = np.random.uniform(0, 2 * np.pi)
            t = np.random.randint(1, 25)

            vx = v * np.cos(psi)
            vy = v * np.sin(psi)

            # Generate output grid
            output_grid = self.generate_output_grid(vx, vy, t)
            output_grids.append(output_grid)
            
            # Prepare input grid and parameters
            # input_params.append([vx, vy, t])
            input_params.append([v, psi, t])

        input_grid_tensor = tf.convert_to_tensor(np.expand_dims(self.grid, axis=[0,-1]), dtype=tf.float32)  # Add batch dimension
        input_params_tensor = tf.convert_to_tensor(input_params, dtype=tf.float32)
        output_grid_tensor = tf.convert_to_tensor(np.expand_dims(np.array(output_grids), axis=-1), dtype=tf.float32)  # Add channel dimension

        return input_grid_tensor, input_params_tensor, output_grid_tensor


    def generate_output_grid(self, vx, vy, t):

        if self.num_ob == 0:
            return self.grid.copy()
        gamma = 0.03
        dx = gamma * vx * t
        dy = gamma * vy * t
        sig_x = 0.5 * np.linalg.norm([dx, dy])
        sig_y = 0.2 * np.linalg.norm([dx, dy])
        if np.linalg.norm([dx, dy]) == 0:
            sig_x = 0.1*t
            sig_y = 0.1*t
        theta = np.arctan2(vy, vx)
        D = np.diag([sig_x**2, sig_y**2])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Sig = R @ D @ R.T
        invS = np.linalg.inv(Sig)
        norm = 1/ (2 * np.pi * sig_x * sig_y)

        centers = np.array(self.ob_indices) + np.array([int(dx), int(dy)])

        result = np.zeros((100, 100))
        x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100)
        X, Y = np.meshgrid(x, y)

        min_thresh = 0.01
        local_grid_size = 10

        for ind in centers:
            x0, y0 = ind
            
            # Skip if center is outside grid
            # if x0 < 0 or x0 >= 100 or y0 < 0 or y0 >= 100:
            #     continue
                
            # Calculate grid bounds with bounds checking
            i_min = max(0, int(x0 - local_grid_size//2))
            i_max = min(100, int(x0 + local_grid_size//2 + (local_grid_size % 2)))
            j_min = max(0, int(y0 - local_grid_size))
            j_max = min(100, int(y0 + local_grid_size + (local_grid_size % 2)))
            
            # Skip if window is completely outside the grid
            if i_max <= i_min or j_max <= j_min:
                continue
            
            # Create slices for the local grid
            # i_slice = slice(i_min, i_max)
            # j_slice = slice(j_min, j_max)
            
            # Calculate centered coordinates for local grid
            x_centered = X[j_min:j_max, i_min:i_max] - x0
            y_centered = Y[j_min:j_max, i_min:i_max] - y0

            # to make the local gaussian with finite support
            # x_centered = X - ind[0]
            # y_centered = Y - ind[1]
            local_gaussian = norm * np.exp(-0.5 * (invS[0, 0] * x_centered**2 + invS[1, 1] * y_centered**2 + 
                                             2 * invS[0, 1] * x_centered * y_centered))

            local_gaussian[local_gaussian < min_thresh] = 0

            clipped = (x0-2*sig_x <= 0 or x0-2*sig_x >= 100 or y0-2*sig_y <= 0 or y0-2*sig_y >= 100)
            if np.sum(local_gaussian)> 0 and not clipped:
                local_gaussian = local_gaussian/np.sum(local_gaussian)

            result[j_min:j_max, i_min:i_max] += local_gaussian

        return result

    def check_generated_grids(self):
        """
        Sanity check to visualize and verify the generated grids.
        """
        # Make sure data is generated
        if not hasattr(self, 'input_grids') or len(self.input_grids) == 0:
            print("No data found. Generating data...")
            self.init_parameters()
            self.generate_data()
        
        
        # Sample a few examples to display
        n_samples = min(5, len(self.input_grids))
        sample_indices = np.random.choice(len(self.input_grids), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # cnorm = mcolors.Normalize(vmin=0, vmax=1)
            # Plot input grid (obstacles)
            im = axs[0].imshow(self.input_grids[idx], cmap='viridis', origin='lower')
            axs[0].set_title('Input Binary Grid')
            fig.colorbar(im, ax=axs[0])
            # axs[0].axis('off')
            
            # Plot output grid (probability distribution)
            im = axs[1].imshow(self.output_grids[idx], cmap='viridis', origin='lower')
            axs[1].set_title('Output Grid (Probability)')
            # axs[1].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axs[1])
            
            # Add parameters information
            vx, vy, t = self.input_params[idx]
            v = np.sqrt(vx**2 + vy**2)
            psi = np.arctan2(vy, vx)
            plt.suptitle(f"Example {i+1}: v={v:.2f}, psi={psi:.2f}, t={t}")
            
            plt.tight_layout()
            plt.show()
        
        print(f"Displayed {n_samples} examples from {len(self.input_grids)} generated grids.")    
    
    
def main():
    # Example usage
    data_gen = DataGenerator()
    # data_gen.init_parameters()
    # data_gen.generate_data()
    
    # train_inputs, train_params, train_outputs, val_inputs, val_params, val_outputs = data_gen.get_train_val_data()
    
    # print(f"Train inputs shape: {train_inputs.shape}")
    # print(f"Train params shape: {train_params.shape}")
    # print(f"Train outputs shape: {train_outputs.shape}")
    
    # # Check generated grids
    # data_gen.check_generated_grids()
    
    # data_gen.save_dataset()
    data_gen.load_dataset(path=os.path.join(os.path.dirname(__file__), 'dataset/train_data_250i.npz'))

if __name__ == "__main__":
    main()