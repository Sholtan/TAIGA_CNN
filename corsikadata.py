print("corsika_data module imported\n")
import struct
import numpy as np

import matplotlib.pyplot as plt

import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, indices):
        self.data = data
        self.labels = labels
        self.indices = indices    # Store only indices, not full sliced arrays

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]   # Access original data lazily
        #return torch.from_numpy(self.data[real_idx]), torch.from_numpy(self.labels[real_idx])
        return self.data[real_idx], self.labels[real_idx]

class CorsikaData:
    header_size = 180                # 4 ints + 8 doubles + 1 int
    pixel_amplitude_size = 28   # 3 ints + 2 doubles
    def __init__(self):
        self.data = np.empty((0, 1, 27, 27), dtype = np.float32)
        self.labels = np.empty((0), dtype = np.int_)
        print("after init data shape", self.data.shape)
        print("after init labels shape", self.labels.shape)

    def load(self, files):
        for file_name in files:
            self._load_single_file_(file_name)
        print("loading files done\n\n")

    def _load_single_file_(self, file_name):
        print("loading file", file_name)
        event_counter = 0

        events = []        
        labels = []
        with open(file_name, "rb") as file_in_bytes:
            header_chunk = file_in_bytes.read(CorsikaData.header_size)
            
            while header_chunk:
                N_run, N_scattering, N_telescope, N_photoelectrons = struct.unpack('<4i', header_chunk[0:16])
                energy, theta, phi, x_core, y_core, z_core, h_1st_interaction, particle_type, xmax, hmax, x_telescope, y_telescope, z_telescope, x_offset, y_offset, theta_telescope, phi_telescope, delta_alpha, alpha_pmt, T_average = struct.unpack('<20d', header_chunk[16:176])
                N_pixels, = struct.unpack('<i', header_chunk[176:180])

                tmp_event = np.zeros((27, 27)) 
                event_size = 0   
                for i in range(N_pixels):
                    pixel_chunk = file_in_bytes.read(CorsikaData.pixel_amplitude_size)
                    amplitude, row_number, column_number = struct.unpack('<3i', pixel_chunk[0:12])
                    average_time, std_time = struct.unpack('<2d', pixel_chunk[12:28])


                    tmp_event[row_number+13, column_number+13] = amplitude
                    event_size += amplitude
                if event_size > 120:
                    events.append(tmp_event)
                    if np.isclose(particle_type, 1.):
                        labels.append(1)
                    elif np.isclose(particle_type, 14.):
                        labels.append(0)
                    else:
                        raise ValueError("particle_type is not gamma or proton")


                event_counter+=1
                if event_counter%30000==0:
                    #print("reading event: ", event_counter)
                    pass

                header_chunk = file_in_bytes.read(CorsikaData.header_size)
        #print("number of events in file:", event_counter)
        #print("after size cut:", len(events))

        assert len(events) == len(labels)

        ######################################################################
        print("\nbefore concatenate data shape", self.data.shape)
        print("\nbefore concatenate labels shape", self.labels.shape)
        self.data = np.concatenate([self.data, np.expand_dims(np.array(events, dtype = np.float32), axis=1)])
        self.labels = np.concatenate([self.labels, np.array(labels, dtype = np.int_)])
        print("after concatenate labels data", self.data.shape, '\n')
        print("after concatenate labels shape", self.labels.shape, '\n')
        #*********************************************************************

        #for i, event in enumerate(events):
        #    self.data[i, :, :] = event

        is_good = []
        for i in range(len(self.data)):
            is_good.append(self._is_image_good_(self.data[i][0]))

        self.data = self.data[is_good]
        self.labels = self.labels[is_good]
        print("after goodness cut data shape", self.data.shape)
        print("after goodness cut labels shape", self.labels.shape)
        #print("data shape", self.data.shape)
        #print("labels shape", self.labels.shape)
        #print("labels[:20]",labels[:20])
        print("data.dtype:", self.data.dtype)
        print("labels.dtype:", self.labels.dtype, '\n\n')





    def _is_image_good_(self, random_image):
        mask = random_image > 7
        center = mask[1:-1, 1:-1]
        if np.any(center & mask[0:-2, 0:-2]):  # upper-left
            return True
        if np.any(center & mask[0:-2, 1:-1]):  # up
            return True
        if np.any(center & mask[0:-2, 2:]):    # upper-right
            return True
        if np.any(center & mask[1:-1, 0:-2]):    # left
            return True
        if np.any(center & mask[1:-1, 2:]):      # right
            return True
        if np.any(center & mask[2:, 0:-2]):      # lower-left
            return True
        if np.any(center & mask[2:, 1:-1]):      # down
            return True
        if np.any(center & mask[2:, 2:]):        # lower-right
            return True
        return False


    def create_data_loader(self, batch_size: int = 128, train_portion: float = 0.8, shuffle_seed: int = 42):
        if len(self.data) != len(self.labels):
            raise ValueError(f"data and labels have different lengths: {len(self.data)} and {len(self.labels)}")

        rng = np.random.default_rng(shuffle_seed)
        permut = rng.permutation(len(self.labels))   # used for initial random distribution of training and testing datasets

        train_portion_index = int(train_portion * len(self.labels))

        #train_loader = torch.utils.data.DataLoader(self.data[permut][:train_portion_index], batch_size = batch_size, shuffle=False, num_workers=2)
        #train_labels_loader = torch.utils.data.DataLoader(self.labels[permut][:train_portion_index], batch_size=batch_size, shuffle=False, num_workers=2)

        #test_loader = torch.utils.data.DataLoader(self.data[permut][train_portion_index:-1], batch_size=batch_size, shuffle=False, num_workers=2)
        #test_labels_loader = torch.utils.data.DataLoader(self.labels[permut][train_portion_index:-1], batch_size=batch_size, shuffle=False, num_workers=2)

        train_indices = permut[:train_portion_index]
        test_indices = permut[train_portion_index:]

        train_dataset = CustomDataset(self.data, self.labels, train_indices)
        test_dataset = CustomDataset(self.data, self.labels, test_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  
        # With shuffle=True, the DataLoader randomizes the order of data at the beginning of each epoch. 
        # This means that even though the training set is initially in a random order, 
        # each epoch will see a different mini-batch composition.
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        # shuffle=False for reproducibility of the testing
        print("create_data_loader done")
        return train_loader, test_loader


