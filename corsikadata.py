print("corsika_data module imported\n")
import struct
import numpy as np

import matplotlib.pyplot as plt

class CorsikaData:
    header_size = 180                # 4 ints + 8 doubles + 1 int
    pixel_amplitude_size = 28   # 3 ints + 2 doubles
    def __init__(self):
        pass

    def load(self, file_name):
        event_counter = 0

        Nc = []
        Nr = []
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


                    Nc.append(row_number)
                    Nr.append(column_number)
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
        print("number of events in file:", event_counter)
        print("after size cut:", len(events))

        assert len(events) == len(labels)

        self.data = np.zeros((len(events), 27,27))
        for i, event in enumerate(events):
            self.data[i, :, :] = event

        is_good = []
        for i in range(len(self.data)):
            is_good.append(self._is_image_good_(self.data[i]))

        self.data = self.data[is_good]
        self.labels = np.array(labels)[is_good]
        print("data shape", self.data.shape)
        print("labels shape", self.labels.shape)
        print("labels[:20]",labels[:20])
        print("loading files done\n\n")

        return self.data

    def _is_image_good_(self, random_image):
        moves = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        image_is_good = False

        for i in range(1, len(random_image)-1):
            for j in range(1, len(random_image)-1):
                if (random_image[i,j] > 7):
                    for move in moves:
                        if random_image[i+move[0], j+move[1]] > 7:
                            image_is_good = True
                            #print("found", i, j)
                if (image_is_good):
                    break
            if (image_is_good):
                break
        return image_is_good





