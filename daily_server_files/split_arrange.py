import wave
import numpy as np
import os
from pathlib import Path
import pprint
import math
import sys


files = list()
def save_wav_channel(fn, wav, channel):
    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.frombuffer(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tobytes())
    outwav.close()

def create_dir(path, num):
    try:
        directory = str(num)
        folder_path_csr = os.path.join(path, 'csr', directory)
        print(folder_path_csr)
        os.makedirs(folder_path_csr)

        folder_path_customer = os.path.join(path, 'customer', directory)
        os.makedirs(folder_path_customer)
    except Exception:
        print('Folders already exists')


def scan_path(path):
    folder_counter = 0
    create_dir_flag = True
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            name, extension = filename.split('.')
            new_csr_filename = '_'.join([name, 'csr'])
            new_csr_filename = '.'.join([new_csr_filename, extension])
            new_customer_filename = '_'.join([name, 'customer'])
            new_customer_filename = '.'.join([new_customer_filename, extension])
            full_path = {
                'full_path': os.path.join(Path(path),Path(filename)),
                'filename': filename,
                'new_full_path_csr': os.path.join(Path(path), 'csr', str(folder_counter), Path(new_csr_filename)),
                'new_full_path_customer': os.path.join(Path(path), 'customer', str(folder_counter), Path(new_customer_filename))
            }
            files.append(full_path)
            if folder_counter == 15:
                folder_counter = 0
                create_dir_flag = False
            else:
                if create_dir_flag:
                    create_dir(path, folder_counter)
                folder_counter += 1

# Change path X:\\training\\wav\\2022_03_01\\server_0X.

if __name__ == "__main__":
    files = []
    path = sys.argv[1]
    print(path)
    scan_path(path)
    for item in files:
        print(item)
        wav = wave.open(item['full_path'])
        # Customer = channel 0, CSR = channel 1
        save_wav_channel(item['new_full_path_customer'], wav, 0)
        save_wav_channel(item['new_full_path_csr'], wav, 1)
