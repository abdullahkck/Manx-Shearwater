import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np

matplotlib.use('Agg')  # No pictures displayed


def convert_calls_to_spectograms(directory):
    count = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".flac"):
                createSpectrogram(directory, filepath, file)
                count = count + 1
    print(str(count) + ' spectrogtams are created.')


def createSpectrogram(directory, filepath, file):
    sig, fs = librosa.load(filepath)

    # make pictures name
    pre, ext = os.path.splitext(file)
    save_path = directory + 'Spectrograms' + os.sep + pre + '.png'

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()


def main():
    directory = '/Users/abdullahkucuk/Desktop/MSc/MSc Project/records.nosync/calls_12_14/'
    convert_calls_to_spectograms(directory)


main()
