
# http://github.com/timestocome

# Working through MEAP Machine Learning w/ TensorFlow Book
# added a few things to their sample code 


# read in wav files, convert to frequencies, notes then histograms and try clustering 




import tensorflow as tf
from tensorflow import py_func
import wave
import numpy as np
import matplotlib.pyplot as plt 




filenames = tf.train.match_filenames_once('./audio_dataset/*.wav')
count_num_files = tf.size(filenames)

filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)



def int_to_note(i):

    if i == 0: return 'A'
    if i == 1: return 'A#'
    if i == 2: return 'B'
    if i == 3: return 'C'
    if i == 4: return 'C#'
    if i == 5: return 'D'
    if i == 6: return 'D#'
    if i == 7: return 'E'
    if i == 8: return 'F'
    if i ==9: return 'F#'
    if i == 10: return 'G'
    if i == 11: return 'G#'




# http://www.liutaiomottola.com/formulae/freqtab.htm
# only converting each to a note, not a scale, this is just a toy problem
def frequency_to_pitch(frequencies):
    
    notes = []
    for f in frequencies:
    
        A = abs(f % 27.5000)
        A_sharp = abs(f % 29.135)
        B = abs(f % 30.868)
        C = abs(f % 16.351)
        C_sharp = abs(f % 17.324)
        D = abs(f % 18.354)
        D_sharp = abs(f % 19.445)
        E = abs(f % 20.601)
        F = abs(f % 21.827)
        F_sharp = abs(f % 23.124)
        G = abs(f % 24.499)
        G_sharp = abs(f % 25.956)

        pitches = [A, A_sharp, B, C, C_sharp, D, D_sharp, E, F, F_sharp, G, G_sharp]
        closest_match = np.argmax(pitches)
        notes.append(closest_match)
    
    return notes






def read_wav(filename):
    
    # convert filename to usable format    
    filename = filename.strip().decode('ascii')

    if 'cough' in filename: target = 1
    if 'scream' in filename:  target = 0


    with wave.open(filename, 'rb') as wr:
    
        sz = wr.getframerate()                  # process one second
        da = np.fromstring(wr.readframes(sz), dtype=np.int16)

        # split da into 1/10th sec fragments
        n_da = len(da) // 4410
        notes = []
        for i in range(n_da):
            begin = i * 4410
            finish = begin + 4410
            data = da[begin : finish]
            f = np.absolute(np.fft.rfft(data))   # get frequency
            
            # get pitch
            notes = frequency_to_pitch(f)

        return notes, target



k = 2 # in this case we know we only have 2 coughs and 3 scream files
max_iterations = 100

# pick random samples to use as intitial centroids
# also tried picking random values from all centroids 
def initial_cluster_centeroids(X, k): 

    c = []
    for i in range(k):
        z = np.random.randint(k)
        c.append(X[i])

    return c


# assign sample to closest cluster
def assign_to_cluster(X, centroids):  

    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)

    return mins
    


def recompute_centroids(X, Y):

    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)

    return sums / counts



with tf.Session() as sess:

    # init everything
    tf.global_variables_initializer().run()
    
    # read files into separate threads
    num_files = sess.run(count_num_files)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    # convert each sound file to notes
    # convert file name to 0 for scream, 1 for cough
    samples = []
    targets = []
    histograms = []
    for i in range(num_files):

        audio_file = sess.run(filename)
        notes, target = read_wav(audio_file)

        histogram, bins = np.histogram(notes)
        samples.append(notes)
        targets.append(target)
        histograms.append(histogram)        
    
    # cluster 
    centroids = initial_cluster_centeroids(histograms, k)
    #centroids = initial_cluster_centeroids(samples, k)
    centroids = np.asarray(centroids, dtype='float64')
    tf.cast(centroids, tf.float64)

    x = np.asarray(histograms, dtype='float64')
    #x = np.asarray(samples, dtype='float64')
    tf.cast(x, tf.float64)


    i, converged = 0, False 
    while not converged and i < max_iterations:
        i += 1
        y = assign_to_cluster(x, centroids)
        centroids = sess.run(recompute_centroids(x, y))


    # check predictions
    print("Centroids")
    print(centroids)

    print("Check groups ")
    print("Created groups", y.eval())
    print("Actual groups", targets)

    # split samples into targets
    group0 = []
    group1 = []
    for i in range(len(x)):
        if targets[i] == 0: group0.append(x[i])
        if targets[i] == 1: group1.append(x[i])



    # sanity check clusters 
    # get average for each value 
    group0 = np.asarray(group0)
    group0_mean = group0.mean(0)
    print(group0_mean)

    group1 = np.asarray(group1)
    group1_mean = group1.mean(0)
    print(group1_mean)



    # clean up threads
    coord.request_stop()
    coord.join(threads)

sess.close()