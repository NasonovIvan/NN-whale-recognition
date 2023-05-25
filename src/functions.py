import imports as im

# mean data
def mean_data(data):
    mean = data[:].mean(axis=0)
    data -= mean
    std = data[:].std(axis=0)
    data /= std
    return data

# normalize images
def norm_images(data):
    return data[:] / 255.0

# plot spectrogram func
def PlotSpecgram(P, freqs, bins):
    Z = im.np.flipud(P) # flip rows so that top goes to bottom, bottom to top, etc.
    xextent = 0, im.np.amax(bins)
    xmin, xmax = xextent
    extent = xmin, xmax, freqs[0], freqs[-1]

    img = im.pl.imshow(Z, extent=extent)
    im.pl.axis('auto')
    im.pl.xlim([0.0, bins[-1]])
    im.pl.ylim([0, 1000])

# Reads the frames from the audio clip and returns the uncompressed data
def ReadAIFF(file):
    s = im.aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return im.np.fromstring(strSig, im.np.short).byteswap()

# creating small good spectrograms from wavs for article whales data
def create_images_data(path_train_audio, path_train_img, width=122, height=168):
    onlywavfiles = [f for f in im.listdir(path_train_audio) if im.isfile(im.join(path_train_audio, f))]
    LENGTH=256
    for file in onlywavfiles:
        whale_sample_file = path_train_audio + file

        fs, x = im.read(whale_sample_file) 
        f, t, Zxx_first = im.signal.stft(x, fs=fs, window=('hamming'), nperseg=LENGTH, noverlap=int(0.875*LENGTH), nfft=LENGTH)

        Zxx = im.np.log(im.np.abs(Zxx_first))

        px = 1/im.plt.rcParams['figure.dpi']

        fig = im.plt.figure(figsize=(width*px, height*px), frameon=False) # 180, 190 (139x146) for Kaggle dataset
        ax1 = im.plt.subplot()
        ax1.pcolormesh(t, f, Zxx, cmap='viridis')
        ax1.set_axis_off()
        im.plt.savefig(path_train_img + file[0:-4:1] + '.png', bbox_inches='tight', pad_inches=0, dpi = 100)
        fig.clear()
        im.plt.close(fig)

# create Kaggle dataset
def CreateKaggleDataset(df, train_index, val_index, path_train_img, slice_name=5):
    '''
    CreateKaggleDataset function

    --Input--
    df (DataFrame): train.csv file from Kaggle
    train_index (int): how many samples should be in training set
    val_index (int): how many samples should be in validation set
    path_train_img (str): path to the images (clear or noised)
    slice_name (int): number, which mean image file name in path (5 for kaggle and 4 for article)

    --Output--
    x_train (array)
    y_train (array)
    x_val (array)
    y_val (array)
    x_test (array)
    y_test (array)
    '''
    x_train = []
    y_train = []

    x_val = []
    y_val = []
    
    x_test = []
    y_test = []
    for i in range(len(df["clip_name"])):
        FILENAME = path_train_img + df["clip_name"][i][:-slice_name] + '.png' # slice_name=5 for kaggle and slice_name=4 for article
        rgba_image = im.Image.open(FILENAME)
        img = rgba_image.convert('RGB')
        img_arr = im.np.asarray(img)
        rgba_image.close()
        #print(FILENAME)

        if i < train_index:
            x_train.append(img_arr)
            y_train.append(df["label"][i])
        elif i < val_index:
            x_val.append(img_arr)
            y_val.append(df["label"][i])
        else:
            x_test.append(img_arr)
            y_test.append(df["label"][i])
    return im.np.array(x_train), im.np.array(y_train), im.np.array(x_val), im.np.array(y_val), im.np.array(x_test), im.np.array(y_test)

# for plotting loss and accuracy history training
def PlotLossAcc(TrainData, ValData, Epochs, TrainLabel, ValLabel, yLabel, title, ColTrain, ColVal, filename):
    pdf = im.PdfPages(filename)
    fig, ax = im.plt.subplots(figsize=(8, 6))
    ax.plot(Epochs, TrainData, color=ColTrain, label=TrainLabel)
    ax.plot(Epochs, ValData, color=ColVal, label=ValLabel)
    im.plt.title(title)
    ax.set_ylabel(yLabel)
    ax.set_xlabel("Epochs")
    im.plt.legend()
    pdf.savefig(fig)
    pdf.close()
    im.plt.show()

# function for making h/c train and test from wavs
def compute_spectral_info(spectrum):
    N = len(spectrum)
        
    H_p = 0
    H_q = 0
    Complexity_sq = 0
    Complexity_jen = 0
    Complexity_abs = 0
    
    yf = spectrum
        
    Sum = sum(yf) 
    square_sum = 0
    abs_sum = 0

    p_is = []
    for s in yf:
        
        p_i = s/Sum
        if (p_i > 0): 
            p_is.append(p_i)
            H_p += -p_i*im.np.log2(p_i)
        
    Nfft = len(p_is)
    q_i = 1.0/Nfft # Noise spectrum

    for k in range(Nfft):
        square_sum += (p_is[k] - q_i)**2
        abs_sum += im.np.abs(p_is[k] - q_i)

    Disequilibrium_sq = square_sum


    H_p /= im.np.log2(Nfft)

    Jensen = im.jensenshannon(p_is, [q_i for j in range(Nfft)])

    Q0 = -2.0/((Nfft+1)*im.np.log2(Nfft+1)/Nfft - 2*im.np.log2(2*Nfft)+im.np.log2(Nfft))
    
    Disequilibrium_jen = (Jensen**2)*Q0
    
    Complexity_sq = H_p*square_sum
    
    Complexity_jen = H_p*Disequilibrium_jen #H_p*square_sum #H_p*(Jensen**2)*Q0 ##H_p*square_sum ##np.exp(H_p)*square_sum # H_p*(0.5*square_sum*len(p_is) - 1.0*third_sum*(len(p_is)**2)/6)
    
    Complexity_abs += H_p*(abs_sum**2)/4

    return H_p, Complexity_sq, Complexity_jen, Complexity_abs

# create Complexity-Entropy datasets, based on the compute_spectral_info function
def create_HC_dataset_wavs(df, train_index, val_index, path_train, Noised=False):
    x_train = []
    y_train = []

    x_val = []
    y_val = []
    
    x_test = []
    y_test = []

    progress_counter_1 = 0
    progress_counter_2 = 10
    
    for i in range(len(df["clip_name"])):
        progress_counter_1 += 1

        if Noised:
            FILENAME = path_train + 'noised_002_' + df["clip_name"][i][:-5] + '.wav'
        else:
            FILENAME = path_train + df["clip_name"][i][:-5] + '.wav'
            # FILENAME = path_train + df["clip_name"][i]
        
        H_s = []
        C_sqs = []
        C_jsds = []
        C_tvs = []
        
        WINDOW_FFT = 256
        WINDOW_HOP = 32
        y_librosa, sr_librosa = im.librosa.load(FILENAME, sr=None)
        y_librosa -= im.np.mean(y_librosa)
        M = im.librosa.feature.melspectrogram(y=y_librosa, sr=sr_librosa, htk=True,
                                              hop_length = WINDOW_HOP, n_fft = WINDOW_FFT,
                                              n_mels=64, fmin = 50, fmax = 300)
        M_db = im.librosa.power_to_db(M, ref=im.np.min)**2

        # for j in range(10, 85):
        for j in range(126):
            local_spect = M_db[:, j]
    
            H, C_sq, C_jsd, C_tv = compute_spectral_info(local_spect)
            H_s.append(H)
            C_sqs.append(C_sq)
            C_jsds.append(C_jsd)
            C_tvs.append(C_tv)
            
        hc_plane = ((H_s, C_sqs, C_jsds, C_tvs))

        if i < train_index:
            x_train.append(hc_plane)
            y_train.append(df["label"][i])
        elif i < val_index:
            x_val.append(hc_plane)
            y_val.append(df["label"][i])
        else:
            x_test.append(hc_plane)
            y_test.append(df["label"][i])

        if (progress_counter_1 / len(df["clip_name"])) * 100 >= progress_counter_2:
            print((progress_counter_1 / len(df["clip_name"])) * 100, '% completed')
            progress_counter_2 += 10
    
    return im.np.array(x_train), im.np.array(y_train), im.np.array(x_val), im.np.array(y_val), im.np.array(x_test), im.np.array(y_test)

# compute Complexity-Entropy data from wav-file with ordpy
def compute_hc_from_ordpy(FILENAME):

    audio_signal, RATE = im.librosa.load(FILENAME, sr=None)
    audio_signal -= im.np.mean(audio_signal)

    # audio_signal -= int(np.mean(audio_signal))

    window_size = 256

    H_ps = []
    Complexitys = []
    Fishers = []

    Hs_my = []
    Complexities_sq_my = []
    Complexities_jen_my = []
    Complexities_tv_my = []

    start_point = 0
    end_point = 0

    i = 0
    
    while (start_point < len(audio_signal)):

        end_point = start_point + window_size

        frame_audio = audio_signal[start_point : end_point : 1]

        start_point = start_point + 32

        HC = im.ordpy.complexity_entropy(frame_audio, dx = 4)

        (patterns, probs) = im.ordpy.ordinal_distribution(frame_audio, dx = 4)

        H_my = 0
        Disequlibrium_sq_my = 0
        Disequlibrium_jen_my = 0
        Disequlibrium_tv_my = 0
        N = len(probs)
        q_i = 1.0/N

        for prob in probs:
            H_my += -prob * im.np.log2(prob)
            Disequlibrium_sq_my += (prob - q_i)**2
            Disequlibrium_tv_my += im.np.abs(prob - q_i)

        Disequlibrium_jen_my = im.jensenshannon(probs, [q_i for j in range(N)])

        H_my /= im.np.log2(N)

        Hs_my.append(H_my)
        Complexities_sq_my.append(H_my * Disequlibrium_sq_my)
        Complexities_jen_my.append(H_my * (Disequlibrium_jen_my**2))
        Complexities_tv_my.append(H_my * (Disequlibrium_tv_my**2))

        #HF = ordpy.fisher_shannon(frame_audio, dx = 3)

        H_ps.append(HC[0])
        Complexitys.append(HC[1])
        #Fishers.append(HF[1])

        i += 1
    
    H_ps = H_ps[10:85:1]
    Complexitys = Complexitys[10:85:1]
    Complexities_sq_my = Complexities_sq_my[10:85:1]
    Complexities_jen_my = Complexities_jen_my[10:85:1]
    Complexities_tv_my = Complexities_tv_my[10:85:1]
    return H_ps, Complexitys, Complexities_sq_my, Complexities_jen_my, Complexities_tv_my

# creating train, validation, test datasets with ordpy function
def create_hc_dataset_ordpy(df, path_train, train_index, val_index):
    labels = []
    for label in df["label"]:
        labels.append(int(label))
    labels = im.np.array(labels)

    HC_list = []

    progress_counter_1 = 0
    progress_counter_2 = 10

    for i in range(len(df["clip_name"])):
        HC_tmp = []
        progress_counter_1 += 1
        # FILENAME = path_train + df["clip_name"][i]
        FILENAME = path_train + df["clip_name"][i][:-5] + '.wav'
        H_ps, Complexitys, Complexities_sq_my, Complexities_jen_my, Complexities_tv_my = compute_hc_from_ordpy(FILENAME)
        HC_tmp.append(H_ps)
        HC_tmp.append(Complexitys)
        HC_tmp.append(Complexities_sq_my)
        HC_tmp.append(Complexities_jen_my)
        HC_tmp.append(Complexities_tv_my)
        HC_list.append(HC_tmp)

        if (progress_counter_1 / len(df["clip_name"])) * 100 >= progress_counter_2:
            print((progress_counter_1 / len(df["clip_name"])) * 100, '% completed')
            progress_counter_2 += 10

    HC_array = im.np.array(HC_list)
    hc_train = HC_array[:train_index]
    hc_y_train = labels[:train_index]
    hc_val = HC_array[train_index:val_index]
    hc_y_val = labels[train_index:val_index]
    hc_test = HC_array[val_index:]
    hc_y_test = labels[val_index:]
    
    return hc_train, hc_y_train, hc_val, hc_y_val, hc_test, hc_y_test
