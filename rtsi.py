import numpy as np
import librosa

class RTSI():
    def __init__(self, win_length=1024, hop_length=256, eps=1e-11):
        assert win_length / hop_length == win_length // hop_length, "pyrtsi does not support win_length that is not an integer multiply of hop_length"

        self.win_length  = win_length
        self.hop_length  = hop_length
        self.eps         = eps # a small non-zero value to prevent division by zero

        self.window      = np.hanning(win_length)
        self.r_overlap   = win_length // hop_length

        self.asymwindow  = self.get_asymwindow()
        self.div_factor  = np.mean(self.window**2) * self.r_overlap

        self.init_curr_window()

    def pad_for_hop_length(self, audio):
        """
        pad zeroes to the right such that len(audio) is an integer multiply of hop_length
        """
        if len(audio) / self.hop_length != len(audio) // self.hop_length:
            audio = np.concatenate([audio, np.zeros(self.hop_length-len(audio)%self.hop_length)], axis=0) # pad for integer amount of frames
        return audio

    def pad_for_shape_preserving_inverse(self, audio):
        """
        pad zeroes to the right such that spect_to_audio(audio_to_spect(audio)) returns audio of same shape
        provided that the len(audio) is an integer multiply of hop_length
        """
        audio = self.pad_for_hop_length(audio)
        audio = np.concatenate([audio, np.zeros(self.win_length-self.hop_length)], axis=0)
        return audio

    def audio_to_spect(self, audio, pad_for_hop_length=True, pad_for_shape_preserving_inverse=False):
        """
        obtain the magnitude spectrogram of the wave
        """
        if pad_for_shape_preserving_inverse:
            audio = self.pad_for_shape_preserving_inverse(audio)
        elif pad_for_hop_length:
            audio = self.pad_for_hop_length(audio)

        spect = np.abs(librosa.stft(audio, win_length=self.win_length, n_fft=self.win_length, hop_length=self.hop_length, center=False, window='hann'))

        return spect

    def init_curr_window(self):
        """
        Initialize the running window randomly
        This running window will collect all past frames that overlap with the current frame
        The phase of next phase is estimated from the phase of the running window
        """
        self.curr_window = np.random.randn(1, self.win_length) * 1e-9 * np.flip(self.asymwindow, axis=0)

    def get_asymwindow(self):
        """
        self.curr_window will naturally have more mass to the left; because all past frames that overlap with it are to the left, the future frames to the right are assumed zero
        The asymwindow will correct natural assymetr
        The asymmetry of this window corrects the natural asymmetry to obtain a symmetric frame to estimate phase from
        """
        asymwindow = sum(
            [
                np.concatenate(
                    [
                        np.zeros(self.win_length-(i*self.hop_length)),
                        self.window[:i*self.hop_length] ** 2
                    ], axis=0
                ) for i in range(1, self.r_overlap)
            ]
        )

        return asymwindow

    def shift_frame(self, x):
        """
        By cropping off one hop_length to the left and padding zeros to the right this function shifts the 'focus' of what frame is 'current' by one hop_length
        """
        return np.concatenate([x[:, self.hop_length:], np.zeros((1, self.hop_length))], axis=1)

    def spect_to_audio(self, spect):
        audio_out = np.zeros(spect.shape[-1]*self.hop_length)
        for i in range(spect.shape[-1]):

            cmplx = np.fft.rfft(self.curr_window * self.asymwindow)
            cmplx = (cmplx * spect[:, i]) / np.clip(np.abs(cmplx), a_min=self.eps, a_max=None)
            curr_window_ = np.fft.irfft(cmplx)

            self.curr_window += curr_window_ * self.window # overlap-and-add

            if np.array_equal(self.curr_window, np.zeros_like(self.curr_window)): # prevent forever all-zero output after a single all-zero spect column
                self.init_curr_window()

            audio_out[i*self.hop_length : (i+1)*self.hop_length] = self.curr_window[0, :self.hop_length] / self.div_factor

            self.curr_window = self.shift_frame(self.curr_window)

        return audio_out
